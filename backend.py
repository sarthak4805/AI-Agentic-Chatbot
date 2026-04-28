from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from langchain_core.tools import tool
from newsapi import NewsApiClient
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import struct
import os
import sqlite3
import requests
import time

load_dotenv()

# ---------- SIMPLE EMBEDDINGS ----------
class SimpleEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vector = [0.0] * 384
        for word in words:
            h = hashlib.md5(word.encode()).digest()
            vals = struct.unpack("96f", h * 24)
            for i, v in enumerate(vals[:384]):
                vector[i] += v
        norm = sum(x**2 for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

embeddings = SimpleEmbeddings()

# ---------- SIMPLE VECTOR STORE ----------
class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = []

    def add_documents(self, docs):
        for doc in docs:
            self.chunks.append(doc)
            self.vectors.append(embeddings._embed(doc.page_content))

    def similarity_search(self, query: str, k: int = 3):
        if not self.vectors:
            return []
        query_vec = embeddings._embed(query)
        scores = []
        for i, vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(query_vec, vec))
            scores.append((dot, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]

# ---------- LLMs ----------
# Gemini for tool calling
# Replace your tool-calling LLM with a Groq model that handles tools well
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "llama3-groq-70b-8192-tool-use-preview"
    api_key=os.getenv("GROQ_API_KEY"),
)
# Groq for simple answers - higher free limits
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)



# ---------- GLOBAL VECTOR STORE ----------
vectorstore = None

def load_documents(file_paths: list) -> str:
    global vectorstore
    try:
        all_docs = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        all_docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path, "page": i}
                        ))
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path}
                ))

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_documents(all_docs)
        vectorstore = SimpleVectorStore()
        vectorstore.add_documents(chunks)
        return f"✅ {len(all_docs)} document(s) loaded and indexed successfully!"

    except Exception as e:
        return f"Failed to load documents: {str(e)}"


# ---------- TOOLS ----------

@tool
def search_documents(query: str) -> str:
    """
    Search through uploaded documents to answer questions.
    Use this ONLY when user explicitly mentions document, uploaded file, pdf, or says 'from provided docs'.
    """
    global vectorstore
    try:
        if vectorstore is None:
            return "No documents uploaded yet. Please upload documents from the sidebar first."
        results = vectorstore.similarity_search(query, k=2)
        if not results:
            return "No relevant content found in the documents."
        output = ""
        for i, doc in enumerate(results):
            output += f"[Chunk {i+1}]:\n{doc.page_content}\n\n"
        return output
    except Exception as e:
        return f"Document search failed: {str(e)}"


@tool
def send_job_application(recipient_email: str, role: str) -> str:
    """
    Send a job application email with resume attached.
    Use this ONLY when user explicitly asks to send a job application or apply for a job.
    Extract recipient_email and role from the user message.
    """
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        resume_path = os.getenv("RESUME_PATH", "resume.pdf")

        subject = f"Application for {role} Position"
        body = f"""Dear Hiring Manager,

I hope this email finds you well. I am writing to express my interest in the {role} position at your organization.

I have attached my resume for your consideration. I would welcome the opportunity to discuss how my skills and experience align with your requirements.

Thank you for your time and consideration. I look forward to hearing from you.

Best regards,
{sender_email}
"""
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if os.path.exists(resume_path):
            with open(resume_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(resume_path)}"
                )
                msg.attach(part)
        else:
            return f"Resume file not found at: {resume_path}. Please upload your resume first."

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        return f"Job application for '{role}' sent successfully to {recipient_email} with resume attached!"

    except Exception as e:
        return f"Failed to send email: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Get latest news and current events using NewsAPI.
    Use this for news, sports scores, IPL, cricket, current events, recent updates.
    """
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=5
        )
        results = []
        for article in articles["articles"]:
            title = article.get("title", "")
            source = article["source"]["name"]
            description = article.get("description", "")
            published = article.get("publishedAt", "")[:10]
            if title and description:
                results.append(f"- [{published}] {title} ({source})\n  {description}")
        return "\n\n".join(results) if results else "No news found"
    except Exception as e:
        return str(e)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """
    Perform basic arithmetic.
    Use this ONLY when user gives explicit numbers and asks for a math result.
    operation must be exactly one of: add, sub, mul, div
    """
    try:
        if operation == "add":
            return str(first_num + second_num)
        elif operation == "sub":
            return str(first_num - second_num)
        elif operation == "mul":
            return str(first_num * second_num)
        elif operation == "div":
            if second_num == 0:
                return "Error: Division by zero"
            return str(first_num / second_num)
        else:
            return "Invalid operation. Must be one of: add, sub, mul, div"
    except Exception as e:
        return str(e)


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for any company.
    Use this ONLY when user asks for stock price of a company.
    IMPORTANT: parameter name is 'symbol' not 'ticker'.
    Convert company names to ticker symbols: Tesla→TSLA, Apple→AAPL, Google→GOOGL,
    Microsoft→MSFT, Amazon→AMZN, Netflix→NFLX, Meta→META, Nvidia→NVDA
    Call with symbol='TSLA' not ticker='TSLA'
    """
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={os.getenv('FINNHUB_API_KEY')}"
        response = requests.get(url).json()
        current_price = response.get("c")
        if current_price and current_price != 0:
            return f"{symbol.upper()} current stock price is ${current_price}"
        return f"Could not fetch price for {symbol}. Check if the ticker symbol is correct."
    except Exception as e:
        return f"Could not fetch price for {symbol}: {str(e)}"


# ---------- TOOL LIST ----------
tools = [web_search, calculator, get_stock_price, send_job_application, search_documents]


llm_with_tools = llm.bind_tools(tools)

# ---------- STATE ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------- SYSTEM PROMPT ----------
system_prompt = SystemMessage(content="""You are a helpful AI assistant with access to tools.

TOOL USAGE RULES:
- 'calculator' → ONLY when user gives explicit numbers and asks for math result
- 'get_stock_price' → ONLY when user asks for stock price. Use parameter 'symbol' e.g. symbol='TSLA'
- 'web_search' → ONLY for news, current events, sports, IPL, cricket scores, recent updates
- 'send_job_application' → ONLY when user explicitly asks to send job application with email
- 'search_documents' → ONLY when user says "from the document", "from uploaded file", "from provided docs", "in the pdf"

ANSWER DIRECTLY WITHOUT ANY TOOL for:
- Who is X, What is X, Tell me about X → use your training knowledge
- History, biography, definitions, explanations → use your training knowledge
- Greetings (hi, hello, how are you) → respond naturally
- General advice, coding help, opinions → respond directly

After using a tool:
- NEWS → short summary + bullet points
- DOCUMENT → extract exact answer from chunks
- EMAIL → short success confirmation
- STOCK → show price clearly
- CALCULATOR → show result only
- Never mention tool names or explain your process
""")

# ---------- RETRY HELPER ----------
def invoke_with_retry(llm_instance, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                wait_time = 30 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries reached. Please try again later.")

# ---------- NODES ----------
def chat_node(state: ChatState):
    messages = state["messages"]

    from langchain_core.messages import ToolMessage, HumanMessage as HM, AIMessage as AI

    # stop infinite loop
    if isinstance(messages[-1], ToolMessage):
        if "no documents uploaded" in messages[-1].content.lower():
            return {"messages": [AI(content="No documents are uploaded yet. Please upload a document from the sidebar under 📚 Documents (RAG) and click 'Load Documents'.")]}

    # find last human message
    last_human_idx = 0
    last_human_content = ""
    for i, m in enumerate(messages):
        if isinstance(m, HM):
            last_human_idx = i
            last_human_content = m.content.lower()

    # trim context
    start_idx = max(0, last_human_idx - 1)
    trimmed_messages = messages[start_idx:]

    tool_intents = {
        "stock": any(k in last_human_content for k in ["stock", "share price", "stock price"]),
        "calc": any(k in last_human_content for k in ["calculate", "what is 2", "add ", "subtract", "multiply", "divide", " + ", " - ", " * ", " / "]),
        "search": any(k in last_human_content for k in ["news", "latest", "current", "today", "ipl", "cricket", "match", "score", "recent", "update"]),
        "email": any(k in last_human_content for k in ["send application", "send my resume", "apply for job", "send email to", "send mail to"]),
        "document": any(k in last_human_content for k in ["from the document", "from provided", "in the pdf", "uploaded file", "from the file", "from docs", "in the document"]),
    }

    intents_count = sum(tool_intents.values())

    # no tool needed → use Groq (saves Gemini quota)
    if intents_count == 0:
        try:
            response = invoke_with_retry(llm_groq, [system_prompt] + trimmed_messages)
        except Exception:
            response = invoke_with_retry(llm, [system_prompt] + trimmed_messages)
        return {"messages": [response]}

    # tool needed → use Gemini (better tool calling)
    try:
        response = invoke_with_retry(llm_with_tools, [system_prompt] + trimmed_messages)
    except Exception:
        response = invoke_with_retry(llm_groq, [system_prompt] + trimmed_messages)

    if intents_count <= 1 and hasattr(response, "tool_calls") and len(response.tool_calls) > 1:
        response.tool_calls = response.tool_calls[:1]

    return {"messages": [response]}

tool_node = ToolNode(tools)

# ---------- MEMORY ----------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)

# auto cleanup - keep only last 50 checkpoints
try:
    cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
    count = cursor.fetchone()[0]
    if count > 50:
        conn.execute("""
            DELETE FROM checkpoints
            WHERE thread_id || '-' || checkpoint_id NOT IN (
                SELECT thread_id || '-' || checkpoint_id
                FROM checkpoints
                ORDER BY checkpoint_id DESC
                LIMIT 50
            )
        """)
        conn.commit()
except:
    pass

checkpointer = SqliteSaver(conn=conn)

# ---------- GRAPH ----------
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=None,
)

# ---------- THREADS ----------
def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)