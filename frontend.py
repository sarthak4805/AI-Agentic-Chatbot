import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import chatbot, retrive_all_threads, load_documents
import backend
import uuid
import os
import tempfile

# ---------- SESSION STATE SETUP ----------

if "current_chat" not in st.session_state:
    threads = retrive_all_threads()
    if threads:
        st.session_state.current_chat = threads[0]
    else:
        st.session_state.current_chat = str(uuid.uuid4())

if "resume_path" not in st.session_state:
    st.session_state.resume_path = None

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

if "loaded_doc_names" not in st.session_state:
    st.session_state.loaded_doc_names = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ← restore vectorstore from session state on every rerun
if st.session_state.vectorstore is not None:
    backend.vectorstore = st.session_state.vectorstore

# ---------- SIDEBAR UI ----------

st.sidebar.title("Conversations 🤖")

if st.sidebar.button("➕ New Chat"):
    st.session_state.current_chat = str(uuid.uuid4())
    st.rerun()


threads = retrive_all_threads()
for chat_id in threads:
    if st.sidebar.button(f"Chat {chat_id[:6]}"):
        st.session_state.current_chat = chat_id
        st.rerun()

st.sidebar.divider()

# ---------- RESUME SECTION IN SIDEBAR ----------

st.sidebar.header("📄 Resume")

if st.session_state.resume_path is None:
    st.sidebar.info("Upload your resume once for job applications.")
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_uploader")

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        resume_path = os.path.join(temp_dir, uploaded_file.name)
        with open(resume_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.resume_path = resume_path
        os.environ["RESUME_PATH"] = resume_path
        st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
        st.rerun()
else:
    st.sidebar.success(f"✅ Resume: {os.path.basename(st.session_state.resume_path)}")
    if st.sidebar.button("🔄 Update Resume"):
        st.session_state.resume_path = None
        os.environ["RESUME_PATH"] = ""
        st.rerun()

st.sidebar.divider()

# ---------- DOCUMENTS SECTION (RAG) ----------

st.sidebar.header("📚 Documents (RAG)")

uploaded_docs = st.sidebar.file_uploader(
    "Upload Documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="doc_uploader"
)

if uploaded_docs:
    if st.sidebar.button("📥 Load Documents"):
        file_paths = []
        temp_dir = tempfile.mkdtemp()

        for doc in uploaded_docs:
            doc_path = os.path.join(temp_dir, doc.name)
            with open(doc_path, "wb") as f:
                f.write(doc.read())
            file_paths.append(doc_path)

        with st.spinner("Loading and indexing documents..."):
            result = load_documents(file_paths)

        # ← save vectorstore in session state so it survives reruns
        st.session_state.vectorstore = backend.vectorstore
        st.session_state.docs_loaded = True
        st.session_state.loaded_doc_names = [doc.name for doc in uploaded_docs]
        st.sidebar.success(result)

if st.session_state.docs_loaded:
    st.sidebar.success(f"✅ {len(st.session_state.loaded_doc_names)} doc(s) ready")
    for name in st.session_state.loaded_doc_names:
        st.sidebar.caption(f"📄 {name}")
    if st.sidebar.button("🗑️ Clear Documents"):
        st.session_state.docs_loaded = False
        st.session_state.loaded_doc_names = []
        st.session_state.vectorstore = None
        backend.vectorstore = None
        st.rerun()

st.sidebar.divider()
st.sidebar.divider()
st.sidebar.header("⚙️ Settings")
if st.sidebar.button("🗑️ Clear All Chat History"):
    import os
    if os.path.exists("chatbot.db"):
        os.remove("chatbot.db")
    st.session_state.current_chat = str(uuid.uuid4())
    st.success("✅ Chat history cleared!")
    st.rerun()

# ---------- MAIN CHAT WINDOW ----------

st.title("Chatbot 🤖")

config = {
    "configurable": {
        "thread_id": st.session_state.current_chat
    }
}

if st.session_state.resume_path:
    os.environ["RESUME_PATH"] = st.session_state.resume_path

# LOAD CHAT HISTORY FROM SQLITE MEMORY
state = chatbot.get_state(config)

if state and "messages" in state.values:
    from langchain_core.messages import ToolMessage
    for msg in state.values["messages"]:
        try:
            # skip tool messages and AI messages with tool calls
            if isinstance(msg, ToolMessage):
                continue
            if isinstance(msg, AIMessage) and msg.tool_calls:
                continue
            if isinstance(msg, HumanMessage) and msg.content and isinstance(msg.content, str):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage) and msg.content and isinstance(msg.content, str):
                st.chat_message("assistant").write(msg.content)
        except Exception:
            continue

# ---------- USER INPUT ----------

user_input = st.chat_input("Type your message...")

if user_input:

    keywords = ["apply", "job", "resume", "application", "send mail", "send email"]
    if any(k in user_input.lower() for k in keywords) and st.session_state.resume_path is None:
        st.warning("⚠️ Please upload your resume in the sidebar before sending a job application.")
    else:
        if st.session_state.resume_path:
            os.environ["RESUME_PATH"] = st.session_state.resume_path

        st.chat_message("user").write(user_input)
        ai_reply = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            tool_placeholder = st.empty()
            try:
                for chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                ):
                    # show tool being called
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for tool_call in chunk.tool_calls:
                            tool_name = tool_call.get("name", "")
                            tool_icons = {
                                "web_search": "🔍 Searching news...",
                                "get_stock_price": "📈 Fetching stock price...",
                                "calculator": "🧮 Calculating...",
                                "send_job_application": "📧 Sending job application...",
                                "search_documents": "📄 Searching documents...",
                            }
                            msg = tool_icons.get(tool_name, f"⚙️ Running {tool_name}...")
                            tool_placeholder.info(msg)

                    if isinstance(chunk, AIMessage) and chunk.content:
                        tool_placeholder.empty()  # clear tool indicator
                        ai_reply += chunk.content
                        placeholder.markdown(ai_reply)

            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    placeholder.warning("⚠️ Token limit reached. Please wait a few minutes and try again.")
                else:
                    placeholder.error(f"Error: {str(e)}")