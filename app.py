import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.graph import chatbot, retrieve_all_threads
from backend.ingestion import ingest_pdf
from backend.store import thread_document_metadata

# =========================== Page Config ===========================
st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================== Custom Styling ===========================
st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
        }

        /* Main title */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            padding-bottom: 0 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #12141c;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stButton button {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.15s ease-in-out;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: #6c5ce7;
            color: #a29bfe;
        }

        /* New chat button emphasis */
        div[data-testid="stSidebar"] div:first-child button[kind="secondary"] {
            font-weight: 600;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.6rem;
            margin-bottom: 0.4rem;
            border: 1px solid rgba(255,255,255,0.05);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: rgba(108, 92, 231, 0.10);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: rgba(255,255,255,0.03);
        }

        /* Chat input bar */
        div[data-testid="stChatInput"] textarea {
            border-radius: 12px;
        }

        /* Sidebar subheaders */
        section[data-testid="stSidebar"] h3 {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9aa0b4;
            margin-top: 1rem;
        }

        /* Badge-style thread id */
        .thread-badge {
            display: inline-block;
            background: rgba(108, 92, 231, 0.15);
            color: #a29bfe;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.72rem;
            font-family: monospace;
        }

        /* Active thread highlight */
        .active-thread button {
            border-color: #6c5ce7 !important;
            background: rgba(108, 92, 231, 0.12) !important;
            color: #a29bfe !important;
        }

        .empty-state {
            text-align: center;
            padding: 4rem 1rem;
            color: #7b8196;
        }
        .empty-state .big {
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def delete_thread(thread_id):
    """Remove a thread from session state and reset if it is currently active."""
    str_tid = str(thread_id)
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)

    st.session_state.get("thread_titles", {}).pop(str_tid, None)
    st.session_state.get("ingested_docs", {}).pop(str_tid, None)

    # If deleting the current active thread, start a fresh conversation
    if str(st.session_state["thread_id"]) == str_tid:
        reset_chat()
    st.rerun()


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


def get_thread_label(thread_id, max_chars=22):
    """Retrieve the user's last question to use as a thread title."""
    str_tid = str(thread_id)
    if str_tid in st.session_state.get("thread_titles", {}):
        return st.session_state["thread_titles"][str_tid]

    messages = load_conversation(thread_id)
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

    if user_messages:
        content = user_messages[-1].content
        label = (content[:max_chars] + "…") if len(content) > max_chars else content
    else:
        label = f"Chat {str_tid[:6]}"

    st.session_state.setdefault("thread_titles", {})[str_tid] = label
    return label


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.markdown("## 🧠 Multi Utility Chatbot")
st.sidebar.markdown(
    f'<span class="thread-badge">Thread {thread_key[:8]}</span>',
    unsafe_allow_html=True,
)
st.sidebar.write("")

if st.sidebar.button("➕  New Chat", use_container_width=True, type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("### 📄 Document")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"**{latest_doc.get('filename')}**\n\n"
        f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages",
        icon="✅",
    )
else:
    st.sidebar.info("No PDF indexed yet for this chat.", icon="📭")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"], label_visibility="collapsed"
)
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.markdown("### 💬 Past Conversations")
if not threads:
    st.sidebar.caption("No past conversations yet — start chatting!")
else:
    for tid in threads:
        label = get_thread_label(tid)
        is_active = str(tid) == thread_key
        icon = "🟣" if is_active else "💬"

        row_class = "active-thread" if is_active else ""
        st.sidebar.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns([0.82, 0.18])
        with col1:
            if st.button(
                f"{icon} {label}",
                key=f"side-thread-{tid}",
                use_container_width=True,
            ):
                selected_thread = tid
        with col2:
            if st.button("🗑️", key=f"del-thread-{tid}", help="Delete chat"):
                delete_thread(tid)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ============================ Main Layout ========================
st.title("🧠 Multi Utility Chatbot")
st.caption("Chat freely, or upload a PDF in the sidebar to ask questions about it.")
st.divider()

# Empty state when no messages yet
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="big">💬</div>
            <div><strong>Start a conversation</strong></div>
            <div>Ask a question below, or upload a document to chat with it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render message history
AVATARS = {"user": "🧑", "assistant": "🤖"}
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools…")

if user_input:
    # Update title cache with current query
    truncated_title = (user_input[:22] + "…") if len(user_input) > 22 else user_input
    st.session_state["thread_titles"][thread_key] = truncated_title

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📎 Document indexed: **{doc_meta.get('filename')}** "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

# Handle thread switching
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
