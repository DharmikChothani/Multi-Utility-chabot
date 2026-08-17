from __future__ import annotations

import streamlit as st

from backend.auth import (
    create_session,
    delete_session,
    make_thread_id,
    register_user,
    validate_session,
    verify_user,
)
from backend.graph import chatbot, delete_thread, retrieve_all_threads, user_owns_thread
from backend.ingestion import ingest_pdf
from backend.store import delete_retriever, thread_document_metadata, thread_has_document

st.set_page_config(page_title="Mutlti Utility Chatbot", page_icon="🛠️", layout="wide")

# =========================================================
# Session state defaults
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None  # thread_id awaiting delete confirmation

# ---- Restore login from a session token in the URL, if present ----
# This is what keeps a user logged in across a browser refresh: on login we
# stash a random token in the URL's query params, and on every rerun (which
# includes a hard refresh) we check that token against the sessions table.
if st.session_state.user is None:
    token_from_url = st.query_params.get("session")
    restored_user = validate_session(token_from_url)
    if restored_user:
        st.session_state.user = restored_user


# =========================================================
# 1. LOGIN / REGISTER — block everything else until authenticated
# =========================================================
def login_screen() -> None:
    st.title("💬 Chatbot — Sign in")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", type="primary"):
            if verify_user(username, password):
                clean_username = username.strip()
                st.session_state.user = clean_username
                st.session_state.thread_id = None
                st.session_state.messages = []
                # Issue a session token and persist it in the URL so a
                # refresh doesn't log the user back out.
                token = create_session(clean_username)
                st.query_params["session"] = token
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        new_username = st.text_input("Choose a username", key="reg_username")
        new_password = st.text_input(
            "Choose a password", type="password", key="reg_password"
        )
        if st.button("Create account"):
            if len(new_password or "") < 6:
                st.error("Password must be at least 6 characters.")
            elif register_user(new_username, new_password):
                st.success("Account created. Please log in.")
            else:
                st.error("That username is already taken.")


if st.session_state.user is None:
    login_screen()
    st.stop()

user = st.session_state.user

# Ensure there is always an active thread before the sidebar renders,
# since the sidebar's PDF section is scoped to the current thread.
if st.session_state.thread_id is None:
    st.session_state.thread_id = make_thread_id(user)

thread_id = st.session_state.thread_id

# Defense in depth: never let a user act on a thread_id that isn't theirs,
# even if one somehow ended up in session_state (e.g. via a stale URL/param).
if not user_owns_thread(user, thread_id):
    st.error("You don't have access to that chat.")
    st.stop()

# =========================================================
# 2. SIDEBAR — logout, PDF upload for the active chat, chat history
# =========================================================
with st.sidebar:
    st.write(f"Logged in as **{user}**")
    if st.button("Log out"):
        delete_session(st.query_params.get("session"))
        st.query_params.clear()
        st.session_state.user = None
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.thread_id = make_thread_id(user)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ---- PDF upload / status for the currently active chat ----
    # Only one document lives per thread at a time. Uploading a new PDF
    # here always replaces whatever was previously indexed for this
    # thread — there's no separate "replace" toggle or "ingest" button;
    # a file appearing in the uploader is itself the trigger to index it.
    st.subheader("📄 Document")
    has_doc = thread_has_document(thread_id)

    if has_doc:
        meta = thread_document_metadata(thread_id)
        st.success(f"**{meta.get('filename', 'Document')}**")
        st.caption(f"{meta.get('documents', '?')} pages · {meta.get('chunks', '?')} chunks")
        st.caption("Uploading a new PDF below will replace this document.")
    else:
        st.caption("No document indexed for this chat yet.")

    pdf_file = st.file_uploader(
        "Upload PDF", type=["pdf"], key=f"uploader_{thread_id}", label_visibility="collapsed"
    )
    # Track the last file we auto-ingested for this thread so a rerun
    # (e.g. from switching threads or sending a chat message) doesn't
    # re-trigger ingestion of the same file over and over.
    last_ingested_key = f"last_ingested_{thread_id}"
    if pdf_file is not None and st.session_state.get(last_ingested_key) != pdf_file.file_id:
        with st.spinner("Indexing document..."):
            try:
                summary = ingest_pdf(
                    pdf_file.getvalue(), thread_id=thread_id, filename=pdf_file.name
                )
                st.session_state[last_ingested_key] = pdf_file.file_id
                st.success(
                    f"Indexed '{summary['filename']}' "
                    f"({summary['documents']} pages, {summary['chunks']} chunks)."
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.divider()

    # ---- This user's own chat history ----
    st.subheader("Your chats")
    # retrieve_all_threads(user) only returns threads namespaced to this user —
    # this is what keeps histories private between accounts.
    my_threads = sorted(retrieve_all_threads(user_id=user), reverse=True)

    if not my_threads:
        st.caption("No chats yet — start a new one above.")

    for tid in my_threads:
        short_id = tid.split("::", 1)[-1][:8]
        is_active = tid == thread_id
        label = f"{'🟢' if is_active else '🗂️'} {short_id}"
        if thread_has_document(tid):
            fname = thread_document_metadata(tid).get("filename")
            if fname:
                label += f" — {fname}"

        if st.session_state.pending_delete == tid:
            # ---- Confirmation step for this specific chat ----
            st.warning(f"Delete this chat ({short_id})? This can't be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("✅ Delete", key=f"confirm_del_{tid}", use_container_width=True):
                    delete_thread(tid)
                    delete_retriever(tid)
                    st.session_state.pending_delete = None
                    if is_active:
                        st.session_state.thread_id = None
                        st.session_state.messages = []
                    st.rerun()
            with cancel_col:
                if st.button("✖️ Cancel", key=f"cancel_del_{tid}", use_container_width=True):
                    st.session_state.pending_delete = None
                    st.rerun()
        else:
            row_col, del_col = st.columns([5, 1])
            with row_col:
                if st.button(
                    label,
                    key=f"thread_{tid}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    st.session_state.thread_id = tid
                    # Reload message history for this thread from the checkpointer
                    state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
                    st.session_state.messages = (
                        state.values.get("messages", []) if state else []
                    )
                    st.rerun()
            with del_col:
                if st.button("🗑️", key=f"del_{tid}", help="Delete this chat"):
                    st.session_state.pending_delete = tid
                    st.rerun()

# =========================================================
# 3. MAIN AREA — chat only
# =========================================================
st.title("💬 Multi-Utility-Chatbot")

for msg in st.session_state.messages:
    role = "user" if msg.type == "human" else "assistant"
    if getattr(msg, "content", None):
        with st.chat_message(role):
            st.markdown(msg.content)

user_input = st.chat_input("Ask something...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke(
                {"messages": [{"role": "user", "content": user_input}]}, config=config
            )
            reply = result["messages"][-1]
            st.markdown(reply.content)

    st.session_state.messages = result["messages"]