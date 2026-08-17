from __future__ import annotations

import sqlite3
from typing import List, Optional

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .auth import thread_owner
from .state import ChatState
from .tools import llm_with_tools, tools

# -------------------
# Nodes
# -------------------
# Note: rag_tool no longer needs thread_id passed in the system prompt or
# supplied by the model at all — it receives thread_id via an injected
# RunnableConfig (see tools.py), which LangGraph's ToolNode populates
# automatically from this same `config`. That removes the earlier failure
# mode where the model had to correctly transcribe a "<username>::<uuid>"
# thread_id string as a tool argument.
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant with access to a PDF the user has uploaded "
        "for this conversation. For ANY question about the document's content — "
        "including vague requests like 'show me the document' or 'what's in this "
        "PDF' — call `rag_tool` immediately, using the user's message (or a broad "
        "query like 'summary of the document') as the search query. Do not ask "
        "the user to clarify before calling the tool. Only tell the user no "
        "document is available if `rag_tool` actually returns no results. You can "
        "also use the web search, stock price, and calculator tools when helpful."
    )
)


def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads(user_id: Optional[str] = None) -> List[str]:
    """
    List thread_ids from the checkpointer.

    thread_ids are namespaced as "<username>::<uuid>" (see auth.make_thread_id).
    When `user_id` is provided, only that user's own threads are returned —
    this is what keeps each user's chat history private to them. When
    `user_id` is None, every thread is returned (e.g. for admin/debug use).
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if user_id is None or thread_owner(tid) == user_id:
            all_threads.add(tid)
    return list(all_threads)


def user_owns_thread(user_id: str, thread_id: str) -> bool:
    """Guard used before loading/continuing a thread from the UI."""
    return thread_owner(thread_id) == user_id


def delete_thread(thread_id: str) -> None:
    """
    Permanently delete a thread's checkpoint history from the sqlite-backed
    checkpointer. LangGraph's SqliteSaver has no public delete API, so this
    removes the rows directly from every table that has a thread_id column
    (works across SqliteSaver schema versions: checkpoints, writes,
    checkpoint_blobs, etc.).
    """
    cursor = conn.cursor()
    tables = [
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    for table in tables:
        columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
        if "thread_id" in columns:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
    conn.commit()