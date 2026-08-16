# Multi Utility Chatbot

A Streamlit chat application backed by a [LangGraph](https://langchain-ai.github.io/langgraph/) agent that can hold multi-turn conversations, answer questions about an uploaded PDF (RAG), search the web, fetch live stock prices, and do basic arithmetic — all with persistent, resumable chat threads.

## Features

- 💬 **Multi-threaded chat** — start new conversations, revisit past ones, and delete threads from the sidebar
- 📄 **Chat with your PDF** — upload a document per thread; it's chunked, embedded, and indexed with FAISS for retrieval-augmented answers
- 🔧 **Tool-using agent** — the LLM can call tools on its own:
  - `rag_tool` — search the uploaded document
  - `search_tool` — DuckDuckGo web search
  - `get_stock_price` — live quote via Alpha Vantage
  - `calculator` — add / sub / mul / div
- 💾 **Persistent state** — conversations are checkpointed to SQLite, so threads survive app restarts
- ✨ **Polished UI** — streaming responses, live tool-use status, avatars, and thread highlighting

## Architecture

```
.
├── app.py                          # Streamlit UI
└── langgraph_rag_backend/          # LangGraph agent + RAG backend
    ├── __init__.py                 # public API (chatbot, ingest_pdf, ...)
    ├── config.py                   # .env loading, LLM + embeddings init
    ├── store.py                    # in-memory per-thread retriever/metadata store
    ├── ingestion.py                # PDF → chunks → FAISS retriever
    ├── tools.py                    # search, calculator, stock price, rag_tool
    ├── state.py                    # LangGraph ChatState definition
    └── graph.py                    # chat node, checkpointer, compiled graph
```

**How a message flows:**

1. `app.py` sends the user's message to `chatbot` (the compiled LangGraph graph), scoped to the active `thread_id`.
2. `chat_node` (in `graph.py`) invokes the LLM with the full message history plus a system prompt.
3. If the LLM requests a tool call (e.g. `rag_tool` for a document question), the `tools` node runs it and loops back to `chat_node` with the result.
4. The final AI response streams back to the UI, along with live status updates ("🔧 Using `rag_tool`…").
5. Every turn is checkpointed to `chatbot.db` (SQLite) via `SqliteSaver`, so you can close the app and resume any thread later.

## Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (for the LLM)
- An [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) (for stock prices) — a demo key is included in the code but should be replaced with your own for real use

### Install dependencies

```bash
pip install streamlit langgraph langchain langchain-core langchain-community \
    langchain-groq langchain-huggingface faiss-cpu pypdf python-dotenv \
    duckduckgo-search requests sentence-transformers
```

> `langchain-huggingface`'s `BAAI/bge-small-en-v1.5` embedding model will be downloaded automatically on first run.

## Setup

1. Clone/copy the project files into a folder, keeping the structure above.
2. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. (Optional) Replace the hard-coded Alpha Vantage API key in `langgraph_rag_backend/tools.py` with your own, ideally read from an environment variable.

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Usage

- Click **➕ New Chat** to start a fresh thread.
- Upload a PDF from the sidebar to index it for the current thread — the assistant will automatically use `rag_tool` when your question relates to the document.
- Ask general questions, request a web search, ask for a stock quote (e.g. "What's the price of AAPL?"), or do quick math — the agent picks the right tool.
- Switch between past conversations or delete them from the **Past Conversations** list in the sidebar.

## Notes & Limitations

- The PDF retriever store (`store.py`) is **in-memory**, so uploaded documents are lost on app restart — only the chat history persists (via SQLite). Re-upload the PDF if you restart the app.
- The included Alpha Vantage key is a shared demo key with strict rate limits; swap in your own for reliable use.
- `chatbot.db` (SQLite checkpoint file) is created automatically in the working directory on first run.
