from __future__ import annotations

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
load_dotenv()

# -------------------
# LLM + embeddings
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5"
)
