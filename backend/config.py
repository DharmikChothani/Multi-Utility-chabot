from __future__ import annotations

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
load_dotenv()

# -------------------
# LLM + embeddings
# -------------------
hf_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    temperature=0.2,
)

# 3. Chat wrapper for chat prompts & system roles
llm = ChatHuggingFace(llm=hf_endpoint)


embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5"
)
