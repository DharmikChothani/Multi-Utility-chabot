from __future__ import annotations

from typing import Any, Dict, Optional

# -------------------
# PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def set_retriever(thread_id: str, retriever: Any, metadata: dict) -> None:
    """Register a retriever + its metadata for a given thread."""
    _THREAD_RETRIEVERS[str(thread_id)] = retriever
    _THREAD_METADATA[str(thread_id)] = metadata


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
