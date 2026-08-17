from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

# -------------------
# PDF retriever store (per thread)
# -------------------
# In-memory cache for the current process, backed by an on-disk FAISS index
# per thread so documents survive a server/app restart.
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

INDEX_ROOT = "faiss_indexes"


def _index_dir(thread_id: str) -> str:
    # thread_id looks like "username::uuid" — "::" isn't safe in all
    # filesystems, so swap it for a plain double-underscore on disk.
    return os.path.join(INDEX_ROOT, str(thread_id).replace("::", "__"))


def _metadata_path(thread_id: str) -> str:
    return os.path.join(_index_dir(thread_id), "metadata.json")


def _dirname_to_thread_id(dirname: str) -> str:
    return dirname.replace("__", "::", 1)


def _load_metadata_from_disk(thread_id: str) -> dict:
    path = _metadata_path(thread_id)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available. Checks the in-memory
    cache first, then falls back to loading a persisted FAISS index from
    disk (e.g. after an app restart) and re-populates the cache.
    """
    if not thread_id:
        return None
    thread_id = str(thread_id)

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    index_dir = _index_dir(thread_id)
    if os.path.isdir(index_dir):
        # Imported lazily so modules that never touch FAISS don't pay the cost.
        from langchain_community.vectorstores import FAISS

        from .config import embeddings

        vector_store = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = _load_metadata_from_disk(thread_id)
        return retriever

    return None


def set_retriever(
    thread_id: str, retriever: Any, metadata: dict, vector_store: Any = None
) -> None:
    """
    Register a retriever + its metadata for a given thread, and persist the
    underlying FAISS index to disk (when `vector_store` is provided) so it
    survives a restart.
    """
    thread_id = str(thread_id)
    _THREAD_RETRIEVERS[thread_id] = retriever
    _THREAD_METADATA[thread_id] = metadata

    if vector_store is not None:
        index_dir = _index_dir(thread_id)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        with open(_metadata_path(thread_id), "w") as f:
            json.dump(metadata, f)


def thread_has_document(thread_id: str) -> bool:
    thread_id = str(thread_id)
    if thread_id in _THREAD_RETRIEVERS:
        return True
    return os.path.isdir(_index_dir(thread_id))


def thread_document_metadata(thread_id: str) -> dict:
    thread_id = str(thread_id)
    if thread_id in _THREAD_METADATA:
        return _THREAD_METADATA[thread_id]
    return _load_metadata_from_disk(thread_id)


def thread_ids_with_documents_for_user(username: str) -> List[str]:
    """
    Return thread_ids (namespaced as "<username>::<id>") that both belong to
    this user and currently have a document indexed — checking both the
    in-memory cache and persisted indexes on disk.
    """
    prefix = f"{username}::"
    ids = {tid for tid in _THREAD_RETRIEVERS if tid.startswith(prefix)}

    if os.path.isdir(INDEX_ROOT):
        for name in os.listdir(INDEX_ROOT):
            tid = _dirname_to_thread_id(name)
            if tid.startswith(prefix):
                ids.add(tid)

    return list(ids)


def delete_retriever(thread_id: str) -> None:
    """Drop any indexed PDF/retriever for a thread, in memory and on disk."""
    thread_id = str(thread_id)
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)

    index_dir = _index_dir(thread_id)
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir, ignore_errors=True)