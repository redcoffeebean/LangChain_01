# ==================================================
# Modular RAG Template — FIXED
# ==================================================
# 변경 요약
# - 뷰 전환(FAISS Dashboard ↔ RAG Mode) 후에도, 이미 빌드된 벡터 인덱스가
#   RAG 화면 상단에서 즉시 보이도록 "Active Vector Index" 패널을 추가했습니다.
# - 인덱스 빌드 직후 자동으로 RAG Mode로 전환되어(세션 유지) UX 혼란을 줄였습니다.
# - 세션 초기화(Reset) 버튼 추가: 인덱스/체인/메타/성능 수집을 한 번에 삭제합니다.
# - 체인이 준비돼 있으면 질문 영역을 항상 활성화하고, 현재 인덱스/구성 요약을 표시합니다.
# - 현재 VectorStore가 FAISS로 동작 중인지, 또는 Chroma/Pinecone인지 명확히 표시합니다.
#
# 실행: streamlit run modularity_03.py
# 필요: streamlit, langchain-community, langchain-text-splitters, langchain-openai,
#      langchain-huggingface, faiss-cpu, chromadb(선택), pinecone(선택), tiktoken 등
# ==================================================

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import importlib.util
import time
from pathlib import Path

# ------------------------------
# 기본 설정
# ------------------------------
DEFAULT_CONFIG = {
    "embeddings": "hf:minilm",   # 'hf:minilm' | 'hf:paraphrase' | 'openai'
    "vectorstore": "faiss",      # 'faiss' | 'chroma' | 'pinecone' | 'oracle'(placeholder)
    "splitter": "tiktoken",      # 'tiktoken' | 'char'
    "llm": "openai:gpt-4o-mini",# 'openai:gpt-4o-mini'
    "chunk_size": 800,
    "chunk_overlap": 100,
}

UI_CHOICES = {
    "Embeddings": [
        ("HuggingFace — all-MiniLM-L6-v2", "hf:minilm"),
        ("HuggingFace — paraphrase-MiniLM-L6-v2", "hf:paraphrase"),
        ("OpenAI — text-embedding-3-large", "openai"),
    ],
    "VectorStore": [
        ("FAISS (in-memory)", "faiss"),
        ("ChromaDB (local client)", "chroma"),
        ("Pinecone (managed)", "pinecone"),
        ("Oracle Vector (placeholder)", "oracle"),
    ],
    "Splitter": [
        ("tiktoken 기반 TokenSplitter", "tiktoken"),
        ("문자 기반 RecursiveCharacter", "char"),
    ],
    "LLM": [
        ("OpenAI — gpt-4o-mini", "openai:gpt-4o-mini"),
    ],
}

# ------------------------------
# 유틸/타입
# ------------------------------
@dataclass
class LoadedDoc:
    page_content: str
    metadata: Dict[str, Any]

def _ensure(msg: str, ok: bool):
    if not ok:
        raise RuntimeError(msg)

def is_pkg_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False

def is_sqlite_supported(min_major=3, min_minor=35, min_patch=0) -> bool:
    try:
        import sqlite3
        v = getattr(sqlite3, "sqlite_version_info", None)
        if not v:
            ver = getattr(sqlite3, "sqlite_version", "0.0.0")
            parts = [int(x) for x in ver.split(".")]
            while len(parts) < 3:
                parts.append(0)
            v = tuple(parts[:3])
        return tuple(v) >= (min_major, min_minor, min_patch)
    except Exception:
        return False

# ------------------------------
# VectorStore 판별/라벨링
# ------------------------------
class _FAISSStub: pass

def is_faiss_backed(provider) -> bool:
    try:
        from langchain_community.vectorstores import FAISS as FAISSClass
    except Exception:
        FAISSClass = _FAISSStub
    if isinstance(provider, FaissVS):
        return True
    vsv = getattr(provider, "vs", None)
    if vsv is None:
        return False
    if FAISSClass is not _FAISSStub and isinstance(vsv, FAISSClass):
        return True
    idx = getattr(vsv, "index", None)
    return hasattr(idx, "ntotal")

def effective_vs_name(provider) -> str:
    if is_faiss_backed(provider):
        return "faiss"
    vsv = getattr(provider, "vs", None)
    try:
        from langchain_community.vectorstores import Chroma
        if isinstance(vsv, Chroma):
            return "chroma"
    except Exception:
        pass
    try:
        from langchain_pinecone import PineconeVectorStore
        if isinstance(vsv, PineconeVectorStore):
            return "pinecone"
    except Exception:
        pass
    return type(provider).__name__

# ------------------------------
# Loader 구현
# ------------------------------

def load_documents(paths: List[str]) -> List[Any]:
    docs: List[Any] = []
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except Exception as e:
                raise RuntimeError("PyPDFLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            docs.extend(PyPDFLoader(p).load())
        elif ext in (".docx", ".doc"):
            try:
                from langchain_community.document_loaders import Docx2txtLoader
            except Exception as e:
                raise RuntimeError("Docx2txtLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            docs.extend(Docx2txtLoader(p).load())
        elif ext in (".pptx", ".ppt"):
            try:
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
            except Exception as e:
                raise RuntimeError("UnstructuredPowerPointLoader 사용을 위해 'unstructured' 계열 의존성이 필요할 수 있습니다.") from e
            docs.extend(UnstructuredPowerPointLoader(p).load())
        else:
            try:
                from langchain_community.document_loaders import TextLoader
            e
