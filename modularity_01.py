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
            except Exception as e:
                raise RuntimeError("TextLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs

# ------------------------------
# Splitter 구현
# ------------------------------

def get_splitter(kind: str, chunk_size: int, chunk_overlap: int):
    if kind == "tiktoken":
        try:
            from langchain_text_splitters import TokenTextSplitter
        except Exception as e:
            raise RuntimeError("tiktoken 기반 splitter를 위해 'langchain-text-splitters'와 'tiktoken'이 필요합니다.") from e
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as e:
            raise RuntimeError("문자 기반 splitter를 위해 'langchain-text-splitters' 설치가 필요합니다.") from e
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(docs: List[Any], splitter) -> List[Any]:
    return splitter.split_documents(docs)

# ------------------------------
# Embeddings 구현
# ------------------------------
class EmbeddingsProvider:
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

class HFEmbeddings(EmbeddingsProvider):
    def __init__(self, model_name: str):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception as e:
            raise RuntimeError("HuggingFace 임베딩을 위해 'langchain-huggingface'와 'sentence-transformers'가 필요합니다.") from e
        self._impl = HuggingFaceEmbeddings(model_name=model_name)
    def embed_documents(self, texts):
        return self._impl.embed_documents(texts)
    def embed_query(self, text):
        return self._impl.embed_query(text)

class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self, model: str = "text-embedding-3-large"):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            raise RuntimeError("OpenAI 임베딩을 위해 'langchain-openai'가 필요합니다.") from e
        _ensure("OPENAI_API_KEY 환경변수 필요", bool(os.getenv("OPENAI_API_KEY")))
        self._impl = OpenAIEmbeddings(model=model)
    def embed_documents(self, texts):
        return self._impl.embed_documents(texts)
    def embed_query(self, text):
        return self._impl.embed_query(text)

def get_embeddings(name: str) -> EmbeddingsProvider:
    if name.startswith("hf:"):
        if name == "hf:minilm":
            return HFEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        elif name == "hf:paraphrase":
            return HFEmbeddings("sentence-transformers/paraphrase-MiniLM-L6-v2")
    elif name == "openai":
        return OpenAIEmbeddingsProvider()
    raise ValueError(f"Unknown embeddings provider: {name}")

# ------------------------------
# VectorStore 구현
# ------------------------------
class VectorStoreProvider:
    def from_documents(self, docs: List[Any], embeddings: EmbeddingsProvider) -> "VectorStoreProvider": ...
    def as_retriever(self, **kwargs): ...
    def persist(self): ...

class FaissVS(VectorStoreProvider):
    def __init__(self):
        self.vs = None
    def from_documents(self, docs, embeddings):
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as e:
            raise RuntimeError("FAISS 사용을 위해 'faiss-cpu'와 'langchain-community'가 필요합니다.") from e
        embed_impl = getattr(embeddings, "_impl", embeddings)
        self.vs = FAISS.from_documents(docs, embed_impl)
        return self
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        pass

class ChromaVS(VectorStoreProvider):
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.vs = None
    def from_documents(self, docs, embeddings):
        try:
            from langchain_community.vectorstores import Chroma
        except Exception as e:
            raise RuntimeError("Chroma 사용을 위해 'chromadb'와 'langchain-community'가 필요합니다.") from e
        if not is_sqlite_supported():
            try:
                import streamlit as st
                st.warning("ChromaDB는 sqlite3 >= 3.35.0이 필요합니다. 현재 환경에서는 **FAISS**로 자동 대체합니다.")
            except Exception:
                pass
            from langchain_community.vectorstores import FAISS
            embed_impl = getattr(embeddings, "_impl", embeddings)
            self.vs = FAISS.from_documents(docs, embed_impl)
            return self
        embed_impl = getattr(embeddings, "_impl", embeddings)
        try:
            self.vs = Chroma.from_documents(docs, embed_impl, collection_name=self.collection_name)
            return self
        except Exception as e:
            if "sqlite" in str(e).lower():
                try:
                    import streamlit as st
                    st.warning("ChromaDB(sqlite) 초기화 실패 → **FAISS**로 자동 대체합니다.")
                except Exception:
                    pass
                from langchain_community.vectorstores import FAISS
                self.vs = FAISS.from_documents(docs, embed_impl)
                return self
            raise
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        try:
            self.vs.persist()
        except Exception:
            pass

class PineconeVS(VectorStoreProvider):
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.vs = None
    def _ensure_pkgs(self):
        if not (is_pkg_available("langchain_pinecone") and is_pkg_available("pinecone")):
            raise RuntimeError("Pinecone 사용을 위해 'langchain-pinecone'과 'pinecone' 패키지가 필요합니다.")
    def from_documents(self, docs, embeddings):
        self._ensure_pkgs()
        from langchain_pinecone import PineconeVectorStore
        try:
            from pinecone import Pinecone as PineconeClient, ServerlessSpec
        except Exception:
            from pinecone import Pinecone as PineconeClient
            ServerlessSpec = None
        api_key = os.getenv("PINECONE_API_KEY")
        _ensure("PINECONE_API_KEY 환경변수 필요 (사이드바에 입력)", bool(api_key))
        pc = PineconeClient(api_key=api_key)
        need_create = False
        try:
            pc.describe_index(self.index_name)
        except Exception:
            need_create = True
        if need_create:
            embed_impl = getattr(embeddings, "_impl", embeddings)
            dim = len(embed_impl.embed_query("dimension_probe_for_pinecone"))
            metric = "cosine"
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            if ServerlessSpec is not None:
                pc.create_index(
                    name=self.index_name,
                    dimension=dim,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
            else:
                pc.create_index(name=self.index_name, dimension=dim, metric=metric)
        embed_impl = getattr(embeddings, "_impl", embeddings)
        self.vs = PineconeVectorStore.from_documents(docs, embed_impl, index_name=self.index_name)
        return self
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        pass

class OracleVectorVS(VectorStoreProvider):
    def __init__(self):
        self.ready = False
    def from_documents(self, docs, embeddings):
        self.ready = True
        return self
    def as_retriever(self, **kwargs):
        if not self.ready:
            raise RuntimeError("OracleVectorVS가 초기화되지 않았습니다.")
        raise NotImplementedError("Oracle Vector Search 어댑터 구현 필요")
    def persist(self):
        pass

def get_vectorstore(name: str) -> VectorStoreProvider:
    if name == "faiss":
        return FaissVS()
    if name == "chroma":
        return ChromaVS()
    if name == "pinecone":
        idx = os.getenv("PINECONE_INDEX_NAME", "my-index")
        return PineconeVS(index_name=idx)
    if name == "oracle":
        return OracleVectorVS()
    raise ValueError(f"Unknown vectorstore provider: {name}")

# ------------------------------
# LLM 구현
# ------------------------------
class LLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._impl = None
    def get(self):
        if self._impl is None:
            try:
                from langchain_openai import ChatOpenAI
            except Exception as e:
                raise RuntimeError("OpenAI LLM을 위해 'langchain-openai'가 필요합니다.") from e
            _ensure("OPENAI_API_KEY 환경변수 필요", bool(os.getenv("OPENAI_API_KEY")))
            self._impl = ChatOpenAI(model=self.model_name, temperature=0.2)
        return self._impl

def get_llm(name: str) -> LLMProvider:
    if name.startswith("openai:"):
        _, model = name.split(":", 1)
        return LLMProvider(model)
    raise ValueError(f"Unknown LLM provider: {name}")

# ------------------------------
# Chain Builder
# ------------------------------

def build_chain(vs_provider: VectorStoreProvider, llm_provider: LLMProvider):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    retriever = vs_provider.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = llm_provider.get()
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ------------------------------
# Streamlit UI
# ------------------------------
import streamlit as st

# --- 사이드바 구성/뷰 토글 ---

def _sidebar_config():
    st.sidebar.header("모듈 선택")

    emb_label = st.sidebar.selectbox(
        "Embeddings", options=[x[0] for x in UI_CHOICES["Embeddings"]], index=0,
    )
    emb_key = dict(UI_CHOICES["Embeddings"])[emb_label]

    vector_candidates = [("FAISS (in-memory)", "faiss")]
    chroma_available = is_pkg_available("chromadb")
    sqlite_ok = is_sqlite_supported()
    if chroma_available:
        if sqlite_ok:
            vector_candidates.append(("ChromaDB (local client)", "chroma"))
        else:
            vector_candidates.append(("ChromaDB (SQLite<3.35 — 자동으로 FAISS 사용)", "chroma"))
    else:
        vector_candidates.append(("ChromaDB (미설치 — 자동으로 FAISS 사용)", "chroma"))

    pinecone_pkgs = is_pkg_available("langchain_pinecone") and is_pkg_available("pinecone")
    if pinecone_pkgs:
        vector_candidates.append(("Pinecone (managed)", "pinecone"))
    else:
        vector_candidates.append(("Pinecone (미설치 — 자동으로 FAISS 사용)", "pinecone"))

    vs_label = st.sidebar.selectbox("VectorStore", options=[x[0] for x in vector_candidates], index=0)
    vs_key_requested = dict(vector_candidates)[vs_label]
    vs_key_effective = vs_key_requested

    if vs_key_requested == "pinecone":
        pinecone_api = st.sidebar.text_input("PINECONE_API_KEY (Pinecone 선택 시 필수)", type="password")
        if pinecone_api:
            os.environ["PINECONE_API_KEY"] = pinecone_api
        pinecone_idx = st.sidebar.text_input("PINECONE_INDEX_NAME", value=os.getenv("PINECONE_INDEX_NAME", "my-index"))
        if pinecone_idx:
            os.environ["PINECONE_INDEX_NAME"] = pinecone_idx
        pinecone_cloud = st.sidebar.text_input("PINECONE_CLOUD (aws/gcp)", value=os.getenv("PINECONE_CLOUD", "aws"))
        if pinecone_cloud:
            os.environ["PINECONE_CLOUD"] = pinecone_cloud
        pinecone_region = st.sidebar.text_input("PINECONE_REGION", value=os.getenv("PINECONE_REGION", "us-east-1"))
        if pinecone_region:
            os.environ["PINECONE_REGION"] = pinecone_region
        if not pinecone_pkgs or not os.getenv("PINECONE_API_KEY"):
            st.sidebar.info("Pinecone 패키지 또는 API 키가 없어 **FAISS**로 자동 대체합니다. 설치: `pip install langchain-pinecone pinecone`.")
            vs_key_effective = "faiss"

    if vs_key_requested == "chroma" and (not chroma_available or not sqlite_ok):
        if not chroma_available:
            st.sidebar.info("ChromaDB가 설치되어 있지 않아 **FAISS**로 자동 대체합니다. 설치: `pip install chromadb`.")
        elif not sqlite_ok:
            st.sidebar.info("현재 sqlite3 버전이 낮아 ChromaDB 사용이 제한됩니다 (>= 3.35.0 필요). **FAISS**로 자동 대체합니다.")
        vs_key_effective = "faiss"

    sp_label = st.sidebar.selectbox("Splitter", options=[x[0] for x in UI_CHOICES["Splitter"]], index=0)
    sp_key = dict(UI_CHOICES["Splitter"])[sp_label]

    llm_label = st.sidebar.selectbox("LLM", options=[x[0] for x in UI_CHOICES["LLM"]], index=0)
    llm_key = dict(UI_CHOICES["LLM"])[llm_label]

    chunk_size = st.sidebar.slider("chunk_size", 200, 2000, DEFAULT_CONFIG["chunk_size"], step=50)
    chunk_overlap = st.sidebar.slider("chunk_overlap", 0, 400, DEFAULT_CONFIG["chunk_overlap"], step=20)

    api_key_input = st.sidebar.text_input("OPENAI_API_KEY (OpenAI 사용 시 필수)", type="password", help="OpenAI LLM/임베딩 사용 시 입력")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    if "view" not in st.session_state:
        st.session_state["view"] = "rag"

    st.sidebar.divider()
    if st.sidebar.button("FAISS Dashboard"):
        st.session_state["view"] = "faiss"
    if st.sidebar.button("RAG Mode"):
        st.session_state["view"] = "rag"

    return {
        **DEFAULT_CONFIG,
        "embeddings": emb_key,
        "vectorstore": vs_key_effective,
        "requested_vectorstore": vs_key_requested,
        "splitter": sp_key,
        "llm": llm_key,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "view": st.session_state["view"],
    }

# --- 공통: 간단 메트릭 계산 ---

def _vs_quick_stats(vs_provider):
    vsv = getattr(vs_provider, "vs", None)
    index = getattr(vsv, "index", None) if vsv else None
    ntotal = getattr(index, "ntotal", 0) if index is not None else 0
    dim = getattr(index, "d", None) if index is not None else None
    idx_type = type(index).__name__ if index is not None else "-"
    return ntotal, dim, idx_type

# --- FAISS Dashboard 렌더 ---

def render_faiss_dashboard(cfg):
    st.header("📊 FAISS Dashboard")
    st.caption(
        "이 화면은 벡터 인덱스(FAISS)의 상태, 구성, 성능, 관리 기능을 한 번에 보여줍니다. "
        "문서 업로드→인덱싱 후 여기서 현황을 확인하세요."
    )

    vs_provider = st.session_state.get("_vs_provider")
    if not vs_provider:
        st.info("아직 VectorStore가 준비되지 않았습니다. 먼저 인덱스를 빌드하세요.")
        return
    if not is_faiss_backed(vs_provider):
        st.info("현재 VectorStore가 FAISS 기반이 아닙니다. 사이드바에서 FAISS를 선택하거나, 환경에서 FAISS 폴백 여부를 확인하세요.")
        return

    vsv = getattr(vs_provider, "vs", None)
    index = getattr(vsv, "index", None) if vsv else None
    if index is None:
        st.warning("FAISS 인덱스가 아직 준비되지 않았습니다. 먼저 인덱스를 빌드하세요.")
        return

    ntotal, dim, index_type = _vs_quick_stats(vs_provider)
    metric_guess = "L2" if "L2" in index_type.upper() else ("IP" if "IP" in index_type.upper() else "?")
    mem_mb = (ntotal * (dim or 0) * 4) / (1024 * 1024) if dim else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Vectors (ntotal)", ntotal)
        st.caption("인덱스에 저장된 벡터(=청크) 개수.")
    with c2:
        st.metric("Dimension (d)", dim if dim is not None else "-")
        st.caption("임베딩 벡터의 차원 수.")
    with c3:
        st.metric("Metric", metric_guess)
        st.caption("IndexFlatL2는 L2, Inner Product는 IP.")
    with c4:
        st.metric("Est. Memory (MB)", f"{mem_mb:.2f}" if mem_mb is not None else "-")
        st.caption("대략적인 벡터 저장 용량 추정치.")

    st.subheader("구성 정보")
    st.table([
        {"Key": "VectorStore", "Value": "FAISS"},
        {"Key": "Index Type", "Value": index_type},
        {"Key": "Embeddings", "Value": cfg["embeddings"]},
        {"Key": "Splitter", "Value": cfg["splitter"]},
        {"Key": "Chunk Size", "Value": cfg["chunk_size"]},
        {"Key": "Chunk Overlap", "Value": cfg["chunk_overlap"]},
        {"Key": "LLM", "Value": cfg["llm"]},
    ])

    st.subheader("성능 정보")
    perf = st.session_state.get("_perf", {})
    chunk_time = perf.get("chunk_time_s")
    index_time = perf.get("index_time_s")
    q_times = perf.get("query_times", [])

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**인덱싱**")
        st.write(f"- Chunking: {chunk_time:.3f}s" if isinstance(chunk_time, (int, float)) else "- Chunking: -")
        st.write(f"- FAISS.from_documents: {index_time:.3f}s" if isinstance(index_time, (int, float)) else "- FAISS.from_documents: -")
    with cols[1]:
        st.markdown("**질의 지연시간(최근)**")
        if q_times:
            st.write(f"- count={len(q_times)}, avg={sum(q_times)/len(q_times):.3f}s, min={min(q_times):.3f}s, max={max(q_times):.3f}s")
            st.line_chart(q_times)
        else:
            st.write("- 수집된 질의가 없습니다.")

    st.subheader("관리")
    st.caption("index.faiss + docstore.pkl + index_to_docstore_id.pkl로 저장/복원합니다.")
    save_col, load_col = st.columns(2)
    with save_col:
        save_dir = st.text_input("저장 폴더", value=st.session_state.get("_faiss_save_dir", "./faiss_store"), key="faiss_save_dir")
        if st.button("FAISS 저장"):
            st.session_state["_faiss_save_dir"] = save_dir
            try:
                if hasattr(vsv, "save_local"):
                    vsv.save_local(save_dir)
                else:
                    import faiss, pickle
                    os.makedirs(save_dir, exist_ok=True)
                    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
                    with open(os.path.join(save_dir, "docstore.pkl"), "wb") as f:
                        import pickle as _p
                        _p.dump(vsv.docstore, f)
                    with open(os.path.join(save_dir, "index_to_docstore_id.pkl"), "wb") as f:
                        import pickle as _p
                        _p.dump(vsv.index_to_docstore_id, f)
                st.success(f"저장 완료: {save_dir}")
            except Exception as e:
                st.exception(e)
    with load_col:
        load_dir = st.text_input("불러오기 폴더", value=st.session_state.get("_faiss_load_dir", "./faiss_store"), key="faiss_load_dir")
        if st.button("FAISS 불러오기"):
            st.session_state["_faiss_load_dir"] = load_dir
            try:
                from langchain_community.vectorstores import FAISS as FAISSClass
                emb = get_embeddings(cfg["embeddings"])
                embed_impl = getattr(emb, "_impl", emb)
                loaded = FAISSClass.load_local(load_dir, embed_impl, allow_dangerous_deserialization=True)
                provider = FaissVS(); provider.vs = loaded
                st.session_state["_vs_provider"] = provider
                st.session_state["_chain"] = build_chain(provider, get_llm(cfg["llm"]))
                st.success("불러오기 성공 및 체인 갱신 완료")
            except Exception as e:
                st.exception(e)

    st.subheader("문서/청크 메타")
    meta = st.session_state.get("_faiss_meta", {})
    if meta:
        st.json(meta)
    else:
        st.write("메타데이터가 없습니다. 인덱싱을 한 번 수행해 보세요.")

# --- RAG 모드 렌더 ---

def render_rag(cfg):
    # 상단: 현재 인덱스 요약 패널(신규)
    st.header("💬 RAG Mode")

    active_vs = st.session_state.get("_vs_provider")
    c1, c2 = st.columns([2, 1])
    with c1:
        if active_vs and getattr(active_vs, "vs", None):
            ntotal, dim, idx_type = _vs_quick_stats(active_vs)
            st.success(
                f"Active Vector Index: **{effective_vs_name(active_vs).upper()}** — "
                f"vectors={ntotal}, dim={dim if dim is not None else '-'}, index={idx_type}"
            )
        else:
            st.info("Active Vector Index 없음. 먼저 인덱스를 빌드하세요.")
    with c2:
        if st.button("Reset (세션 초기화)"):
            for k in ["_vs_provider", "_chain", "_faiss_meta", "_perf"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

    # 설명/파이프라인
    st.markdown(
        """
**RAG-Corpus:**
Loader → Splitter(Seperator|tokenizer) → (Chunk → Embedding) → (Vector Store → Vector Index)

**Query-Serving:**
Query → Query Embedding → Retriever (Vector Search:Similarity|MMR|MetaFiltering) → Prompt → LLM (호출|추론|응답생성) → Answer
        """
    )

    uploaded_files = st.file_uploader(
        "문서 업로드 (PDF/DOCX/PPT/TXT)",
        type=["pdf", "docx", "doc", "pptx", "ppt", "txt"],
        accept_multiple_files=True,
        key="uploader",  # 키 고정: 뷰 전환 후에도 위젯 ID 안정화
    )

    build_col, chat_col = st.columns([1, 2])

    with build_col:
        st.subheader("1) Vector Index")
        if st.button("Vector Index Build", use_container_width=True):
            if not uploaded_files:
                st.error("최소 1개 파일을 업로드하세요.")
            else:
                tmp_paths = []
                base_dir = Path(st.secrets.get("_tmp_dir", "."))
                base_dir.mkdir(parents=True, exist_ok=True)
                for uf in uploaded_files:
                    p = base_dir / uf.name
                    with open(p, "wb") as f:
                        f.write(uf.getbuffer())
                    tmp_paths.append(str(p))
                try:
                    docs = load_documents(tmp_paths)
                    splitter = get_splitter(cfg["splitter"], cfg["chunk_size"], cfg["chunk_overlap"])
                    t_split0 = time.perf_counter()
                    splits = split_documents(docs, splitter)
                    t_split1 = time.perf_counter()

                    emb = get_embeddings(cfg["embeddings"])
                    t_index0 = time.perf_counter()
                    vs = get_vectorstore(cfg["vectorstore"]).from_documents(splits, emb)
                    t_index1 = time.perf_counter()

                    st.session_state["_vs_provider"] = vs
                    used_name = effective_vs_name(vs)
                    st.session_state["_faiss_meta"] = {
                        "files": [uf.name for uf in uploaded_files],
                        "num_docs": len(docs),
                        "num_chunks": len(splits),
                        "vectorstore_used": used_name,
                    }
                    perf = st.session_state.get("_perf", {})
                    perf["chunk_time_s"] = t_split1 - t_split0
                    perf["index_time_s"] = t_index1 - t_index0
                    perf.setdefault("query_times", [])
                    st.session_state["_perf"] = perf

                    st.session_state["_chain"] = build_chain(vs, get_llm(cfg["llm"]))
                    st.session_state["view"] = "rag"  # 빌드 후 자동 RAG 모드 유지
                    st.success("인덱싱 및 체인 준비 완료")
                except Exception as e:
                    st.exception(e)

    with chat_col:
        st.subheader("2) Query")
        q = st.text_input("질문입력")
        if st.button("질문하기", use_container_width=True):
            chain = st.session_state.get("_chain")
            if not chain:
                st.warning("먼저 문서 인덱싱을 수행하세요.")
            else:
                if not q or not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    t0 = time.perf_counter()
                    try:
                        res = chain.invoke({"question": q})
                    except Exception:
                        res = chain({"question": q})
                    t1 = time.perf_counter()

                    perf = st.session_state.get("_perf", {})
                    q_times = perf.setdefault("query_times", [])
                    q_times.append(t1 - t0)
                    if len(q_times) > 50:
                        q_times[:] = q_times[-50:]
                    st.session_state["_perf"] = perf

                    answer = res.get("answer") or res.get("result")
                    st.write(answer)

        # 현재 구성/메타 간단 표시(신규)
        st.divider()
        st.markdown("**현재 구성 요약**")
        st.table([
            {"Key": "Embeddings", "Value": cfg["embeddings"]},
            {"Key": "VectorStore", "Value": st.session_state.get("_faiss_meta", {}).get("vectorstore_used", cfg["vectorstore"])},
            {"Key": "Splitter", "Value": cfg["splitter"]},
            {"Key": "Chunk Size", "Value": cfg["chunk_size"]},
            {"Key": "Chunk Overlap", "Value": cfg["chunk_overlap"]},
            {"Key": "LLM", "Value": cfg["llm"]},
        ])
        meta = st.session_state.get("_faiss_meta", {})
        if meta:
            st.json(meta)

# --- 메인 ---

def main():
    st.set_page_config(page_title="Modular RAG Template", page_icon="📚", layout="wide")
    st.title("📚 Modular RAG Template — FIXED")

    cfg = _sidebar_config()

    if (cfg["llm"].startswith("openai:") or cfg["embeddings"] == "openai") and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OpenAI 사용 시 OPENAI_API_KEY를 입력하세요.")
    if cfg.get("requested_vectorstore") != cfg["vectorstore"]:
        st.sidebar.warning(
            f"요청한 VectorStore '{cfg.get('requested_vectorstore')}'을(를) 사용할 수 없어 "
            f"**{cfg['vectorstore'].upper()}** 로 자동 대체했습니다. 필요한 패키지/키/환경을 확인하세요."
        )

    if cfg.get("view") == "faiss":
        render_faiss_dashboard(cfg)
        return

    render_rag(cfg)

if __name__ == "__main__":
    main()
