# ==================================================
# RAG Single-File Template (모듈식 교체가 쉬운 단일 파일 구조)
# ==================================================
# 목적: 한 파일 안에서 Loader / Splitter / Embeddings / VectorStore / LLM / Chain / UI
#       각 기능을 구획하고, 설정(config) 또는 선택(selectbox)만 바꾸면 구현체를 교체할 수 있게 함.
# 사용: Streamlit로 실행 (예: `streamlit run streamlit_rag_singlefile_template.py`)
# 주의: 필요 라이브러리 설치
#   pip install streamlit langchain-community langchain-text-splitters langchain-openai langchain-huggingface faiss-cpu chromadb tiktoken langchain-pinecone pinecone
#   (선택) transformers accelerate sentence-transformers
# ==================================================


# ##################################################
# 목차 (Flow Overview)
# ##################################################
# 1) CONFIG & REGISTRY                         — 구현 선택, 기본 옵션
# 2) 공통 타입 & 유틸                          — 간단 타입/헬퍼
# 3) Loader 구현                               — PDF/DOCX/PPT/TXT
# 4) Splitter 구현                             — tiktoken/char 기반
# 5) Embeddings 구현                           — HuggingFace/OpenAI
# 6) VectorStore 구현                          — FAISS/Chroma/Pinecone/(Oracle placeholder)
# 7) LLM 구현                                  — OpenAI(기본), (옵션) 로컬
# 8) Chain Builder                             — ConversationalRetrievalChain 조립
# 9) Streamlit UI                              — 파일 업로드, 구성 선택, 채팅


# ##################################################
# 1) CONFIG & REGISTRY
# ##################################################
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import importlib.util
import time, math

DEFAULT_CONFIG = {
    "embeddings": "hf:minilm",   # 'hf:minilm' | 'hf:paraphrase' | 'openai'
    "vectorstore": "faiss",      # 'faiss' | 'chroma' | 'oracle' (placeholder)
    "splitter": "tiktoken",      # 'tiktoken' | 'char'
    "llm": "openai:gpt-4o-mini", # 'openai:gpt-4o-mini' | (옵션) 'local:...'
    "chunk_size": 800,
    "chunk_overlap": 100,
}

# 선택지 라벨 → 내부 키 매핑 (UI에 표시될 보기)
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


# ##################################################
# 2) 공통 타입 & 유틸
# ##################################################
@dataclass
class LoadedDoc:
    page_content: str
    metadata: Dict[str, Any]


def _ensure(msg: str, ok: bool):
    if not ok:
        raise RuntimeError(msg)

# 패키지 설치 여부 체크 (예: chromadb 유무에 따라 UI에서 자동 대체)
def is_pkg_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False

# sqlite3 최소 버전(Chroma 요구: >= 3.35.0) 충족 여부 점검
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

# --- Helper: 현재 VectorStore가 실질적으로 FAISS인지 판단 (Chroma 내부 폴백 포함) ---
def is_faiss_backed(provider) -> bool:
    try:
        from langchain_community.vectorstores import FAISS as FAISSClass
    except Exception:
        FAISSClass = None
    if isinstance(provider, FaissVS):
        return True
    vsv = getattr(provider, "vs", None)
    if vsv is None:
        return False
    if FAISSClass is not None and isinstance(vsv, FAISSClass):
        return True
    # Heuristic: FAISS 인덱스는 .index.ntotal 특성이 존재
    idx = getattr(vsv, "index", None)
    return hasattr(idx, "ntotal")

# --- Helper: 실제 사용 중인 VectorStore 이름 ---
def effective_vs_name(provider) -> str:
    if is_faiss_backed(provider):
        return "faiss"
    # Chroma/Pinecone 감지 (가능하면)
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


# ##################################################
# 3) Loader 구현 — PDF/DOCX/PPT/TXT (langchain-community 권장)
# ##################################################
from pathlib import Path

def load_documents(paths: List[str]) -> List[Any]:
    """파일 확장자별 Loader 선택 후 Document 리스트 반환.
    반환: langchain의 Document 객체 리스트 (downstream 호환)
    """
    docs: List[Any] = []
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except Exception as e:
                raise RuntimeError("PyPDFLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        elif ext in (".docx", ".doc"):
            try:
                from langchain_community.document_loaders import Docx2txtLoader
            except Exception as e:
                raise RuntimeError("Docx2txtLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            loader = Docx2txtLoader(p)
            docs.extend(loader.load())
        elif ext in (".pptx", ".ppt"):
            try:
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
            except Exception as e:
                raise RuntimeError("UnstructuredPowerPointLoader 사용을 위해 'unstructured' 계열 의존성이 필요할 수 있습니다.") from e
            loader = UnstructuredPowerPointLoader(p)
            docs.extend(loader.load())
        else:  # 기본 TXT/기타는 TextLoader 시도
            try:
                from langchain_community.document_loaders import TextLoader
            except Exception as e:
                raise RuntimeError("TextLoader 사용을 위해 'langchain-community'가 필요합니다.") from e
            loader = TextLoader(p, encoding="utf-8")
            docs.extend(loader.load())
    return docs


# ##################################################
# 4) Splitter 구현 — tiktoken / char
# ##################################################

def get_splitter(kind: str, chunk_size: int, chunk_overlap: int):
    if kind == "tiktoken":
        try:
            from langchain_text_splitters import TokenTextSplitter
        except Exception as e:
            raise RuntimeError("tiktoken 기반 splitter를 위해 'langchain-text-splitters'와 'tiktoken'이 필요합니다.") from e
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        # 문자 기반 분할 (RecursiveCharacterTextSplitter)
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as e:
            raise RuntimeError("문자 기반 splitter를 위해 'langchain-text-splitters' 설치가 필요합니다.") from e
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(docs: List[Any], splitter) -> List[Any]:
    return splitter.split_documents(docs)


# ##################################################
# 5) Embeddings 구현 — HuggingFace / OpenAI
# ##################################################
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


# ##################################################
# 6) VectorStore 구현 — FAISS / Chroma / Pinecone / Oracle(placeholder)
# ##################################################
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
        # ★ 수정: LangChain 버전 차이를 흡수하기 위해 내부 임베딩 구현체를 전달
        embed_impl = getattr(embeddings, "_impl", embeddings)
        self.vs = FAISS.from_documents(docs, embed_impl)
        return self
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        pass  # in-memory 기본


class ChromaVS(VectorStoreProvider):
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.vs = None
    def from_documents(self, docs, embeddings):
        try:
            from langchain_community.vectorstores import Chroma
        except Exception as e:
            raise RuntimeError("Chroma 사용을 위해 'chromadb'와 'langchain-community'가 필요합니다.") from e

        # sqlite 버전이 낮으면 FAISS로 자동 폴백
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

        # 내부 임베딩 구현체를 전달하여 호환성 보장
        embed_impl = getattr(embeddings, "_impl", embeddings)
        try:
            self.vs = Chroma.from_documents(docs, embed_impl, collection_name=self.collection_name)
            return self
        except Exception as e:
            # 런타임에서 sqlite 관련 에러가 난 경우에도 FAISS로 폴백
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


# ---- Pinecone VectorStore ----
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
            from pinecone import Pinecone as PineconeClient  # ServerlessSpec 미사용 구버전 호환
            ServerlessSpec = None

        api_key = os.getenv("PINECONE_API_KEY")
        _ensure("PINECONE_API_KEY 환경변수 필요 (사이드바에 입력)", bool(api_key))

        pc = PineconeClient(api_key=api_key)

        # 인덱스 존재 여부 확인 → 없으면 생성
        need_create = False
        try:
            pc.describe_index(self.index_name)
        except Exception:
            need_create = True

        if need_create:
            # 임베딩 차원 자동 추론
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
                # 구 SDK 호환 (serverless spec 없이)
                pc.create_index(name=self.index_name, dimension=dim, metric=metric)

        embed_impl = getattr(embeddings, "_impl", embeddings)
        self.vs = PineconeVectorStore.from_documents(
            docs, embed_impl, index_name=self.index_name
        )
        return self

    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)

    def persist(self):
        pass  # 원격 관리형


class OracleVectorVS(VectorStoreProvider):
    """Oracle Vector Search 연동 Placeholder.
    - 실제 구현은 Oracle 23ai 테이블+HNSW 인덱스 스키마, upsert, 검색(SQL) 어댑터 필요.
    - 여기서는 설계 포인트만 유지(학습/발표용 템플릿).
    """
    def __init__(self):
        self.ready = False
    def from_documents(self, docs, embeddings):
        # TODO: docs → (id, text, embedding) 변환 후 Oracle 테이블 업서트 + 벡터 인덱스 생성
        # 예시) embedding = embeddings.embed_documents([d.page_content for d in docs])
        #       cx_Oracle/oracledb 사용, VECTOR 컬럼과 HNSW 인덱스 구성
        self.ready = True
        return self
    def as_retriever(self, **kwargs):
        if not self.ready:
            raise RuntimeError("OracleVectorVS가 초기화되지 않았습니다.")
        # TODO: 질의 임베딩 후 SQL로 top-k 검색, 결과를 langchain Document로 어댑트
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


# ##################################################
# 7) LLM 구현 — OpenAI(기본)
# ##################################################
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


# ##################################################
# 8) Chain Builder — ConversationalRetrievalChain
# ##################################################

def build_chain(vs_provider: VectorStoreProvider, llm_provider: LLMProvider):
    try:
        # LangChain 0.2+ 경로 우선
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
    except Exception:
        # 일부 구버전 호환 (필요 시 다른 import)
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory

    retriever = vs_provider.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = llm_provider.get()

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return chain


# ##################################################
# 9) Streamlit UI — 업로드/선택/빌드/질의
# ##################################################
import streamlit as st


def _sidebar_config():
    st.sidebar.header("구성 선택 (한 파일 내 교체)")

    emb_label = st.sidebar.selectbox(
        "Embeddings",
        options=[x[0] for x in UI_CHOICES["Embeddings"]],
        index=0,
    )
    emb_key = dict(UI_CHOICES["Embeddings"])[emb_label]

    # VectorStore 선택지: chromadb / pinecone 유무 + sqlite 지원 여부에 따라 라벨 안내 및 자동 대체
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

    vs_label = st.sidebar.selectbox(
        "VectorStore",
        options=[x[0] for x in vector_candidates],
        index=0,
    )
    vs_key_requested = dict(vector_candidates)[vs_label]
    vs_key_effective = vs_key_requested

    # Pinecone 설정 입력 (선택 시 표시)
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

    # Chroma 사용 불가 조건 → FAISS로 자동 대체
    if vs_key_requested == "chroma" and (not chroma_available or not sqlite_ok):
        if not chroma_available:
            st.sidebar.info("ChromaDB가 설치되어 있지 않아 **FAISS**로 자동 대체합니다. 설치: `pip install chromadb`.")
        elif not sqlite_ok:
            st.sidebar.info("현재 sqlite3 버전이 낮아 ChromaDB 사용이 제한됩니다 (>= 3.35.0 필요). **FAISS**로 자동 대체합니다.")
        vs_key_effective = "faiss"

    sp_label = st.sidebar.selectbox(
        "Splitter",
        options=[x[0] for x in UI_CHOICES["Splitter"]],
        index=0,
    )
    sp_key = dict(UI_CHOICES["Splitter"])[sp_label]

    llm_label = st.sidebar.selectbox(
        "LLM",
        options=[x[0] for x in UI_CHOICES["LLM"]],
        index=0,
    )
    llm_key = dict(UI_CHOICES["LLM"])[llm_label]

    chunk_size = st.sidebar.slider("chunk_size", 200, 2000, DEFAULT_CONFIG["chunk_size"], step=50)
    chunk_overlap = st.sidebar.slider("chunk_overlap", 0, 400, DEFAULT_CONFIG["chunk_overlap"], step=20)

    # OpenAI API 키 입력 (입력 시 즉시 환경변수에 반영)
    api_key_input = st.sidebar.text_input("OPENAI_API_KEY (OpenAI 사용 시 필수)", type="password", help="OpenAI LLM/임베딩 사용 시 입력")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    # === View 전환 버튼 ===
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


def render_faiss_dashboard(cfg):
    st.header("📊 FAISS Dashboard")
    st.caption(
        "이 화면은 벡터 인덱스(FAISS)의 상태, 구성, 성능, 관리 기능을 한 번에 보여줍니다. "
        "문서 업로드→인덱싱 후 여기서 현황을 확인하세요.")

    vs_provider = st.session_state.get("_vs_provider")
    if not vs_provider:
        st.info("아직 VectorStore가 준비되지 않았습니다. 먼저 인덱스를 빌드하세요.")
        return
    if not is_faiss_backed(vs_provider):
        st.info("현재 VectorStore가 FAISS 기반이 아닙니다. 사이드바에서 FAISS를 선택하거나, Chroma가 내부적으로 FAISS로 폴백되지 않았는지 확인하세요.")
        return

    vsv = getattr(vs_provider, "vs", None)
    index = getattr(vsv, "index", None) if vsv else None
    if index is None:
        st.warning("FAISS 인덱스가 아직 준비되지 않았습니다. 먼저 인덱스를 빌드하세요.")
        return

    # --- 요약 메트릭 ---
    ntotal = getattr(index, "ntotal", 0)
    dim = getattr(index, "d", None)
    index_type = type(index).__name__
    metric_guess = "L2" if "L2" in index_type.upper() else ("IP" if "IP" in index_type.upper() else "?")
    mem_mb = (ntotal * (dim or 0) * 4) / (1024 * 1024) if dim else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Vectors (ntotal)", ntotal)
        st.caption("인덱스에 저장된 벡터(=청크) 개수. 많을수록 검색 후보가 늘어 정확도에 유리하지만, 검색/메모리 비용이 증가합니다.")
    with c2:
        st.metric("Dimension (d)", dim if dim is not None else "-")
        st.caption("임베딩 벡터의 차원 수. 사용한 임베딩 모델에 의해 결정됩니다. (예: all-MiniLM-L6-v2 → 384)")
    with c3:
        st.metric("Metric", metric_guess)
        st.caption("FAISS 인덱스의 거리/유사도 지표. IndexFlatL2는 L2(유클리드 거리), Inner Product는 IP로 표시됩니다.")
    with c4:
        st.metric("Est. Memory (MB)", f"{mem_mb:.2f}" if mem_mb is not None else "-")
        st.caption("대략적인 벡터 저장 용량 추정치(ntotal×dim×4byte). 인덱스/메타 등 부가 오버헤드는 포함하지 않습니다.")

    # --- 구성 정보 ---
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
    st.caption(
        "• VectorStore: 실제 검색 백엔드
"
        "• Index Type: 인덱스 구조(예: IndexFlatL2 — 정확하지만 큰 데이터에서 느릴 수 있음)
"
        "• Embeddings: 임베딩 모델 (차원·성능 지표 차이에 영향)
"
        "• Splitter/Chunk Size/Overlap: 분할 전략과 크기 (검색 품질·인덱스 크기·속도에 영향)
"
        "• LLM: 최종 답변을 생성하는 모델 (응답 속도·비용·품질에 영향)")

    # --- 성능 정보 ---
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
        st.caption("Chunking은 문서를 청크로 나누는 시간, from_documents는 임베딩 계산 + 인덱스 빌드를 합친 시간입니다.")
    with cols[1]:
        st.markdown("**질의 지연시간(최근)**")
        if q_times:
            st.write(f"- count={len(q_times)}, avg={sum(q_times)/len(q_times):.3f}s, min={min(q_times):.3f}s, max={max(q_times):.3f}s")
            st.line_chart(q_times)
            st.caption("질의 버튼을 누른 순간부터 답변이 나올 때까지의 총 소요 시간입니다. 검색 + LLM 호출/생성을 모두 포함합니다.")
        else:
            st.write("- 수집된 질의가 없습니다.")
            st.caption("여기에 시간이 누적되도록, 오른쪽 ‘대화’ 섹션에서 질문을 실행해 보세요.")

    with st.expander("🔧 튜닝 팁/설명 더보기"):
        st.markdown(
            "- **느린 경우**: retriever `k`를 5→3으로 낮추거나, 더 가벼운 LLM을 선택하세요. 초기 몇 번은 모델/런타임 워밍업으로 오래 걸릴 수 있습니다.
"
            "- **대용량 데이터**: IndexFlatL2 대신 IVF/IVFPQ/HNSW 같은 ANN 인덱스를 검토하세요(정확도 일부 희생, 속도/메모리 절감).
"
            "- **정확도 향상**: chunk_size를 문서 특성에 맞춰 조정하고, MMR/메타 필터링/프롬프트 개선을 병행하세요.
"
            "- **코사인 유사도**: 벡터를 L2 정규화하고 Inner Product(IP)로 검색하면 코사인과 동일 효과를 얻을 수 있습니다.")

    # --- 관리(저장/불러오기) ---
    st.subheader("관리")
    st.caption(
        "‘FAISS 저장’은 index.faiss(인덱스) + docstore.pkl + index_to_docstore_id.pkl을 지정 폴더에 기록합니다.
"
        "‘FAISS 불러오기’는 해당 파일들을 읽어 인덱스를 복원하고, 체인을 자동으로 재빌드합니다.")
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
                        pickle.dump(vsv.docstore, f)
                    with open(os.path.join(save_dir, "index_to_docstore_id.pkl"), "wb") as f:
                        pickle.dump(vsv.index_to_docstore_id, f)
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

    # --- 문서/청크 메타 ---
    st.subheader("문서/청크 메타")
    st.caption("업로드 파일, 문서/청크 수, 실제 사용된 VectorStore를 보여줍니다. Chroma 선택→환경 문제로 FAISS 폴백된 경우도 구분할 수 있습니다.")
    meta = st.session_state.get("_faiss_meta", {})
    if meta:
        st.json(meta)
    else:
        st.write("메타데이터가 없습니다. 인덱싱을 한 번 수행해 보세요.")

def main():
    st.set_page_config(page_title="RAG Single-File Template", page_icon="📚", layout="wide")
    st.title("📚 RAG Single-File Template — 모듈 교체형")

    cfg = _sidebar_config()

    # 뷰 전환: FAISS Dashboard 모드
    if cfg.get("view") == "faiss":
        render_faiss_dashboard(cfg)
        return

    # ★ 추가: OpenAI 선택됐는데 키가 없으면 사이드바 경고
    if (cfg["llm"].startswith("openai:") or cfg["embeddings"] == "openai") and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OpenAI 사용 시 OPENAI_API_KEY를 입력하세요.")
    # VectorStore 대체 안내 (요청값과 실제 사용값이 다른 경우)
    if cfg.get("requested_vectorstore") != cfg["vectorstore"]:
        st.sidebar.warning(
            f"요청한 VectorStore '{cfg.get('requested_vectorstore')}'을(를) 사용할 수 없어 "
            f"**{cfg['vectorstore'].upper()}** 로 자동 대체했습니다. 필요한 패키지/키/환경을 확인하세요."
        )

    st.markdown("""
**RAG-Corpus:**


Loader → Splitter(Seperator|tokenizer) → (Chunk → Embedding) → (Vector Store → Vector Index)


**Query-Serving:**


Query → Query Embedding → Retriever (Vector Search:Similarity|MMR|MetaFiltering) → Prompt → LLM (호출|추론|응답생성) → Answer
    """)

    uploaded_files = st.file_uploader("문서 업로드 (PDF/DOCX/PPT/TXT)", type=["pdf", "docx", "doc", "pptx", "ppt", "txt"], accept_multiple_files=True)

    build_col, chat_col = st.columns([1, 2])

    with build_col:
        st.subheader("1) 인덱스 빌드")
        if st.button("문서 인덱싱 시작", use_container_width=True):
            if not uploaded_files:
                st.error("최소 1개 파일을 업로드하세요.")
            else:
                tmp_paths = []
                for uf in uploaded_files:
                    p = Path(st.secrets.get("_tmp_dir", ".")) / uf.name
                    with open(p, "wb") as f:
                        f.write(uf.getbuffer())
                    tmp_paths.append(str(p))

                try:
                    docs = load_documents(tmp_paths)
                    splitter = get_splitter(cfg["splitter"], cfg["chunk_size"], cfg["chunk_overlap"])
                    # 성능 측정: split
                    t_split0 = time.perf_counter()
                    splits = split_documents(docs, splitter)
                    t_split1 = time.perf_counter()

                    # 성능 측정: 벡터 인덱스 빌드
                    emb = get_embeddings(cfg["embeddings"])
                    t_index0 = time.perf_counter()
                    vs = get_vectorstore(cfg["vectorstore"]).from_documents(splits, emb)
                    t_index1 = time.perf_counter()

                    # 상태 저장
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
                    st.success("인덱싱 및 체인 준비 완료")
                except Exception as e:
                    st.exception(e)

    with chat_col:
        st.subheader("2) 대화")
        q = st.text_input("질문 입력")
        if st.button("질의", use_container_width=True):
            chain = st.session_state.get("_chain")
            if not chain:
                st.warning("먼저 문서 인덱싱을 수행하세요.")
            else:
                if not q or not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    t0 = time.perf_counter()
                    try:
                        res = chain.invoke({"question": q})  # langchain 0.2+ invoke
                    except Exception:
                        # 일부 버전에서는 __call__ 사용
                        res = chain({"question": q})
                    t1 = time.perf_counter()

                    # 질의 성능 누적
                    perf = st.session_state.get("_perf", {})
                    q_times = perf.setdefault("query_times", [])
                    q_times.append(t1 - t0)
                    # 최근 50개만 유지
                    if len(q_times) > 50:
                        q_times[:] = q_times[-50:]
                    st.session_state["_perf"] = perf

                    answer = res.get("answer") or res.get("result")
                    st.write(answer)


if __name__ == "__main__":
    main()
