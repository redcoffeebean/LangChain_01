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
# 6) VectorStore 구현                          — FAISS/Chroma/(Oracle placeholder)
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
    st.sidebar.header("Module Setting")

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

    return {
        **DEFAULT_CONFIG,
        "embeddings": emb_key,
        "vectorstore": vs_key_effective,
        "requested_vectorstore": vs_key_requested,
        "splitter": sp_key,
        "llm": llm_key,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def main():
    st.set_page_config(page_title="Modular RAG Template", page_icon="📚", layout="wide")
    st.title("📚 Modular RAG Template")

    cfg = _sidebar_config()

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


Query → Query Embedding → Retriever (Vector Search:Similarity|MMR|MetaFiltering) → Prompt → LLM (호출|추론|응답생성) → Answer → History
    """)

    uploaded_files = st.file_uploader("문서 업로드 (PDF/DOCX/PPT/TXT)", type=["pdf", "docx", "doc", "pptx", "ppt", "txt"], accept_multiple_files=True)

    build_col, chat_col = st.columns([1, 2])

    with build_col:
        st.subheader("1) Vector Index")
        if st.button("Vector Index Build", use_container_width=True):
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
                    splits = split_documents(docs, splitter)

                    emb = get_embeddings(cfg["embeddings"])
                    vs = get_vectorstore(cfg["vectorstore"]).from_documents(splits, emb)

                    st.session_state["_chain"] = build_chain(vs, get_llm(cfg["llm"]))
                    st.success("인덱싱 및 체인 준비 완료")
                except Exception as e:
                    st.exception(e)

    with chat_col:
        st.subheader("Query")
        q = st.text_input("질문 입력")
        if st.button("질의", use_container_width=True):
            chain = st.session_state.get("_chain")
            if not chain:
                st.warning("먼저 문서 인덱싱을 수행하세요.")
            else:
                if not q or not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    try:
                        res = chain.invoke({"question": q})  # langchain 0.2+ invoke
                    except Exception:
                        # 일부 버전에서는 __call__ 사용
                        res = chain({"question": q})
                    answer = res.get("answer") or res.get("result")
                    st.write(answer)


if __name__ == "__main__":
    main()
