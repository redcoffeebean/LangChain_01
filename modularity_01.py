# ==================================================
# RAG Single-File Template (모듈식 교체가 쉬운 단일 파일 구조)
# ==================================================
# 목적: 한 파일 안에서 Loader / Splitter / Embeddings / VectorStore / LLM / Chain / UI
#       각 기능을 구획하고, 설정(config) 또는 선택(selectbox)만 바꾸면 구현체를 교체할 수 있게 함.
# 사용: Streamlit로 실행 (예: `streamlit run streamlit_rag_singlefile_template.py`)
# 주의: 필요 라이브러리 설치
#   pip install streamlit langchain-community langchain-text-splitters langchain-openai langchain-huggingface faiss-cpu chromadb tiktoken
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
# 6) VectorStore 구현 — FAISS / Chroma / Oracle(placeholder)
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
        self.vs = FAISS.from_documents(docs, embeddings)  # embeddings는 langchain 임베딩 호환 필요
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
        # langchain의 Chroma 래퍼는 임베딩 객체(호환형)를 인자로 받음
        self.vs = Chroma.from_documents(docs, embeddings)
        return self
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        try:
            self.vs.persist()
        except Exception:
            pass


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

    vs_label = st.sidebar.selectbox(
        "VectorStore",
        options=[x[0] for x in UI_CHOICES["VectorStore"]],
        index=0,
    )
    vs_key = dict(UI_CHOICES["VectorStore"])[vs_label]

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

    # ★ 추가: OpenAI API 키 입력 팝업 (입력 시 즉시 환경변수에 설정)
    api_key_input = st.sidebar.text_input("OPENAI_API_KEY (OpenAI 사용 시 필수)", type="password", help="OpenAI LLM/임베딩 사용 시 입력하세요")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    return {
        **DEFAULT_CONFIG,
        "embeddings": emb_key,
        "vectorstore": vs_key,
        "splitter": sp_key,
        "llm": llm_key,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def main():
    st.set_page_config(page_title="RAG Single-File Template", page_icon="📚", layout="wide")
    st.title("📚 RAG Single-File Template — 모듈 교체형")

    cfg = _sidebar_config()

    # ★ 추가: OpenAI 선택됐는데 키가 없으면 사이드바 경고
    if (cfg["llm"].startswith("openai:") or cfg["embeddings"] == "openai") and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OpenAI 사용 시 OPENAI_API_KEY를 입력하세요.")

    st.markdown("""
    **흐름:** Loader → Splitter → Embeddings → VectorStore → (Retriever) → LLM → Chain (ConversationalRetrieval) → 답변
    
    좌측에서 구현체를 바꾸면 한 파일 안에서 즉시 교체가 가능합니다.
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
                    splits = split_documents(docs, splitter)

                    emb = get_embeddings(cfg["embeddings"])
                    vs = get_vectorstore(cfg["vectorstore"]).from_documents(splits, emb)

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
                try:
                    res = chain.invoke({"question": q})  # langchain 0.2+ invoke
                except Exception:
                    # 일부 버전에서는 __call__ 사용
                    res = chain({"question": q})
                answer = res.get("answer") or res.get("result")
                st.write(answer)


if __name__ == "__main__":
    main()
