from __future__ import annotations
import os
import time
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import streamlit as st

# ------------------------------
# 기본 설정
# ------------------------------
DEFAULT_CONFIG = {
    "embeddings": "hf:minilm",   # 'hf:minilm' | 'hf:paraphrase' | 'openai'
    "vectorstore": "faiss",      # 'faiss' | 'chroma' | 'pinecone' | 'oracle'(placeholder)
    "splitter": "tiktoken",      # 'tiktoken' | 'char'
    "llm": "openai:gpt-4o-mini", # 'openai:gpt-4o-mini'
    "chunk_size": 800,
    "chunk_overlap": 100,
}

UI_CHOICES = {
    "Embedding Model": [
        ("HuggingFace — all-MiniLM-L6-v2", "hf:minilm"),
        ("HuggingFace — paraphrase-MiniLM-L6-v2", "hf:paraphrase"),
        ("OpenAI — text-embedding-3-large", "openai"),
    ],
    "VectorStore": [
        ("FAISS (in-memory)", "faiss"),
        ("ChromaDB (local client)", "chroma"),
        ("Pinecone (managed)", "pinecone"),
    ],
    "Splitter": [
        ("tiktoken 기반 TokenSplitter", "tiktoken"),
        ("문자 기반 RecursiveCharacter", "char"),
    ],
    "LLM Model": [
        ("OpenAI — gpt-4o-mini", "openai:gpt-4o-mini"),
    ],
}

# ------------------------------
# 유틸
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
# VectorStore 감지
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
# 문서 로더
# ------------------------------
def load_documents(paths: List[str]) -> List[Any]:
    docs: List[Any] = []
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs.extend(PyPDFLoader(p).load())
        elif ext in (".docx", ".doc"):
            from langchain_community.document_loaders import Docx2txtLoader
            docs.extend(Docx2txtLoader(p).load())
        elif ext in (".pptx", ".ppt"):
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            docs.extend(UnstructuredPowerPointLoader(p).load())
        else:
            from langchain_community.document_loaders import TextLoader
            docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs

# ------------------------------
# 스플리터
# ------------------------------
def get_splitter(kind: str, chunk_size: int, chunk_overlap: int):
    if kind == "tiktoken":
        from langchain_text_splitters import TokenTextSplitter
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def split_documents(docs: List[Any], splitter) -> List[Any]:
    return splitter.split_documents(docs)

# ------------------------------
# 임베딩
# ------------------------------
class EmbeddingsProvider:
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

class HFEmbeddings(EmbeddingsProvider):
    def __init__(self, model_name: str):
        from langchain_huggingface import HuggingFaceEmbeddings
        self._impl = HuggingFaceEmbeddings(model_name=model_name)
    def embed_documents(self, texts):
        return self._impl.embed_documents(texts)
    def embed_query(self, text):
        return self._impl.embed_query(text)

class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self, model: str = "text-embedding-3-large"):
        from langchain_openai import OpenAIEmbeddings
        _ensure("OPENAI_API_KEY 환경변수 필요", bool(os.getenv("OPENAI_API_KEY")))
        self._impl = OpenAIEmbeddings(model=model)
    def embed_documents(self, texts):
        return self._impl.embed_documents(texts)
    def embed_query(self, text):
        return self._impl.embed_query(text)

def get_embeddings(name: str) -> EmbeddingsProvider:
    if name == "hf:minilm":
        return HFEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    if name == "hf:paraphrase":
        return HFEmbeddings("sentence-transformers/paraphrase-MiniLM-L6-v2")
    if name == "openai":
        return OpenAIEmbeddingsProvider()
    raise ValueError(name)

# ------------------------------
# 벡터스토어
# ------------------------------
class VectorStoreProvider:
    def from_documents(self, docs: List[Any], embeddings: EmbeddingsProvider) -> "VectorStoreProvider": ...
    def as_retriever(self, **kwargs): ...
    def persist(self): ...

class FaissVS(VectorStoreProvider):
    def __init__(self):
        self.vs = None
    def from_documents(self, docs, embeddings):
        from langchain_community.vectorstores import FAISS
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
        from langchain_community.vectorstores import Chroma, FAISS
        embed_impl = getattr(embeddings, "_impl", embeddings)
        if not is_sqlite_supported():
            self.vs = FAISS.from_documents(docs, embed_impl)
            return self
        try:
            self.vs = Chroma.from_documents(docs, embed_impl, collection_name=self.collection_name)
            return self
        except Exception:
            self.vs = FAISS.from_documents(docs, embed_impl)
            return self
    def as_retriever(self, **kwargs):
        return self.vs.as_retriever(**kwargs)
    def persist(self):
        try:
            self.vs.persist()
        except Exception:
            pass

def get_vectorstore(name: str) -> VectorStoreProvider:
    if name == "faiss":
        return FaissVS()
    if name == "chroma":
        return ChromaVS()
    return FaissVS()

# ------------------------------
# LLM/체인
# ------------------------------
class LLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._impl = None
    def get(self):
        if self._impl is None:
            from langchain_openai import ChatOpenAI
            _ensure("OPENAI_API_KEY 환경변수 필요", bool(os.getenv("OPENAI_API_KEY")))
            self._impl = ChatOpenAI(model=self.model_name, temperature=0.2)
        return self._impl

def get_llm(name: str) -> LLMProvider:
    if name.startswith("openai:"):
        _, model = name.split(":", 1)
        return LLMProvider(model)
    raise ValueError(name)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def build_chain(vs_provider: VectorStoreProvider, llm_provider: LLMProvider):
    retriever = vs_provider.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = llm_provider.get()
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ------------------------------
# 사이드바
# ------------------------------
def _sidebar_config():
    st.sidebar.header("모듈 선택")

    emb_label = st.sidebar.selectbox(
        "Embeddings", options=[x[0] for x in UI_CHOICES["Embedding Model"]], index=0,
    )
    emb_key = dict(UI_CHOICES["Embedding Model"])[emb_label]

    vector_candidates = [("FAISS (in-memory)", "faiss")]
    chroma_available = is_pkg_available("chromadb")
    sqlite_ok = is_sqlite_supported()
    if chroma_available:
        vector_candidates.append(("ChromaDB (local client)", "chroma"))
    else:
        vector_candidates.append(("ChromaDB (미설치 — 자동으로 FAISS 사용)", "chroma"))

    vs_label = st.sidebar.selectbox("VectorStore", options=[x[0] for x in vector_candidates], index=0)
    vs_key_requested = dict(vector_candidates)[vs_label]
    vs_key_effective = vs_key_requested if (vs_key_requested != "chroma" or chroma_available) else "faiss"

    sp_label = st.sidebar.selectbox("Splitter", options=[x[0] for x in UI_CHOICES["Splitter"]], index=0)
    sp_key = dict(UI_CHOICES["Splitter"])[sp_label]

    llm_label = st.sidebar.selectbox("LLM", options=[x[0] for x in UI_CHOICES["LLM Model"]], index=0)
    llm_key = dict(UI_CHOICES["LLM Model"])[llm_label]

    chunk_size = st.sidebar.slider("chunk_size", 200, 2000, DEFAULT_CONFIG["chunk_size"], step=50)
    chunk_overlap = st.sidebar.slider("chunk_overlap", 0, 400, DEFAULT_CONFIG["chunk_overlap"], step=20)

    api_key_input = st.sidebar.text_input("OPENAI_API_KEY (OpenAI 사용 시 필수)", type="password", help="OpenAI LLM/임베딩 사용 시 입력")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    if "view" not in st.session_state:
        st.session_state["view"] = "rag"
    st.sidebar.divider()
    if st.sidebar.button("FAISS Dashboard", use_container_width=True):
        st.session_state["view"] = "faiss"
    if st.sidebar.button("RAG Mode", use_container_width=True):
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

# ------------------------------
# FAISS Dashboard (캡션 유지)
# ------------------------------
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

    ntotal = getattr(index, "ntotal", 0)
    dim = getattr(index, "d", None)
    index_type = type(index).__name__
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

    st.subheader("문서/청크 메타")
    meta = st.session_state.get("_faiss_meta", {})
    if meta:
        st.json(meta)
    else:
        st.write("메타데이터가 없습니다. 인덱싱을 한 번 수행해 보세요.")

# ------------------------------
# 메인 (RAG)
# ------------------------------
def main():
    st.set_page_config(page_title="Modular RAG Template", page_icon="📚", layout="wide")
    st.title("📚 Modular RAG Template")

    cfg = _sidebar_config()

    # 뷰 전환
    if cfg.get("view") == "faiss":
        render_faiss_dashboard(cfg)
        return

    # OpenAI 키 안내
    if (cfg["llm"].startswith("openai:") or cfg["embeddings"] == "openai") and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OpenAI 사용 시 OPENAI_API_KEY를 입력하세요.")
    if cfg.get("requested_vectorstore") != cfg["vectorstore"]:
        st.sidebar.warning(
            f"요청한 VectorStore '{cfg.get('requested_vectorstore')}'을(를) 사용할 수 없어 "
            f"**{cfg['vectorstore'].upper()}** 로 자동 대체했습니다. 필요한 패키지/키/환경을 확인하세요."
        )

    # 파이프라인 설명
    st.markdown(
        """
📢 **RAG-Corpus:** Loader → Splitter(Seperator|tokenizer) → (Chunks → Embedding) → (Vector Store → Vector Index)

👽 **Query-Serving:** Instruction → User Query → Query Embedding → Retriever (Similarity|MMR|MetaFiltering) → Top-k Chunks → LLM (호출|추론|응답생성) → Answer
        """
    )
#Similarity 검색: query와 가장 비슷한 문서/청크를 순서대로 뽑음 → 중복이 많아질 수 있음
#MMR 검색: query와 비슷하면서도 이미 뽑힌 것들과 덜 겹치는 것을 뽑음 → coverage(범위)가 좋아짐
#MMR = Maximal Marginal Relevance = 최대 한계 관련성
#ANN(Approximate Nearest Neighbor) 인덱스 (HNSW/IVF등)
#토큰은 문자/단어가 아니라 기존 학습된 조각 사전(vocabulary)에 맞춰 쪼갠 결과

    # --- Active Vector Index banner (inside main) ---
    active_vs = st.session_state.get("_vs_provider")
    if active_vs and getattr(active_vs, "vs", None) is not None:
        vsv = active_vs.vs
        index = getattr(vsv, "index", None)
        if index is not None:
            ntotal = getattr(index, "ntotal", 0)
            dim = getattr(index, "d", None)
            idx_type = type(index).__name__
            st.success(
                f"Active Vector Index — {effective_vs_name(active_vs).upper()} | "
                f"vectors={ntotal} | dim={dim if dim is not None else '-'} | index={idx_type}"
            )
    else:
        st.info("Active Vector Index 없음. 먼저 인덱스를 빌드하세요.")

    # 업로드(순서: 라벨 → 옵션들 → key)
    uploaded_files = st.file_uploader(
        "문서 업로드 (PDF/DOCX/PPT/TXT)",
        type=["pdf", "docx", "doc", "pptx", "ppt", "txt"],
        accept_multiple_files=True,
        key="uploader",
    )

    build_col, chat_col = st.columns([1, 1], gap="large")  # 1대1화면분할 #"small"|"medium"|"large"

    with build_col:
        st.subheader("1) Vector Index")
        if st.button("Vector Index Build", use_container_width=True):
            if not uploaded_files:
                st.error("최소 1개 파일을 업로드하세요.")
            else:
                tmp_paths = []
                base_dir = Path("./_tmp")
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
                    st.success("인덱싱 및 체인 준비 완료")
                except Exception as e:
                    st.exception(e)

    with chat_col:
        st.subheader("2) Query")
        q = st.text_input(label="질문입력", placeholder="예: 업로드한 문서 내용에서 질문을 해 보세요.", label_visibility="collapsed")
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

if __name__ == "__main__":
    main()
