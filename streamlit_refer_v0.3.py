import os
import io
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from loguru import logger

# =========================
# LangChain 관련 임포트
# =========================
# (중요) ChatOpenAI는 langchain_openai에서 제공
from langchain_openai import ChatOpenAI

# (중요) 허깅페이스 임베딩은 langchain_huggingface에서 제공
#  - 예전의 langchain_community.embeddings.HuggingFaceEmbeddings는 0.2.2에서 Deprecated
from langchain_huggingface import HuggingFaceEmbeddings

# 커뮤니티 벡터스토어/로더
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

# 텍스트 스플리터(별도 패키지)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 체인/메모리
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="RAG Chatbot (OpenAI + HF)", page_icon="🤖")
st.title("RAG Chatbot ✨")
# st.text("RAG (Retrieval Augmented Generation)는 검색 증강 생성을 의미합니다. 이는 LLM이 지니는 통제하기 어려운 한계, 즉 ‘사실 관계 오류 가능성’ (Hallucination)과 ‘맥락 이해의 한계’를 개선하고 보완하는 데 초점을 둔 방법입니다. RAG는 LLM에 외부 지식베이스를 연결하여, 답변 생성 능력은 물론 특정 산업 도메인에 특화된 정보와 최신 데이터를 기반으로 한 사실 관계 파악 능력까지 향상시키는 기술입니다.")


# =========================
# 유틸/헬퍼
# =========================
def _persist_upload(file) -> Path:
    """업로드된 파일을 임시 폴더에 저장 후 경로 반환 (Streamlit 업로드 객체 → 실제 파일)
    - PDF, DOCX, PPTX 모두 동일하게 처리
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="st_docs_"))
    out_path = tmp_dir / file.name
    out_path.write_bytes(file.getbuffer())
    logger.info(f"업로드 파일 저장 경로: {out_path}")
    return out_path


def _load_document(path: Path):
    """파일 확장자에 따라 적절한 문서 로더 반환"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    if ext in (".ppt", ".pptx"):
        return UnstructuredPowerPointLoader(str(path))
    raise ValueError(f"지원하지 않는 파일 형식: {ext}")


# =========================
# 캐시: 임베딩/스플리터
# =========================
@st.cache_resource(show_spinner=False)
def get_hf_embeddings():
    """허깅페이스 임베딩 모델 로드 (캐시)
    - 최초 1회만 모델 파일을 로드하고, 이후 세션에서는 재사용
    - normalize_embeddings=True 는 FAISS 검색 안정성에 도움
    - Streamlit Cloud에서 모델 캐시 디렉터리를 /tmp/hf 로 지정
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="/tmp/hf",
    )


@st.cache_resource(show_spinner=False)
def get_splitter():
    """텍스트 스플리터(캐시)
    - chunk_size/overlap 튜닝: 청크 수 감소 → 인덱싱 속도 향상
    - 문서 성격에 따라 1800/80 → 2000/60 등으로 조정 가능
    """
    return RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=80)


# =========================
# 벡터스토어 빌드
# =========================
def build_vectorstore(doc_paths: List[Path]):
    """업로드 문서들을 로드→청크→임베딩→FAISS 인덱스 생성
    - PDF가 스캔 이미지일 경우 파싱이 느릴 수 있음(가능하면 텍스트 기반 PDF 권장)
    """
    docs = []
    for p in doc_paths:
        loader = _load_document(p)
        docs.extend(loader.load())

    splitter = get_splitter()
    splits = splitter.split_documents(docs)

    embeddings = get_hf_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


# =========================
# 체인 구성 (LLM + Retriever + 메모리)
# =========================
def get_chain(vectorstore, openai_api_key: str):
    """ConversationalRetrievalChain 구성
    - LLM은 OpenAI (gpt-4o-mini) 사용
    - retriever는 FAISS.as_retriever(search_type="mmr") 사용
    - 메모리는 대화 기록을 LLM에 제공 (답변 품질 향상)
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",  # 저렴 & 빠름 (필요 시 gpt-4o 로 교체 가능)
        temperature=0,
        max_retries=3,  # 간단한 재시도 (429 등 레이트리밋 대비)
        timeout=30,     # 너무 오래 기다리지 않도록 타임아웃
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(search_type="mmr")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",             # 간단하게 문서 스터핑 방식 사용(필요시 map_reduce 등 변경 가능)
        memory=memory,
        get_chat_history=lambda h: h,   # 대화 내역을 그대로 전달
        return_source_documents=True,   # 소스 문서 반환(근거 표시용)
        verbose=True,                   # 디버깅 로그
    )
    return chain


# =========================
# 사이드바(UI): API 키/문서 업로드/인덱스 버튼
# =========================
with st.sidebar:
    st.subheader("🔑 OpenAI API Key")

    # 기본값: Streamlit Secrets에 OPENAI_API_KEY가 있다면 자동 사용
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="Streamlit Cloud의 Secrets에 OPENAI_API_KEY를 등록해 두면 자동으로 불러옵니다.",
    )

    uploaded_files = st.file_uploader(
        "문서 업로드 (PDF/DOCX/PPTX)",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
    )

    build_btn = st.button("벡터 인덱스 생성")


# =========================
# 세션 상태 초기화
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# 인덱스 빌드 실행
# =========================
if build_btn:
    if not openai_api_key:
        st.error("🔑 OpenAI API Key를 입력하세요.")
    elif not uploaded_files:
        st.warning("최소 1개 이상의 문서를 업로드하세요.")
    else:
        with st.spinner("문서 인덱싱 중… (최초 1회는 모델 로드로 시간이 걸릴 수 있습니다)"):
            try:
                doc_paths = [_persist_upload(f) for f in uploaded_files]
                vs = build_vectorstore(doc_paths)
                st.session_state.vectorstore = vs
                st.session_state.chain = get_chain(vs, openai_api_key)
                st.success("✅ Vector Index 생성 완료!")
            except Exception as e:
                logger.exception("Vector Index 실패")
                st.error(f"😖 인덱스 생성 실패: {e}")


# =========================
# 질의 UI
# =========================
st.divider()
st.subheader("💬 문서 기반 자연어 질문")
user_q = st.text_input("질문 입력:", placeholder="예: 업로드한 문서의 내용을 질문해  주세요")
ask = st.button("질문하기")


# =========================
# QA 실행
# =========================
if ask:
    if not openai_api_key:
        st.error("OpenAI API Key를 입력하세요.")
    elif st.session_state.chain is None:
        st.warning("먼저 문서를 업로드하고 인덱스를 생성하세요.")
    elif not user_q.strip():
        st.info("질문을 입력하세요.")
    else:
        with st.spinner("응답 생성 중…"):
            try:
                result = st.session_state.chain({"question": user_q})
                answer = result.get("answer", "(답변 없음)")
                sources = result.get("source_documents", [])

                # 화면용 간단한 대화 기록 (메모리에도 저장되지만 UI에 다시 보여주기 위함)
                st.session_state.chat_history.append(("user", user_q))
                st.session_state.chat_history.append(("assistant", answer))

                st.markdown("### 🧠 답변")
                st.write(answer)

                # 근거 문서 표시
                if sources:
                    st.markdown("### 📎 참고 문서")
                    with st.expander("참고 문서와 원문 일부 보기"):
                        for i, doc in enumerate(sources, 1):
                            src = doc.metadata.get("source", f"source_{i}")
                            st.markdown(f"**{i}.** {src}")
                            preview = (doc.page_content or "").strip()
                            if len(preview) > 600:
                                preview = preview[:600] + " …"
                            st.code(preview)
            except Exception as e:
                logger.exception("질문 처리 실패")
                st.error(f"질문 처리 실패: {e}")


# =========================
# 대화 히스토리 표시
# =========================
if st.session_state.chat_history:
    st.divider()
    st.subheader("🗂️ 현재 세션 대화 기록")
    for role, msg in st.session_state.chat_history[-10:]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
