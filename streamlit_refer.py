import os
import io
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from loguru import logger

# 변경: LangChain 패키지 분리로 인한 import 경로 수정
from langchain_openai import ChatOpenAI  # 기존: from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings  # 기존: from langchain.embeddings import ...
from langchain_community.vectorstores import FAISS  # 기존: from langchain.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

# 변경: text splitters가 별도 패키지로 이동됨
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -----------------------------
# 업로드 파일 저장 함수
# -----------------------------
def _persist_upload(file) -> Path:
    """업로드된 파일을 임시 폴더에 저장 후 경로 반환"""
    suffix = Path(file.name).suffix
    tmp_dir = Path(tempfile.mkdtemp(prefix="st_docs_"))
    out_path = tmp_dir / file.name
    out_path.write_bytes(file.getbuffer())
    logger.info(f"업로드 파일 저장 경로: {out_path}")
    return out_path

# -----------------------------
# 문서 로드 함수
# -----------------------------
def _load_document(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    if ext in (".ppt", ".pptx"):
        return UnstructuredPowerPointLoader(str(path))
    raise ValueError(f"지원하지 않는 파일 형식: {ext}")

# -----------------------------
# 벡터스토어 생성
# -----------------------------
def build_vectorstore(doc_paths: List[Path]):
    """HF 임베딩을 이용하여 FAISS 인덱스 생성"""
    docs = []
    for p in doc_paths:
        loader = _load_document(p)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# -----------------------------
# 체인 구성
# -----------------------------
def get_chain(vectorstore, openai_api_key: str):
    # 변경: ChatOpenAI 임포트 경로 및 모델명 최신화
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 변경: as_retriever() 불필요/오타 파라미터 제거
    retriever = vectorstore.as_retriever(search_type="mmr")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
    )
    return chain

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Chat (Patched)", page_icon="🧩")

st.title("RAG Chat with Your Files ✨")

with st.sidebar:
    st.subheader("🔑 API & 설정")

    # 기본값: secrets에서 불러오기
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="Secrets에 OPENAI_API_KEY를 저장할 수도 있습니다"
    )

    uploaded_files = st.file_uploader(
        "문서 업로드 (PDF/DOCX/PPTX)",
        type=["pdf", "docx", "pptx"],  # 변경: pptx 지원 추가
        accept_multiple_files=True,
    )

    build_btn = st.button("인덱스 생성/재생성")

# 세션 상태 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 인덱스 빌드
if build_btn:
    if not openai_api_key:
        st.error("OpenAI API Key를 입력하세요.")
    elif not uploaded_files:
        st.warning("최소 1개 이상의 문서를 업로드하세요.")
    else:
        with st.spinner("문서 인덱싱 중…"):
            try:
                doc_paths = [_persist_upload(f) for f in uploaded_files]
                vs = build_vectorstore(doc_paths)
                st.session_state.vectorstore = vs
                st.session_state.chain = get_chain(vs, openai_api_key)
                st.success("벡터 인덱스 생성 완료!")
            except Exception as e:
                logger.exception("인덱스 생성 실패")
                st.error(f"인덱스 생성 실패: {e}")

# 채팅 입력
st.divider()
st.subheader("💬 문서 기반 질문하기")
user_q = st.text_input("질문 입력", placeholder="예: 문서 핵심 요약 알려줘")
ask = st.button("질문하기")

# QA 실행
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

                st.session_state.chat_history.append(("user", user_q))
                st.session_state.chat_history.append(("assistant", answer))

                st.markdown("### 🧠 답변")
                st.write(answer)

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

# 대화 히스토리 표시
if st.session_state.chat_history:
    st.divider()
    st.subheader("🗂️ 현재 세션 대화 기록")
    for role, msg in st.session_state.chat_history[-10:]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
