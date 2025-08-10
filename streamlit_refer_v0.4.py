"""
streamlit_refer.py — RAG(Chat + Retrieval) 앱 (OpenAI LLM + HuggingFace 임베딩)

[구성 요약]
- LLM: OpenAI Chat (gpt-4o-mini; 저렴/빠름)
- Embeddings: HuggingFace (paraphrase-MiniLM-L6-v2)
- 문서 로더: PDF, DOCX, PPTX
- 벡터스토어: FAISS
- 체인: ConversationalRetrievalChain (대화형 RAG)
- 폴백: 인덱스가 없으면 LLM-only로 간단 답변 + "RAG: OFF" 안내

[왜 이렇게?]
- 비용/유연성: LLM은 OpenAI, 임베딩은 HF로 분리 → 비용·성능 균형
- 최신 LangChain 패키지 분리 반영: langchain-openai / langchain-huggingface / langchain-community
- 초기 로드/인덱싱 성능: 임베딩/스플리터 캐시(@st.cache_resource)
- 안정성: LLM 호출에 timeout/retries, FAISS 미설치 안내 처리

[requirements.txt 예시]
streamlit>=1.32
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-text-splitters>=0.2.0
langchain-huggingface>=0.1.0
sentence-transformers
torch
pypdf
docx2txt
unstructured
unstructured[pptx]
tiktoken
loguru
faiss-cpu   # ← Streamlit Cloud/일반 CPU 환경
"""

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

# 커뮤니티 벡터스토어/로더 (FAISS 임포트는 환경에 따라 실패 가능 → 안전 가드)
try:
    from langchain_community.vectorstores import FAISS
except Exception as _e:  # ImportError 포함
    FAISS = None
    _faiss_import_err = _e

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

# (LLM-only 폴백용) 메시지 타입
from langchain.schema import SystemMessage, HumanMessage


# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="RAG Chatbot (OpenAI + HF)", page_icon="🤖")
st.title("RAG Chatbot ✨")
st.caption("문서가 업로드된 경우는 RAG로 답변하고, 업로드되지 않은 경우는 LLM에서 답변을 제공합니다.")


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
    raise ValueError(f"😖 지원하지 않는 파일 형식: {ext}")


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
    - FAISS 미설치 시 친절한 에러 안내
    """
    if FAISS is None:
        raise RuntimeError(
            f"😖 FAISS 모듈을 불러오지 못했습니다. (원인: {repr(_faiss_import_err)})\n"
            "CPU 환경에서는 requirements.txt에 'faiss-cpu'를 추가해 주세요.\n"
            "GPU(CUDA) 환경에서는 'faiss-gpu'를 사용합니다."
        )

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
        timeout=20,     # 너무 오래 기다리지 않도록 타임아웃
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
# LLM 단독(비 RAG) 응답 헬퍼
# =========================
def answer_without_rag(question: str, openai_api_key: str) -> str:
    """문서 인덱스가 없을 때, LLM만으로 간결한 답변을 생성
    - 2~3문장 이내로 짧고 핵심만
    - RAG가 아님을 UI에서 별도 안내
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3,
        timeout=20,
    )
    sys = SystemMessage(content="너는 간결한 조수다. 모든 답변은 2~3문장 이내로 핵심만 요약해서 말해라.")
    user = HumanMessage(content=question)
    resp = llm.invoke([sys, user])  # Chat 모델의 invoke 사용
    return getattr(resp, "content", str(resp))


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
        help="Streamlit Cloud의 Secrets에 OPENAI_API_KEY를 등록하면 자동으로 사용됩니다.",
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
        with st.spinner("문서 벡터 인덱싱 중… (최초에는 모델 로딩 시간이 걸릴 수 있습니다)"):
            try:
                doc_paths = [_persist_upload(f) for f in uploaded_files]
                vs = build_vectorstore(doc_paths)
                st.session_state.vectorstore = vs
                st.session_state.chain = get_chain(vs, openai_api_key)
                st.success("✅ Vector Index 생성 완료! (RAG 가능)")
            except Exception as e:
                logger.exception("Vector Index 실패")
                st.error(f"😖 인덱스 생성 실패: {e}")


# =========================
# 질의 UI
# =========================
st.divider()
st.subheader("💬 문서 기반 자연어 질문")
user_q = st.text_input("질문 입력:", placeholder="예: 업로드한 문서의 핵심만 간단히 알려주세요")
ask = st.button("질문하기")


# =========================
# QA 실행 (RAG ON/OFF 폴백 포함)
# =========================
if ask:
    if not openai_api_key:
        st.error("🔑 OpenAI API Key를 입력하세요.")
    elif not user_q.strip():
        st.info("질문을 입력하세요.")
    else:
        # 1) 인덱스/체인 준비 여부 확인
        if st.session_state.chain is None:
            # 🔁 폴백: 문서 인덱스가 없으므로 LLM 단독 간단 답변
            with st.spinner("LLM 답변 생성 중… (RAG OFF)"):
                try:
                    answer = answer_without_rag(user_q, openai_api_key)
                    st.session_state.chat_history.append(("user", user_q))
                    st.session_state.chat_history.append(("assistant", answer))

                    st.markdown("### 🧠 답변  `RAG: OFF`")
                    st.write(answer)
                    st.info("RAG 비활성화 상태입니다. 업로드한 문서/인덱스가 없어 일반 LLM으로 간단 답변을 제공했습니다.")
                except Exception as e:
                    logger.exception("LLM-only 질문 처리 실패")
                    st.error(f"😖 질문 처리 실패(LLM-only): {e}")
        else:
            # ✅ RAG 경로
            with st.spinner("RAG 응답 생성 중… (RAG ON)"):
                try:
                    result = st.session_state.chain({"question": user_q})
                    answer = result.get("answer", "(답변 없음)")
                    sources = result.get("source_documents", [])

                    st.session_state.chat_history.append(("user", user_q))
                    st.session_state.chat_history.append(("assistant", answer))

                    st.markdown("### 🧠 답변  `RAG: ON`")
                    st.write(answer)

                    # 근거 문서 표시
                    if sources:
                        st.markdown("### 💡 참고 문서")
                        with st.expander("참고 문서 위치 및 원문 일부 보기"):
                            for i, doc in enumerate(sources, 1):
                                src = doc.metadata.get("source", f"source_{i}")
                                st.markdown(f"**{i}.** {src}")
                                preview = (doc.page_content or "").strip()
                                if len(preview) > 600:
                                    preview = preview[:600] + " …"
                                st.code(preview)
                    else:
                        st.info("해당 질문과 직접적으로 매칭되는 문서 청크를 찾지 못했습니다. (필요 시 질문을 더 구체화해 보세요.)")
                except Exception as e:
                    logger.exception("질문 처리 실패(RAG)")
                    st.error(f"😖 질문 처리 실패(RAG): {e}")


# =========================
# 대화 히스토리 표시
# =========================
if st.session_state.chat_history:
    st.divider()
    st.subheader("🗂️ 세션 아카이브")
    for role, msg in st.session_state.chat_history[-10:]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
