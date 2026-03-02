import os                          
import io                          
import tempfile                    
from pathlib import Path           
from typing import List, Optional  
import streamlit as st             
from loguru import logger 
        
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    from langchain_community.chains import ConversationalRetrievalChain
    
from langchain.memory import ConversationBufferMemory

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    from langchain.schema import SystemMessage, HumanMessage
    
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

try:                                                        
    from langchain_community.vectorstores import FAISS      
except Exception as _e:                                     
    FAISS = None                                            
    _faiss_import_err = _e                                  

from langchain_community.document_loaders import (          
    PyPDFLoader,            
    Docx2txtLoader,                  
    UnstructuredPowerPointLoader,
    TextLoader,                                            
) 

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:                                
    import tiktoken                 
except Exception as _tk_err:        
    tiktoken = None                 
    _tiktoken_import_err = _tk_err  

TOKEN_ENCODING_NAME = "cl100k_base" 
TOKEN_CHUNK_SIZE = 800            
TOKEN_CHUNK_OVERLAP = 80 
 
# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("CSS RAG Chatbot v0.7 ✨")  
st.caption("문서가 업로드된 경우는 RAG 기반으로 답변하고, 업로드되지 않은 경우는 LLM 기반으로 답변을 제공합니다.")

# =========================
# 유틸리티(헬퍼 함수) 모음
# =========================
def _persist_upload(file) -> Path:                      
    tmp_dir = Path(tempfile.mkdtemp(prefix="st_docs_")) 
    out_path = tmp_dir / file.name                      
    out_path.write_bytes(file.getbuffer()) 
    logger.info(f"업로드 파일 저장 경로: {out_path}")      
    return out_path                                     

def _load_document(path: Path):                        
    ext = path.suffix.lower()                          
    if ext == ".pdf":                                  
        return PyPDFLoader(str(path))                  
    if ext in (".doc",".docx"):                        
        return Docx2txtLoader(str(path))               
    if ext in (".ppt", ".pptx"):                       
        return UnstructuredPowerPointLoader(str(path)) 
    if ext == ".txt":                                  
        return TextLoader(str(path), encoding="utf-8") 
    raise ValueError(f"😖 지원하지 않는 파일 형식: {ext}")

# ============================
# 캐시: 임베딩/스플리터 관련 코드 블록 영역
# ============================
@st.cache_resource(show_spinner=False)
def get_hf_embeddings():              
    return HuggingFaceEmbeddings(     
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"},                             
        encode_kwargs={"normalize_embeddings": True},  
        cache_folder="/tmp/hf",                                     
    )

def _get_tiktoken_encoding(name: str):         
    if tiktoken is None:                       
        raise RuntimeError(                    
            f"tiktoken을 불러오지 못했습니다. (원인: {repr(_tiktoken_import_err)})\n" 
            "requirements.txt에 'tiktoken'을 추가하고 재배포하세요."
        )
    try:                                            
        return tiktoken.get_encoding(name)          
                                                    
    except Exception:                               
        return tiktoken.get_encoding("cl100k_base") 

@st.cache_resource(show_spinner=False)         
def get_token_splitter(                        
    chunk_tokens: int = TOKEN_CHUNK_SIZE,      
    overlap_tokens: int = TOKEN_CHUNK_OVERLAP, 
    encoding_name: str = TOKEN_ENCODING_NAME,  
):
    _ = _get_tiktoken_encoding(encoding_name)  
                                               
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder( 
        chunk_size=chunk_tokens,                                 
        chunk_overlap=overlap_tokens,                            
        encoding_name=encoding_name,                             
    )
 
# =========================
# 벡터스토어 빌드
# =========================
def build_vectorstore(doc_paths: List[Path]):
    if FAISS is None:                        
        raise RuntimeError(                  
            f"😖 FAISS 모듈을 불러오지 못했습니다. (원인: {repr(_faiss_import_err)})\n"
            "CPU 환경에서는 requirements.txt에 'faiss-cpu'를 설정해 주세요.\n"
            "GPU 환경에서는 requirements.txt에 'faiss-gpu'를 설정해 주세요."
        )

    docs = []                              
    for p in doc_paths:                    
        loader = _load_document(p)         
        docs.extend(loader.load()) 
    splitter = get_token_splitter()
    splits = splitter.split_documents(docs)
    embeddings = get_hf_embeddings() 
    vectorstore = FAISS.from_documents(splits, embeddings) 
    return vectorstore                                     
   
# ===============================
# 체인 구성 (LLM + Retriever + 메모리)
# ===============================
def get_chain(vectorstore, openai_api_key: str): 
    llm = ChatOpenAI(                            
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",  # 저렴,빠름 (필요 시 gpt-4o 로 교체 가능)
        temperature=0,        # 0: 창의성 낮추고, 일관성 높은 답변 생성
        max_retries=3,        # 간단한 재시도 (429 등 레이트리밋 대비)
        timeout=10,           # 10초 이상 걸리면 요청 중단
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
        chain_type="stuff",            
        memory=memory,                 
        get_chat_history=lambda h: h,  
        return_source_documents=True,  
        verbose=True,                  
    )                                  
    return chain 

# =====================================================
# LLM 단독 (NOT RAG) 응답 헬퍼 >> RAG 구성 없을 경우 그냥 LLM을 사용
# =====================================================
def answer_without_rag(question: str, openai_api_key: str) -> str: 
    llm = ChatOpenAI(                                              
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3,
        timeout=10,
    )
    sys = SystemMessage(content="너는 간결한 조수다. 모든 답변은 2~3문장 이내로 핵심만 요약해서 말해라.")    # 프롬프트 중 시스템 롤에 해당(role="system")
    user = HumanMessage(content=question)       
    resp = llm.invoke([sys, user])     
    return getattr(resp, "content", str(resp)) 
    
# =================================
# 사이드바(UI): API 키/문서 업로드/인덱스 버튼
# ================================= 
with st.sidebar:
    st.subheader("🔑 OpenAI API Key")
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""    # 기본값: Streamlit Secrets에 OPENAI_API_KEY가 있다면 사용
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="Streamlit Cloud의 Secrets에 OPENAI_API_KEY를 등록하면 자동으로 사용됩니다.",
    )

    uploaded_files = st.file_uploader(
        "1MB 미만 문서 업로드 권장 (PDF/DOCX/PPTX/TXT)", 
        type=["pdf", "doc", "docx", "ppt", "pptx", "txt"], 
        accept_multiple_files=True,
    )

    build_btn = st.button("벡터 인덱스 생성")
    delete_btn = st.button("벡터 인덱스 삭제") 

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
# 벡터 인덱스 생성
# =========================
if build_btn:                                     
    if not openai_api_key:                        
        st.error("🔑 OpenAI API Key를 입력하세요.") 
    elif not uploaded_files:                      
        st.warning("⚠️ 최소 1개 이상의 문서를 업로드하세요.")   
    else:                                                  
        with st.spinner("🏃🏻 Vector Index 생성 중… (최초에는 로딩 시간이 걸릴 수 있습니다.)"):
            try:                                                                       
                doc_paths = [_persist_upload(f) for f in uploaded_files]               
                                                                                       
                vs = build_vectorstore(doc_paths)                                      
                                                                                       
                st.session_state.vectorstore = vs                                      
                st.session_state.chain = get_chain(vs, openai_api_key)                 
                                                                                       
                st.success("✅ Vector Index 생성 완료! (RAG: ON)")                      
            except Exception as e:                                                     
                logger.exception("Vector Index 생성 실패")                              
                st.error(f"😖 Vector Index 생성 실패: {e}")                             

# =========================
# 벡터 인덱스 삭제
# =========================
if delete_btn:
    if st.session_state.get("vectorstore") is None and st.session_state.get("chain") is None:
        st.info("⛔ 삭제할 Vector Index가 없습니다.")
    else:
        with st.spinner("🏃🏻 Vector Index 삭제 중… "): 
            try:
                vs = st.session_state.get("vectorstore")
                ch = st.session_state.get("chain")

                try:
                    if vs is not None and hasattr(vs, "index"):
                        idx = getattr(vs, "index")
                        if hasattr(idx, "reset"):
                            idx.reset()
                except Exception:
                    logger.debug("FAISS Vector Index Reset 시도 중 오류 발생(무시):", exc_info=True)

                st.session_state.vectorstore = None
                st.session_state.chain = None

                import gc
                gc.collect()

                st.success("❎ Vector Index 및 RAG Chain 삭제 완료! (RAG: OFF)")
            except Exception as e:
                logger.exception("Vector Index 삭제 실패")
                st.error(f"😖 Vector Index 삭제 실패: {e}")

# =========================
# 질의 UI
# =========================
st.divider()                             
st.subheader("💬 문서 기반 자연어 질의") 
user_q = st.text_input("질문 입력:", placeholder="예: 업로드한 문서 내용에서 질문을 해 보세요.") 
ask = st.button("질문하기")                 

# ==============================================
# QA 실행: RAG-ON 경우 vs. RAG-OFF 경우 >> 폴백로직포함🧠
# ==============================================
# 폴백(fallback) == (의존성 없을 때/예외 발생할 때의) 대체행동

if ask:                                                             
    if not openai_api_key:                                          
        st.error("🔑 OpenAI API Key를 입력하세요.")
    elif not user_q.strip():                                        
        st.info("질문을 입력하세요.")
    else:                                                           
        # 1) 인덱스/체인 준비 여부 확인
        if st.session_state.chain is None:                          
            # 🔁 폴백(fallback): RAG-OFF 경로                        
            with st.spinner("LLM 답변 생성 중… (RAG: OFF)"):                      
                try:                                                            
                    answer = answer_without_rag(user_q, openai_api_key)         
                    st.session_state.chat_history.append(("user", user_q))      
                    st.session_state.chat_history.append(("assistant", answer)) 

                    st.markdown("### 👽 답변  `RAG: OFF`")                        
                    st.text(answer)                                               
                    st.info("👽 RAG 비활성화 상태입니다. Vector Index가 없기에 LLM으로 일반적인 답변을 제공합니다.")

                except Exception as e:                            
                    logger.exception("LLM-Only 질문 처리 실패")     
                    st.error(f"😖 질문 처리 실패(LLM-Only): {e}")   
        else:                                                     
            # ✅ RAG-ON 경로
            with st.spinner("RAG 답변 생성 중… (RAG: ON)"):                          
                try:                                                               
                    result = st.session_state.chain({"question": user_q})          
                                                                                   
                    answer = result.get("answer", "(답변 없음)")                     
                    sources = result.get("source_documents", [])                   
                                                                                   
                    st.session_state.chat_history.append(("user", user_q))         
                    st.session_state.chat_history.append(("assistant", answer))    
                                                                                   
                    st.markdown("### 🧠 답변  `RAG: ON`")                           
                    # st.write(answer)                                             
                    st.text(answer)                                                
                                                                                   
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
                        st.info("해당 질문과 직접적으로 매칭되는 문서 청크를 찾지 못했습니다. (질문을 더 구체화하거나 인덱싱 범위를 늘려 보세요.)")
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
