# =============================
# 코드시작 (임포트 상단 배치)
# =============================
# import xxx as x --> xxx라는 모듈/패키지 전체를 로드해서 현재 네임스페이스에 x라는 alias로 붙임(바인딩).  
#     import math --> print(math.pi)
# from yyy import xxx as x --> 모듈/패키지 yyy 안에 정의된 이름 xxx(클래스/함수/변수/서브모듈 등)를 가져와서 직접 현재 네임스페이스에 x라는 alias로 붙임(바인딩). 
#     from math import pi  --> print(pi)  
# from yyy import a, b as c --> 먼저 yyy 모듈/패키지을 찾고 로드 >> yyy 모듈에서 이름 a를 찾아 현재 네임스페이스에 a라는 이름으로 바인딩 >> yyy 모듈에서 이름 b를 찾아 현재 네임스페이스에 c라는 alias으로 바인딩(즉 c가 yyy.b를 가리킴) >> a, c 2개 함수만 가져온 것 
# from yyy import a, b as c, d --> 먼저 yyy 모듈/패키지을 찾고 로드 >> a 함수, b as c(b 함수를 c alias 바인딩), d 함수 >> a, c, d 3개 함수만 가져온 것 

import os                          # 운영체제 경로, 환경변수 제어 — 파일 저장, 경로 조작 등에 사용
import io                          # 메모리 버퍼 I/O — BytesIO, StringIO 등 파일처럼 다루는 객체 제공
import tempfile                    # 임시 파일/폴더 생성 — 업로드 파일 저장 후 처리에 사용
from pathlib import Path           # 경로 객체화 — 경로 조작을 직관적이고 플랫폼 독립적으로 수행
from typing import List, Optional  # 타입 힌트 — List, Optional 등으로 함수 인자/반환 타입 명시
import streamlit as st             # Streamlit — 대화형 웹 앱 UI 생성 라이브러리
from loguru import logger          # Loguru — 깔끔하고 강력한 로깅 기능 제공

# LangChain 핵심/유틸
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage # LangChain에서 대화(채팅) 메시지 타입 두 개를 가져오는

# OpenAI / HF 연결 모듈 (계속 사용)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# langchain_community에서 기본 제공하는 문서로더
# FAISS 파이쓰는 환경에 따라 import 실패 가능하므로 try/except로 처리
try:                                                        # try 블록 시작, 다음에 오는 import 시도를 하고, 실패하면 except로 분기함.
    from langchain_community.vectorstores import FAISS      # langchain_community 패키지 안의 vectorstores 모듈에서 FAISS 파이쓰 클래스를 가져오기
except Exception as _e:                                     # import 중에 어떤 예외 (ImportError나 ModuleNotFoundError)가 발생하면 여기로 오며, _e라는 변수에 예외 객체가 담기게 됨.
    FAISS = None                                            # 예외 발생 시 FAISS 파이쓰 변수를 None으로 설정함.  
    _faiss_import_err = _e                                  # 예외 객체(어떤 이유로 import가 실패했는지에 대한 정보)를 _faiss_import_err 변수에 저장함.
                                                            # import 에러가 날 수 있으니, 안전하게 None으로 처리하고 대체 저장소를 선택하도록 코드를 짜는 것이 일반적임. 
                                                            # 그래서 코드에서 if FAISS is None:으로 분기하여 Chroma, Milvus, Pinecone(원격), Weaviate 등 다른 vectorstore로 폴백할 수도 있음.
from langchain_community.document_loaders import (          # 여러 문서 로더 클래스를 한 번에 import
    PyPDFLoader,            
    Docx2txtLoader,                  
    UnstructuredPowerPointLoader,
    TextLoader,                                             # return TextLoader(str(path), encoding="utf-8") >> TextLoader 사용 시 인코딩을 utf-8 및 euc-kr 등으로 설정 가능
) 

# =============================
# 텍스트스플리터 
# 토큰: 토크나이저사용 vs 문자(기본)사용
# =============================
# 1) 토크나이저는 “토큰 수를 알려주는 도구”이며,
# 2) 텍스트스플리터는 의미 단위(문단/문장/중첩 규칙), 중복(overlap), 메타데이터를 고려해 실제로 청크를 만드는 도구
# 3) 따라서 토크나이저와 텍스트스플리터는 함께 보완하며 사용해야하며 아래는 그 상세 이유:
# - 토크나이저만 있으면 토큰 길이 제어는 가능하지만 문장/문단 경계를 무시하고 잘라서 의미 파괴(중간 문장 잘림)가 일어남.
# - 텍스트스플리터는 separator, chunk_size, chunk_overlap, prefer_sentence_boundary 같은 규칙을 적용해 의미론적으로 더 자연스러운 청크를 만듦.
# - 검색,임베딩,LLM 컨텍스트 윈도우 최적화: 스플리터는 임베딩/검색 효율과 LLM 입력 한계를 실제 운영 요구에 맞춰 관리.
# - 메모리/메타데이터: 각 청크에 출처·위치 정보를 붙이기 쉬움(검색 결과에 근거 표시에 필요).
# - 성능/비용: 적절한 크기/겹침으로 불필요한 토큰 낭비(=비용)를 줄일 수 있음.
#
# Splitter(텍스트스플리터):
#   RecursiveCharacterTextSplitter 클래스는 문단/문장 경계와 지정한separators구분자(예:["\n\n", "\n", ". ", "? ", "! ", " ", ""])를 활용해, 
#   지정한 문자 길이(chunk_size)와 청크 간 겹침(chunk_overlap)을 적용하여 의미를 최대한 보존하면서 문자 기반 청크를 생성하는 도구임.
#     + 텍스트 스플리터
#        -- 토크나이저기반 (tiktoken 사용)
#        -- 문자기반(기본) 스플리터
#      
#     + 텍스트 스플리터
#       -- 토크나이저기반 (tiktoken 사용)
#       -- tiktoken이 설치되어 있어야 함
#           from langchain_text_splitters import RecursiveCharacterTextSplitter
#           splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#                       chunk_size=800,       
#                       chunk_overlap=80,
#                       encoding_name="cl100k_base"
#                   )
#           chunks = splitter.split_documents(documents)
#
#       -- 문자기반(기본) 스플리터 
#           from langchain_text_splitters import RecursiveCharacterTextSplitter
#           splitter = RecursiveCharacterTextSplitter(
#                       chunk_size=1800,
#                       chunk_overlap=200,
#                       separators=["\n\n", "\n", " ", ""]  # 수동 지정한 separators(구분자)
#                   )
#           chunks = splitter.split_documents(documents)
#           
# langchain_text_splitters 라이브러리에서 RecursiveCharacterTextSplitter 클래스를 가져오기
from langchain_text_splitters import RecursiveCharacterTextSplitter

# tiktoken 라이브러리 >> 내장인코딩(encoding)리소스 == 토그나이저 >> cl100k_base, r50k_base, p50k_base
# 참고로 cl100k_base 코크나이저는 디코딩/인코딩 양방향 모두 가능하지만 "인코딩(encoding)리소스" 라고만 일반적으로 부른다.
# 
# cl100k_base는 모델이 아니고 토크나이저(토큰 인코더/디코더)이다
# cl100k_base 토크나이저는 tiktoken 라이브러리에서 제공된다 (import tiktoken)
# cl100k_base 토크나이저는 "토큰 수만 알려주는 도구"일 뿐, 문단/문장 경계(구분자)를 자동으로 생성 및 판단하지 않는다
# cl100k_base 토크나이저는 OpenAI가 권장하는 최신 인코딩 중 하나로, tiktoken 라이브러리에서 제공됩니다. 토큰 수를 정확히 셀 때, 또는 토큰 기준 스플리터를 만들 때 사용합니다.
# cl100k_base 토크나이저는 텍스트를 토큰(정수 ID)으로 바꾸는 규칙(사전 + BPE 분해 방식)을 제공하고, OpenAI 계열 모델들과 호환되도록 설계된 인코딩입니다.
# 
# cl100k_base는 인코딩(텍스트->토큰)과 디코딩(토큰->텍스트) 변환을 양방향으로 처리하는 토크나이저(토큰 인코더/디코더)이다
# 내부적으로는 바이트기반 BPE (Byte-Level BPE) 스타일의 분해 알고리즘과 토큰사전(vocabulary)을 사용해 문자열을 토큰ID로 인코딩하고, 반대로 토큰ID 시퀀스를 원래(또는 재생성가능한) 문자열로 디코딩한다
# BPE (Byte Pair Encoding)? 자주 함께 나오는 문자(혹은 바이트) 쌍을 반복적으로 하나의 토큰으로 합쳐서 어휘(vocab)를 만드는 압축·토크나이저 기법
# 
# 모델과 토크나이저(encoding) 차이점
# 
# 토크나이저 (예: cl100k_base)
#   텍스트를 토큰 ID의 시퀀스로 변환하고, 다시 디코딩하는 규칙 (인코딩 표준)
#   모델 입력/출력의 형식을 정의할 뿐, 텍스트 생성/추론(문장 생성)의 능력은 없음
# 모델 (예: gpt-4o, gpt-3.5-turbo 등)
#   토큰을 받아서 실제로 확률 분포를 계산하고 다음 토큰을 생성하는 "추론엔진"이다
#   어떤 토크나이저(인코딩)를 쓰는지는 모델 사양과 모델 호환성에 따라 달라진다 (따라서 토크나이저와 모델은 역할이 다름)
# 

# 토크나이저(tiktoken) 설치 여부 확인 및 예외 처리
try:                                # try 블록 시작, 다음에 오는 코드(여기서는 import tiktoken)를 시도하고, 실패하면 except로 분기됨
    import tiktoken                 # tiktoken 모듈을 가져오기
except Exception as _tk_err:        # 이전 import tiktoken에서 모듈없음이나 버전문제 등으로 어떤 예외가 발생하면 이 블록으로 들어와서 예외 객체를 _tk_err라는 이름으로 받도록함.
    tiktoken = None                 # 예외 발생 시 tiktoken 변수를 None으로 설정
    _tiktoken_import_err = _tk_err  # 예외 객체(_tk_err)를 _tiktoken_import_err에 저장함

# ============================
# 토큰 청크 설정 (필요 시 여기만 바꾸면 됨)
# ============================
TOKEN_ENCODING_NAME = "cl100k_base"  # TOKEN_ENCODING_NAME 변수에 GPT-4/4o 계열과 호환되는 토크나이저 인코딩 이름 "cl100k_base" 할당
                                     # tiktoken 같은 라이브러리에서 어떤 인코딩(토큰화 규칙)을 쓸지 지정하는 문자열임.
                                     # 모델별(혹은 토크나이저별) 인코딩이 달라서 동일한 문자열이라도 토큰 수가 달라질 수 있음.
TOKEN_CHUNK_SIZE = 800               # 청크 하나의 최대 토큰 수 (문자 아님!)
                                     # 하나의 문서 청크(조각)가 가질 수 있는 최대 토큰 수
                                     # 문자(글자 수)가 아니라 토큰 단위라는 점 중요함 — 토큰은 어절/부분어절/부분문자열 단위로 나뉠 수 있음.
TOKEN_CHUNK_OVERLAP = 80             # 청크 간 겹침(토큰 단위)
                                     # 연속된 청크들 사이에 중복으로 포함될 토큰 수를 의미함(슬라이딩 윈도우 방식)
                                     # 문장/문단 경계에서 정보 손실을 줄이고, 검색 시 문맥 연결을 유지하게 함. 또한 요약·임베딩 시 경계에서 끊긴 정보 보완용.
                                     # 보통 chunk_size의 5~20% 수준을 많이 사용(여기선 80/800 = 10%). 적당한 값은 문서 특성에 따라 조정.
 
# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")                 # 브라우저 탭 제목
st.title("CSS RAG Chatbot v0.7 ✨")                                          # 웹페이지 상단 큰제목(헤더)
st.caption("문서가 업로드된 경우는 RAG로 답변하고, 업로드되지 않은 경우는 LLM에서 답변을 제공합니다.") # st.caption은 짧은 안내문에 적합, 길거나 포멧있는 텍스트는 st.markdown() 또는 st.write() 사용


# =========================
# 유틸리티(헬퍼 함수) 모음
# =========================
def _persist_upload(file) -> Path:                      # _persist_upload 함수명에 file 파라미터로 반환타입힌트로(-> Path) pathlib.Path 객체를 반환 예정
    tmp_dir = Path(tempfile.mkdtemp(prefix="st_docs_")) # mkdtemp가 자동으로 "st_docs_" 뒤에 랜덤 8자리 고정 문자열을 붙여 새 디렉토리를 만들고 그 전체 경로(문자열)를 반환 >> import tempfile, os
                                                        # 즉 OS temp 영역에 고유한 새디렉토리 생성(예:/tmp/st_docs_abcd1234) 후 경로 문자열을 반환
    out_path = tmp_dir / file.name                      # 업로드된 파일의 원래 이름(file.name)을 temp 디렉토리 내에 파일명으로 사용한 Path를 생성
    out_path.write_bytes(file.getbuffer())              # 파일 내용 디스크 기록 >> out_path.write_bytes(...)는 해당 바이트 데이터를 파일로 써서 실제 파일을 생성
                                                        # file.getbuffer()는 업로드된 파일의 바이트 내용을 가리키는 객체(보통 memoryview)를 리턴(반환)해 줌
                                                        # memoryview는 별도의 복사 없이(또는 최소한의 복사로) 데이터의 일부분을 읽거나 슬라이스할 수 있는 장점 존재함
                                                        # file.getbuffer() vs file.read() 차이점
                                                        # - file.read(): 파일 내용을 복사해서 bytes 객체를 반환 >> 복사본(메모리) 추가 차지
                                                        # - file.getbuffer(): 원본 데이터(버퍼)를 가리키는 뷰 객체(memoryview)를 반환 >> 복사본(메모리) 없이 참조
    logger.info(f"업로드 파일 저장 경로: {out_path}")          # logger(예: loguru.logger)를 사용해 저장된 파일 경로를 info 레벨로 기록 >> 디버깅용
    return out_path                                     # 저장된 파일의 pathlib.Path 객체를 호출자에게 반환 >> 호출자는 반환된 경로로 파일을 열어 로더에 전달하거나 파싱하는 등의 후속 처리함.


def _load_document(path: Path):                         # _load_document라는 함수이며 path 인자는 pathlib.Path 타입을 기대한다는 의미(타입힌트) 
    ext = path.suffix.lower()                           # path의 확장자(마지막 점을 포함한 부분, 예:.pdf)를 lower 소문자로 가져와 ext에 저장 >> path.suffix는 마지막 확장자만 반환
    if ext == ".pdf":                                   # 확장자가 .pdf인지 검사
        return PyPDFLoader(str(path))                   # .pdf면 PyPDFLoader 인스턴스를 생성해 반환 >> str(path)로 Path를 문자열 파일 경로로 변환해서 로더에 전달
    if ext in (".doc",".docx"):                         # 확장자가 .doc 또는 docx인지 검사
        return Docx2txtLoader(str(path))                # .docx면 Docx2txtLoader 인스턴스를 반환 >> str(path)로 Path를 문자열 파일 경로로 변환해서 로더에 전달
    if ext in (".ppt", ".pptx"):                        # 확장자가 .ppt 또는 .pptx인지 검사
        return UnstructuredPowerPointLoader(str(path))  # .pptx면 UnstructuredPowerPointLoader 인스턴스를 반환 
    if ext == ".txt":                                   # 확장자가 .txt인지 검사 
        return TextLoader(str(path), encoding="utf-8")  # 텍스트 파일이면 TextLoader를 반환하고, 인코딩을 utf-8로 명시 >> euc-kr 인코딩으로 변경도 가능
    raise ValueError(f"😖 지원하지 않는 파일 형식: {ext}")      # 위의 어느 분기도 만족하지 않으면(지원하지 않는 확장자면) ValueError 예외를 발생



# ============================
# 캐시: 임베딩/스플리터 관련 코드 블록 영역
# ============================
@st.cache_resource(show_spinner=False)          # show_spinner=False 옵션은 Streamlit이 로딩 중에 보여주는 회전 스피너를 표시하지 않음을 의미(기본 스피너 숨김)
def get_hf_embeddings():                        # get_hf_embeddings() 함수(인자없음)는 캐시된 임베딩 객체(HuggingFaceEmbeddings)를 반환
    return HuggingFaceEmbeddings(               # HuggingFaceEmbeddings 클래스의 인스턴스를 생성해 반환 >> 아래의 허깅페이스 임베딩 모델 로드 (캐시)
                                                # 이 객체는 내부적으로 Hugging Face Transformer 모델과 토크나이저를 로드하고, embed_documents() 및 embed_query() 같은 메서드로 텍스트를 임베딩(벡터)으로 변환
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",     # model_name: 사용할 Hugging Face 모델의 식별자(허깅페이스 허브 이름) >> paraphrase-MiniLM-L6-v2는 경량급 문장 임베딩 모델
        model_kwargs={"device": "cpu"},                                 # model_kwargs: 모델 로드 시 전달할 추가 설정 >> {"device": "cpu"}는 모델을 CPU에서 실행하도록 설정 >> GPU 사용가능하면 "cuda" 또는 {"device": "cuda:0"} 설정
        encode_kwargs={"normalize_embeddings": True},                   # encode_kwargs: 임베딩(인코딩) 호출 시 사용할 옵션들 >> {"normalize_embeddings": True}는 생성된 벡터를 정규화(L2-normalize) 하여 반환
                                                                        # 그 이유는 정규화된 벡터는 FAISS 파이쓰에서 검색 안정성에 도움 >> inner-product를 cosine-similarity 처럼 안전하게 사용하거나, 거리 계산 안정성(스케일 영향 제거)을 높이는 데 도움이됨
        cache_folder="/tmp/hf",                                         # Hugging Face 모델/토크나이저의 로컬 캐시 디렉터리 경로 지정 >> 만약 영구적인 저장이 필요하면(재부팅 후 보존) 다른 디렉토리 경로를 설정할 것
    )


def _get_tiktoken_encoding(name: str):          # _get_tiktoken_encoding 함수 (파라미터 name은 문자열 string 타입)
    if tiktoken is None:                        # 전역 변수 tiktoken이 None인지(즉 import tiktoken 실패해서 모듈을 사용할 수 없는 상태인지) 검사하는 조건문 >> True이면 아래 코드(예외발생)를 실행
        raise RuntimeError(                     # RuntimeError 예외를 발생시켜 현재 함수(또는 실행 흐름)를 즉시 중단하고 예외를 호출자 쪽으로 전달
            f"tiktoken을 불러오지 못했습니다. (원인: {repr(_tiktoken_import_err)})\n"   # 예외 메시지 f-string으로 _tiktoken_import_err를 repr()으로 문자열화해서 왜 import가 실패했는지 상세 정보를 표시함
            "requirements.txt에 'tiktoken'을 추가하고 재배포하세요."
        )
    try:                                            # 아래에 오는 동작(여기서는 tiktoken.get_encoding(name))을 시도(try)하되, 에러가 나면 except 블록으로 제어를 넘기겠다는 뜻이며 만약 에러가 없으면 try 블록 내부가 끝나고 함수는 정상 종료됨
        return tiktoken.get_encoding(name)          # tiktoken 모듈의 get_encoding 함수를 호출해 요청한 인코딩 이름(name)에 해당하는 인코딩 핸들(인코더/디코더 객체) 을 반환
                                                    # 이때 반환되는 객체는 보통 encode()이나 decode() 같은 메서드를 갖고 있어서 토큰화(문자열->토큰) 또는 역변환도 수행할 수 있음
    except Exception:                               # try 안의 코드 실행 중 어떤 예외사항이든 발생하면 여기로 들어옴 
        return tiktoken.get_encoding("cl100k_base") # 폴백(fallback) 동작으로 tiktoken.get_encoding("cl100k_base") 를 호출해 그 인코딩 핸들을 반환
                                                    # 즉 try블록의 get_encoding(name) 호출이 실패하면, 기본값으로 cl100k_base 인코딩을 사용하겠다는 의미 
                                                    # 폴백(fallback) == (의존성 없을 때/예외 발생할 때의) 대체행동 


@st.cache_resource(show_spinner=False)              # Streamlit 데코레이터. 이 함수가 반환하는 리소스(객체)를 앱 런타임 동안 캐시 >> show_spinner=False는 로딩 중에 Streamlit의 로딩 스피너를 보여주지 않음
def get_token_splitter(                             # get_token_splitter 함수는 토큰 기준 텍스트 스플리터(청크 생성기)를 반환
    chunk_tokens: int = TOKEN_CHUNK_SIZE,           # chunk_tokens 정의(정수) = 기본값은 전역 상수 TOKEN_CHUNK_SIZE (토큰 단위 한 청크 최대 크기)
    overlap_tokens: int = TOKEN_CHUNK_OVERLAP,      # overlap_tokens 정의(정수) = 기본값은 전역 상수 TOKEN_CHUNK_OVERLAP (인접 청크 간 중첩 토큰 수)
    encoding_name: str = TOKEN_ENCODING_NAME,       # encoding_name 정의(문자열) = 기본값은 TOKEN_ENCODING_NAME (어떤 tiktoken 인코딩을 쓸지 지정) >> (예: "cl100k_base")
):
    _ = _get_tiktoken_encoding(encoding_name)       # _get_tiktoken_encoding(encoding_name)를 호출해 해당 인코딩이 유효한지(그리고 tiktoken이 설치되어 있는지) 확인
                                                    # 만약 tiktoken이 없거나 인코딩 조회에 실패하면 _get_tiktoken_encoding이 예외를 던져 여기서 함수 실행이 중단됨
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(    # RecursiveCharacterTextSplitter의 클래스메서드 from_tiktoken_encoder를 호출하여 토큰 기준 텍스트 스플리터 인스턴스를 생성해 반환 >> 이 반환값이 st.cache_resource에 의해 캐시되어 재사용
        chunk_size=chunk_tokens,                                    # from_tiktoken_encoder 에 전달되는 인자: chunk_size는 토큰 단위로 한 청크의 최대 토큰 수(위 chunk_tokens)로 설정
        chunk_overlap=overlap_tokens,                               # chunk_overlap 은 인접 청크 간 중첩 토큰 수(위 overlap_tokens)로 설정
        encoding_name=encoding_name,                                # encoding_name은 tiktoken 인코딩 이름을 그대로 전달(예: "cl100k_base") >> 이 값으로 토큰화를 정확히 계산해 문자 기반 분할을 토큰 단위에 맞춰 수행
    )
  
 
# =========================
# 벡터스토어 빌드
# =========================
def build_vectorstore(doc_paths: List[Path]):       # build_vectorstore 함수 (인자 doc_paths는 pathlib.Path들의 리스트라는 타입힌트) >> 주어진 문서 경로 목록을 읽어 임베딩을 만들고 FAISS 파이쓰 벡터스토어(인메모리 인덱스)를 생성해 반환
    if FAISS is None:                               # 위에서 이전에 "try: from langchain_community.vectorstores import FAISS except: FAISS=None"으로 처리했으므로, FAISS 파이쓰가 import 실패로 None인지 확인
        raise RuntimeError(                         # 예외 발생 FAISS가 없으면 즉시 RuntimeError를 발생함
            f"😖 FAISS 모듈을 불러오지 못했습니다. (원인: {repr(_faiss_import_err)})\n"
            "CPU 환경에서는 requirements.txt에 'faiss-cpu'를 설정해 주세요.\n"
            "GPU 환경에서는 requirements.txt에 'faiss-gpu'를 설정해 주세요."
        )

    docs = []                                   # 빈 리스트 초기화: 이후 로더로 읽어들인 문서(문서 객체들)를 담을 컨테이너
    for p in doc_paths:                         # for문 시작 >> 전달된 각 문서 경로 p에 대해 처리
        loader = _load_document(p)              # 문서 로더 선택 >> _load_document(path) 함수로 파일 확장자에 맞는 로더(PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader 등)를 생성/반환 받음
        docs.extend(loader.load())              # 문서 읽기 및 누적 >> loader.load()를 호출해 그 파일에서 추출한 Document 객체들(보통 텍스트와 메타데이터 포함)을 반환받아 docs 리스트에 확장(extend)으로 추가
                                                # 결과로 docs는 모든 입력 파일에서 읽어온 원문 청크(또는 원문 페이지)들의 리스트가 됨

    splitter = get_token_splitter()             # 토큰 기반 스플리터 생성 >> get_token_splitter()는 위의 RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)로 생성된 스플리터(토큰단위로 chunk_size/overlap을 정확히 맞춤)를 반환
                                                # 토큰 단위 분할을 사용해 LLM 컨텍스트/임베딩 목적에 맞는 청크를 만들겠다는 의미 >> 토큰 단위 스플리터 사용
    splits = splitter.split_documents(docs)     # 문서 분할 실행: splitter.split_documents(docs)는 docs의 각 문서를 토큰 기준으로 잘라 여러 청크(문서 조각)를 생성함
                                                # 그 반환값 splits는 청크(각 청크도 Document 타입으로 텍스트 및 메타 포함)의 리스트로, 이걸 임베딩/색인에 입력하게 됨
    embeddings = get_hf_embeddings()            # 임베딩 모델 핸들 획득: get_hf_embeddings()는 캐시된 HuggingFaceEmbeddings 인스턴스(예: sentence-transformers/paraphrase-MiniLM-L6-v2)를 반환
                                                # 이 객체는 .embed_documents() 또는 LangChain 내부에서 자동으로 문서들을 벡터로 변환하는 데 사용됨
    vectorstore = FAISS.from_documents(splits, embeddings)  # FAISS 인덱스 생성: FAISS.from_documents(splits, embeddings)는 다음을 수행:
                                                            # 1) splits의 각 청크 텍스트에 대해 embeddings를 사용해 벡터(임베딩)를 계산
                                                            # 2) 계산된 벡터와 메타데이터(문서 출처, 텍스트 등)를 FAISS 파이쓰 인덱스에 삽입하여 벡터 검색 인스턴스를 생성
                                                            # 결과: vectorstore는 검색(유사도검색)을 지원하는 객체 (메모리 내 FAISS Index wrapper) >> 필요하면 이후 vectorstore.search(...)로 쿼리할 수 있음.
    return vectorstore                          # 완성된 FAISS 파이쓰 벡터스토어 객체를 호출자에게 반환 >> 호출자는 이 객체를 retriever로 감싸거나 ConversationalRetrievalChain에 넣어 RAG(문서기반응답) 체인을 제공

    
# ===============================
# 체인 구성 (LLM + Retriever + 메모리)
# ===============================
def get_chain(vectorstore, openai_api_key: str):  # get_chain 함수 선언 >>  이 get_chain 함수는 vectorstore와 openai_api_key 인자를  RAG용 체인(ConversationalRetrievalChain 등)을 구성(반환)
                                                  # 첫번째 인자 vectorstore : 호출자가 전달하는 LangChain 벡터스토어(검색인덱스나 retriever 역할을할 객체)를 받는 인자입 
                                                  # 두번째 인자 openai_api_key: str : 문자열 타입임을 명시한 타입힌트임(LLM 호출 시 사용할 OpenAI API 키)
    llm = ChatOpenAI(                             # ChatOpenAI라는 래퍼 클래스로부터 인스턴스를 생성하여 llm 변수에 할당(바인딩) 
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",  # 저렴,빠름 (필요 시 gpt-4o 로 교체 가능)
        temperature=0,        # 0: 창의성 낮추고, 일관성 높은 답변 생성
        max_retries=3,        # 간단한 재시도 (429 등 레이트리밋 대비)
        timeout=10,           # 10초 이상 걸리면 요청 중단
    )

    # 클래스(class) >> 설계도(타입) >> 예)class Dog: ... >> 규칙(속성/메서드)을 정의
    # 인스턴스(instance) >> 클래스(설계도)로부터 실제로 만든 물건 >> 예) my_dog = Dog() >> my_dog는 Dog의 인스턴스이자 파이썬 객체임
    # 객체(object) >> 파이썬의 모든 값(숫자, 문자열, 리스트, 함수, 클래스, 인스턴스 등)은 객체이다 >> 인스턴스는 객체의 하위 개념 (특정클래스의 객체)
    # 클래스도 객체다 >> Dog 자체도 파이썬 객체(타입객체)라서 type(Dog) 같은 조사가 가능함 >> “객체 vs 인스턴스” 구분할 때는 “인스턴스 = 클래스에서 생성된 객체”라는 정의가 유용함
    memory = ConversationBufferMemory(  # ConversationBufferMemory 클래스의 인스턴스를 생성하여 memory라는 변수에 할당 
                                        # >> memory 변수가 가리키는 ConversationBufferMemory 인스턴스는 대화형체인(Conversational chain)에서 이전 대화(히스토리)를 저장/관리하는 메모리 역할 담당
                                        # memory는 변수 이름이고, 그 변수는 ConversationBufferMemory 클래스의 인스턴스(실제객체)를 가리킨다 >> 그래서 말할 때 이 객체라고 하면 변수 memory가 가리키는 그 인스턴스를 뜻한다
        memory_key="chat_history",      # 체인 내부에서 대화 기록을 찾을 때 쓰는 키 이름
        return_messages=True,           # 대화 메시지 객체 리스트로 반환
        output_key="answer",            # 최종 생성된 답변의 키 이름
    )


    retriever = vectorstore.as_retriever(search_type="mmr")
    # vectorstore의 as_retriever(...) 메서드를 호출해서 검색기(retriever) 객체를 만들고 >> retriever 변수에 할당
    # vectorstore: 업로드된 문서 임베딩을 저장한 FAISS 파이쓰 벡터스토어
    # retriever: 질문을 임베딩하여 유사한 문서 청크 검색
    # search_type="mmr": "mmr"은 Maximal Marginal Relevance (최대 주변 관련성) >> 중복을 줄이고 다양성을 확보하는 검색
 
    chain = ConversationalRetrievalChain.from_llm(  # ConversationalRetrievalChain 클래스의 클래스메서드 from_llm(...)을 호출해 >> {대화형 RAG 체인(retriever + llm + memory)을 생성하고} >> 그 객체를 chain 변수에 할당
        llm=llm,                        # 체인이 답변을 생성할 때 사용할 LLM 인스턴스 (앞에서 만든 ChatOpenAI 객체)
        retriever=retriever,            # 문서 검색을 담당할 retriever 객체 (vectorstore.as_retriever(...)으로 만든것) 
        chain_type="stuff",             # 검색된 여러 청크(문서 조각)를 그대로(있는순서대로) 하나의 프롬프트에 집어넣어(stuff) LLM에게 전달하는 가장 단순한 처리 방식 >> 청크 수가 많거나 길면 토큰 한도 초과 위험있어 소량/짧은 청크에 적합 (대안: map_reduce, refine 등)
        memory=memory,                  # 앞에서 만든 ConversationBufferMemory 객체 >> 체인은 이 memory에서 과거 대화(히스토리)를 포함하고, LLM의 응답을 메모리에 자동추가(업데이트)
        get_chat_history=lambda h: h,   # get_chat_history는 체인이 내부적으로 메모리에서 꺼낸 히스토리 = lambda h: h로 “입력 그대로 반환”하게 설정되어 있으므로 >> 메모리의 대화 내역을 아무 변경 없이 프롬프트에 전달
        return_source_documents=True,   # 체인을 실행한 결과에 retriever가 반환한 원문 청크들(source documents)을 함께 포함해서 돌려주도록 True 설정 
        verbose=True,                   # 디버깅 로그
    )
    return chain                        # 생성된 chain(객체)을 get_chain(...) 함수를 호출한 곳으로 돌려주겠다는 의미
                                        # chain은 완성된 요리 / get_chain(...)함수는 요리사 / get_chain(...)함수 호출자는 손님 ㅎㅎ
                                        # return chain >> 요리사가 완성된 요리를 손님(함수를 호출한 코드)에 건네주는 동작
  

# =====================================================
# LLM 단독 (NOT RAG) 응답 헬퍼 >> RAG 구성 없을 경우 그냥 LLM을 사용
# =====================================================
def answer_without_rag(question: str, openai_api_key: str) -> str:      # answer_without_rag 함수 선언 >> (question:문자열 변수, openai_api_key:문자열 변수) >> 반환타입 힌트 -> str: 이 함수는 문자열을 반환할 것이라는 의미
    llm = ChatOpenAI(                                                   # ChatOpenAI(LangChain의 OpenAI 래퍼)를 사용해 llm 인스턴스 생성
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3,
        timeout=10,
    )
    sys = SystemMessage(content="너는 간결한 조수다. 모든 답변은 2~3문장 이내로 핵심만 요약해서 말해라.")    # 시스템 메시지(지침)생성: 모델에게 역할과 제약(톤/형식)을 지시하는 메시지 >> 프롬프트의 시스템 롤에 해당(role="system")
    user = HumanMessage(content=question)                                               # 사용자 메시지 객체 생성: 실제 사용자의 질의(question)를 HumanMessage로 래핑 >> 프롬프트의 user 부분에 해당
    resp = llm.invoke([sys, user])                                                      # llm.invoke(...)에 메시지 리스트 [sys, user]를 보내 모델에 질의 >> 반환값 resp는 모델 응답을 담은 응답 객체 
                                                                                        # llm = 인스턴스(객체) // invoke = llm 인스턴스가 가진 메서드(method)
                                                                                        # 위 invoke는 LangChain/버전에 따라 메서드명이 다를 수 있고, 반환 객체의 구조도 구현에 따라 다름으로 주의
    return getattr(resp, "content", str(resp))                                          # 반환처리: resp 객체에 content 속성이 있으면 그것을 반환하고(가장일반적), 만약 없으면 str(resp)로 변환한 값을 반환
                                                                                        # 그 이유는 resp 타입이 {"content": "..."} 형태인지, 아니면 단순 문자열인지 등 변동성을 안전하게 처리하려는 패턴입니다.

  
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
if "vectorstore" not in st.session_state:       # 세션 상태에 "vectorstore"라는 키가 없으면(처음 접속했거나 초기화되지 않았으면) 아래 블록을 실행
    st.session_state.vectorstore = None         # "vectorstore" 키를 세션 상태에 추가하고 값으로 None을 설정해서 초기화
if "chain" not in st.session_state:             # 세션 상태에 "chain" 키가 없으면 아래 블록을 실행
    st.session_state.chain = None               # "chain" 키를 추가하고 기본값을 None으로 설정해서 초기화
if "chat_history" not in st.session_state:      # 세션 상태에 "chat_history" 키가 없으면 아래 블록을 실행
    st.session_state.chat_history = []          # "chat_history" 키를 빈 리스트로 생성해서 초기화


# =========================
# 벡터 인덱스 생성
# =========================
if build_btn:                                       # build_btn(Streamlit 버튼의 반환값)이 True면 (사용자가 빌드버튼을 눌렀으면) 아래를 실행
    if not openai_api_key:                          # openai_api_key가 비어있으면(입력되지 않았으면) 아래 에러 메시지 출력
        st.error("🔑 OpenAI API Key를 입력하세요.")      # 에러 메시지 출력
    elif not uploaded_files:                        # API 키는 존재하지만 업로드된 파일(uploaded_files)이 없으면(빈 값이면) 아래를 실행  >>>>>>>>>>>>>> (고민) 없으면 그냥 RAG없이 LLM으로 응답 받도록 수정했음!!!!!!!!!!!!!!!!!!!!!!!!!
        st.warning("⚠️ 최소 1개 이상의 문서를 업로드하세요.")     # 경고 메시지 출력
    else:                                           # 위 두 조건(키없음/파일없음)을 모두 지나면 실제 벡터인덱스 생성작업을 수행하는 본 작업 수행
        with st.spinner("🏃🏻 Vector Index 생성 중… (최초에는 모델/토크나이저 로딩 시간이 걸릴 수 있습니다)"):   # Streamlit의 스피너(회전 표시) 컨텍스트 매니저를 실행
            try:                                                                # 오류가 날 수 있는 작업을 시도하기 위해 try 블록을 시작 >> 실패하면 except로 처리
                doc_paths = [_persist_upload(f) for f in uploaded_files]        # 리스트 컴프리헨션: 업로드된 각 파일 f에 대해 _persist_upload(f)를 호출(업로드객체를 임시파일로 저장)하고, 그 반환값(저장된Path)들을 doc_paths 리스트으로 생성/반환
                                                                                # 즉, 메모리 업로드 객체 >> 실제 임시파일 경로들로 변환하는 단계
                vs = build_vectorstore(doc_paths)                               # doc_paths(임시파일 경로 리스트)를 build_vectorstore()에 넘겨 벡터인덱스(예: FAISS인스턴스)를 생성
                                                                                # 이 단계에서 문서 파싱->스플리팅->임베딩 계산->벡터 인덱스 삽입 같은 무거운 작업이 실행됨 >> 시간소요
                st.session_state.vectorstore = vs                               # 생성된 벡터스토어 객체인 vs를 Streamlit 세션 상태(st.session_state)의 vectorstore 키에 저장
                st.session_state.chain = get_chain(vs, openai_api_key)          # get_chain(...)을 호출해 vs(벡터스토어)와 openai_api_key를 사용해 RAG 체인(ConversationalRetrievalChain 등)을 생성하고, 그 체인 객체를 세션상태의 chain 키에 저장
                                                                                # 이후 채팅 요청 시 이 chain을 바로 꺼내 쓸 수 있음                
                st.success("✅ Vector Index 생성 완료! (RAG: ON)")                 # 성공적으로 인덱스와 체인을 생성되었음을 Streamlit UI에 성공 메시지 출력
            except Exception as e:                                              # try 블록에서 어떤 예외가 발생하면 이 except 블록이 실행되며 >> 예외 객체를 e에 바인딩
                logger.exception("Vector Index 생성 실패")                         # logger.exception(...)는 예외 발생 시 메세지를 포함해 로그를 생성
                st.error(f"😖 Vector Index 생성 실패: {e}")                        # 사용자에게 에러 메시지 출력


# =========================
# 벡터 인덱스 삭제
# =========================
if delete_btn:
    # 먼저 Vector Index가 있는지 확인
    if st.session_state.get("vectorstore") is None and st.session_state.get("chain") is None:
        st.info("⛔ 삭제할 Vector Index가 없습니다.")
    else:
        with st.spinner("🏃🏻 Vector Index 삭제 중… "):   # Streamlit의 스피너(회전 표시) 컨텍스트 매니저를 실행
            try:
                # 1) 참조 제거 (세션 상태에서 삭제)
                vs = st.session_state.get("vectorstore")
                ch = st.session_state.get("chain")

                # 선택적) FAISS 파이쓰 내부 리소스 해제 시도: 안전하게 hasattr로 체크
                try:
                    if vs is not None and hasattr(vs, "index"):
                        # 일부 FAISS 파이쓰 래퍼는 index.reset() 또는 index = None으로 메모리 해제 가능
                        idx = getattr(vs, "index")
                        if hasattr(idx, "reset"):
                            idx.reset()
                except Exception:
                    # 실패해도 무시하고 계속 진행 (환경/버전에 따라 다름)
                    logger.debug("FAISS Vector Index Reset 시도 중 오류 발생(무시):", exc_info=True)

                # 2) 세션상태에서 제거
                st.session_state.vectorstore = None
                st.session_state.chain = None

                # 3) (선택적) 가비지 콜렉션으로 메모리 빨리 회수 시도
                import gc
                gc.collect()

                st.success("❎ Vector Index 및 RAG Chain 삭제 완료! (RAG: OFF)")
            except Exception as e:
                logger.exception("Vector Index 삭제 실패")
                st.error(f"😖 Vector Index 삭제 실패: {e}")

# =========================
# 질의 UI
# =========================
st.divider()                            # Streamlit UI 페이지에 가로 구분선(visual divider) 표시
st.subheader("💬 문서 기반 자연어 질의")       # Streamlit UI 페이지에 소제목(subheader)을 표시
user_q = st.text_input("질문 입력:", placeholder="예: 업로드한 문서 내용에서 질문을 해 보세요.")  # Streamlit UI 페이지에 텍스트 입력 상자 표시
ask = st.button("질문하기")                # Streamlit UI 페이지에 버튼을 화면에 생성하고, 클릭 여부를 불리언으로 반환 >> 라벨 "질문하기" 글자가 버튼에 표시


# ==============================================
# QA 실행: RAG-ON 경우 vs. RAG-OFF 경우 >> 폴백 로직 포함 🧠
# ==============================================
# 폴백(fallback) == (의존성 없을 때/예외 발생할 때의) 대체행동

if ask:                                                             # 이전에 만든 버튼(ask = st.button("질문하기"))이 사용자이벤트로 클릭되어 >> ask가 True일 때만 내부 블록을 실행
    if not openai_api_key:                                          # OpenAI API 키가 비어있거나 false(예:"",None)이면 아래 오류 발생
        st.error("🔑 OpenAI API Key를 입력하세요.")
    elif not user_q.strip():                                        # 사용자가 입력한 질문 user_q를 strip()으로 좌우 공백을 제거한 뒤 비어있는지 검사
        st.info("질문을 입력하세요.")
    else:                                                           # 위의 두 입력 검사를 모두 통과한 경우(키있고,질문입력됨) 실제 질의 처리 로직을 실행
        # 1) 인덱스/체인 준비 여부 확인
        if st.session_state.chain is None:                          # 세션상태에 저장된 chain이 None인지 확인 >> None이면 문서기반 체인(ConversationalRetrievalChain 등)이 아직 준비되지 않은 상태                              
            # 🔁 폴백(fallback): RAG-OFF 경로                                   # 문서 인덱스가 없으므로 LLM 단독 간단 답변 >> 인덱스/체인이 없을 때의 정책 설명 >> RAG 대신 LLM 단독 모드로 간단히 답변을 제공하는 폴백(fallback) 동작 수행
            with st.spinner("LLM 답변 생성 중… (RAG: OFF)"):                            # Streamlit의 스피너(로딩 표시)를 보여주는 컨텍스트 블록
                try:                                                                # try 블록 시작: 내부에서 발생할 수 있는 예외(에러)를 잡아서 사용자에게 친절히 알리기 위해 예외 처리 블록인 try를 사용했음
                    answer = answer_without_rag(user_q, openai_api_key)             # RAG 없이 LLM만으로 답을 생성하는 함수 answer_without_rag를 호출해서 >> user_q(질문)과 openai_api_key를 반환 >> 반환된 문자열(모델응답)을 answer에 저장
                    st.session_state.chat_history.append(("user", user_q))          # 세션상태에 저장된 chat_history 리스트에 사용자의 질문 항목을 추가 >> (role, message) 형식의 튜플로써 여기서는 ("user", user_q) 표현
                    st.session_state.chat_history.append(("assistant", answer))     # 세션상태에 저장된 chat_history 리스트에 사용자의 질문 항목을 추가 >> (role, message) 형식의 튜플로써 여기서는 ("assistant", answer) 표현

                    st.markdown("### 👽 답변  `RAG: OFF`")                                # Streamlit UI 페이지의 결과 영역에 마크다운으로 소제목을 출력
                    # st.write(answer)                                                  # st.write(answer)는 마크다운/리치 텍스트를 렌더링할 수 있고, 포맷 유지가고 글꼴/스타일 차이 발생 가능
                    st.text(answer)                                                     # st.text(answer)는 한글/영문 서식을 제거하고 순수 텍스트(포맷 없이)로 표시
                    st.info("👽 RAG 비활성화 상태입니다. Vector Index가 없기에 LLM으로 일반적인 답변을 제공합니다.")

                except Exception as e:                            # 예외처리: try 블록 내부에서 어떤 오류가 발생하면 여기로 분기되어 예외객체를 e에 바인딩
                    logger.exception("LLM-Only 질문 처리 실패")        # 로거에 예외와 스택 트레이스를 함께 기록
                    st.error(f"😖 질문 처리 실패(LLM-Only): {e}")      # 실패 사실과 예외 메시지(간단히 e의 문자열) 출력
        else:                                                     # 위의 if st.session_state.chain is None: 블록의 else 분기 >> 즉 체인(문서인덱스+RAG)이 준비되어 있을 때 실행되는 분기 >> RAG(문서기반 응답)경로
            # ✅ RAG-ON 경로
            with st.spinner("RAG 답변 생성 중… (RAG: ON)"):                             # Streamlit 스피너 메시지와 로딩 표시
                try:                                                                # 내부에서 일어날 수 있는 오류(예:검색실패, LLM호출실패)를 잡기 위해 try 블록을 사용
                    result = st.session_state.chain({"question": user_q})           # 세션상태에 저장된 chain(ConversationalRetrievalChain 등)을 호출(실행)하여 >> user_q 질문 처리
                                                                                    # 반환값 result는 일반적으로 딕셔너리 형태로, answer(LLM생성답변)와 source_documents(retriever가 가져온 근거문서 리스트)을 포함
                    answer = result.get("answer", "(답변 없음)")                       # result에서 "answer" 키를 꺼내고 >> 값이 없으면 기본값 "(답변 없음)"을 출력
                    sources = result.get("source_documents", [])                    # result에서 "source_documents" 키(근거문서 리스트)를 꺼내고 >> 키가 없으면 빈 리스트 []로 처리

                    st.session_state.chat_history.append(("user", user_q))          # 세션상태에 저장된 chat_history 리스트에 사용자의 질문 항목을 추가 >> (role, message) 형식의 튜플로써 여기서는 ("user", user_q) 표현 
                    st.session_state.chat_history.append(("assistant", answer))     # 세션상태에 저장된 chat_history 리스트에 사용자의 질문 항목을 추가 >> (role, message) 형식의 튜플로써 여기서는 ("assistant", answer) 표현

                    st.markdown("### 🧠 답변  `RAG: ON`")                                 # Streamlit UI 페이지의 결과 영역에 마크다운으로 소제목을 출력
                    # st.write(answer)                                                  # st.write(answer)는 마크다운/리치 텍스트를 렌더링할 수 있고, 포맷 유지가고 글꼴/스타일 차이 발생 가능
                    st.text(answer)                                                     # st.text(answer)는 한글/영문 서식을 제거하고 순수 텍스트(포맷 없이)로 표시
                  
                    if sources:                                                     # sources(근거문서 리스트)가 비어있지 않으면 >> 근거문서를 화면에 표시하는 로직
                        st.markdown("### 💡 참고 문서")                                # 소제목 표시: 근거 문서 영역 제목을 출력
                        with st.expander("참고 문서 위치 및 원문 일부 보기"):                 # st.expander 생성 >> 접이식(펼치기/접기)
                            for i, doc in enumerate(sources, 1):                    # sources의 각 문서(보통 Document 객체)를 순서대로 for 루프로 열거 >> i는 1부터 시작하는 인덱스 번호
                                src = doc.metadata.get("source", f"source_{i}")     # 문서 메타데이터에서 source 키(예: 파일경로, 문서명)를 가져옴 >> 해당 키가 없으면 기본값 "source_{i}"를 사용
                                st.markdown(f"**{i}.** {src}")                      # 문서 번호와 출처를 굵은 번호 형식으로 마크다운으로 출력 >> 예: 1. path/to/file.pdf
                                preview = (doc.page_content or "").strip()          # doc.page_content를 읽어서 >> 만약 doc.page_content가 None,(값없음) 또는 ""(빈문자열)이라면 >> 그 결과 문자열에 .strip()을 호출해 앞뒤공백(스페이스,탭,줄바꿈 등) 제거 >> 최종 문자열을 미리보기 preview에 저장
                                if len(preview) > 600:                              # 만약 preview가 600자 보다 길면 
                                    preview = preview[:600] + " …"                  # 앞 600자만 취하고 >> 말줄임표 " …"를 붙여서 표시 >> 미리보기 길이 제한(가독성확보)
                                st.code(preview)                                    # 코드블록(모노스페이스 폰트)으로 preview를 출력 >> (또는 st.text 대신 사용)
                    else:                                                           # sources가 비어있는 경우(근거문서 없음) 아래의 사용자에게 정보 박스 출력
                        st.info("해당 질문과 직접적으로 매칭되는 문서 청크를 찾지 못했습니다. (질문을 더 구체화하거나 인덱싱 범위를 늘려 보세요.)")
                except Exception as e:                                              # try 블록 내부에서 예외가 발생하면 여기로 이동하여 예외 객체를 e에 바인딩
                    logger.exception("질문 처리 실패(RAG)")                             # 예외와 스택 트레이스를 로깅
                    st.error(f"😖 질문 처리 실패(RAG): {e}")                             # 사용자에게 에러 박스로 실패 사실과 예외 메시지 출력
 
 
# =========================
# 대화 히스토리 표시
# =========================
if st.session_state.chat_history:           # st.session_state에 저장된 chat_history가 비어있지 않으면(truthy면) 아래 블록을 실행
    st.divider()                            # Streamlit UI 페이지에 가로 구분선(visual divider) 표시
    st.subheader("🗂️ 세션 아카이브")             # 소제목(subheader)을 표시
    for role, msg in st.session_state.chat_history[-10:]:   # chat_history의 마지막 10개 항목만 순서대로 for문 순회 >> [-10:]은 리스트 길이가 10 미만이면 전체를, 10 이상이면 끝에서 10개를 가져오는 슬라이스
                                                            # 각 항목은 (role, msg) 형태의 튜플이라고 가정 >> 예: ("user", "질문 내용"), ("assistant", "응답 내용")
        if role == "user":                                  # 현재 항목의 role이 "user"이면(사용자 발화이면) 아래를 수행
            st.markdown(f"**You:** {msg}")                  # **You:**는 굵게(강조) 표시되고, 뒤의 {msg}는 실제 메시지 출력 >> 마크다운 형식으로 출력  
                                                            # msg에 Markdown 문법이 포함되어 있으면 그대로 렌더링됨 (원치않으면 st.text(msg) 또는 st.write(escape) 방식 사용 가능)
        else:                                               # role이 "user"가 아니면 (보통 어시스턴트) 아래 블록을 실행
            st.markdown(f"**Assistant:** {msg}")            # **Assistant:**는 굵게(강조) 표시되고, 뒤의 {msg}는 실제 메시지 출력 >> 마크다운 형식으로 출력
