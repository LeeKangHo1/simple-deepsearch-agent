# config/settings.py
"""
환경 변수 로드 및 애플리케이션 설정 관리
"""

import os
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv
import logging

# .env 파일 로드
load_dotenv()

class Settings:
    """애플리케이션 설정 클래스"""
    
    def __init__(self):
        self._validate_required_env()
    
    # ================================
    # LLM 설정
    # ================================
    LLM_PROVIDER: Literal["gpt", "gemini"] = os.getenv("LLM_PROVIDER", "gpt")
    
    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL_MINI", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Google Gemini 설정
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CHAT_MODEL: str = os.getenv("GOOGLE_CHAT_MODEL_MINI", "gemini-2.5-flash")
    GOOGLE_EMBEDDING_MODEL: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    
    # ================================
    # 검색 API 설정
    # ================================
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    DUCKDUCKGO_MAX_RESULTS: int = int(os.getenv("DUCKDUCKGO_MAX_RESULTS", "10"))
    
    # ================================
    # 벡터 데이터베이스 설정
    # ================================
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "research_docs")
    
    # ================================
    # LangSmith 모니터링 설정
    # ================================
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "simple-deepresearch-agent")
    
    # ================================
    # 애플리케이션 설정
    # ================================
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
    MAX_DOCUMENTS_TO_PROCESS: int = int(os.getenv("MAX_DOCUMENTS_TO_PROCESS", "15"))
    MAX_SUMMARY_LENGTH: int = int(os.getenv("MAX_SUMMARY_LENGTH", "300"))
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "2000"))
    MAX_VALIDATION_RETRIES: int = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))
    
    # ================================
    # 기타 설정
    # ================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # ================================
    # Streamlit 설정
    # ================================
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    def _validate_required_env(self):
        """필수 환경 변수 검증"""
        errors = []
        
        # LLM 제공자에 따른 API 키 검증
        if self.LLM_PROVIDER == "gpt":
            if not self.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=gpt")
        elif self.LLM_PROVIDER == "gemini":
            if not self.GOOGLE_API_KEY:
                errors.append("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        else:
            errors.append(f"Invalid LLM_PROVIDER: {self.LLM_PROVIDER}. Must be 'gpt' or 'gemini'")
        
        # Tavily API 키 검증 (선택사항이지만 권장)
        if not self.TAVILY_API_KEY:
            logging.warning("TAVILY_API_KEY not found. Only DuckDuckGo search will be available.")
        
        if errors:
            raise ValueError(f"Environment validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    @property
    def chat_model(self) -> str:
        """현재 LLM 제공자의 채팅 모델 반환"""
        if self.LLM_PROVIDER == "gpt":
            return self.OPENAI_CHAT_MODEL
        else:
            return self.GOOGLE_CHAT_MODEL
    
    @property
    def embedding_model(self) -> str:
        """현재 LLM 제공자의 임베딩 모델 반환"""
        if self.LLM_PROVIDER == "gpt":
            return self.OPENAI_EMBEDDING_MODEL
        else:
            return self.GOOGLE_EMBEDDING_MODEL
    
    @property
    def api_key(self) -> str:
        """현재 LLM 제공자의 API 키 반환"""
        if self.LLM_PROVIDER == "gpt":
            return self.OPENAI_API_KEY
        else:
            return self.GOOGLE_API_KEY
    
    def ensure_data_directories(self):
        """필요한 데이터 디렉토리들 생성"""
        directories = [
            self.CHROMA_PERSIST_DIRECTORY,
            "./data/temp",
            "./logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_info(self) -> dict:
        """설정 정보 요약 반환 (민감한 정보 제외)"""
        return {
            "llm_provider": self.LLM_PROVIDER,
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "max_search_results": self.MAX_SEARCH_RESULTS,
            "max_documents_to_process": self.MAX_DOCUMENTS_TO_PROCESS,
            "langsmith_enabled": self.LANGCHAIN_TRACING_V2,
            "tavily_available": bool(self.TAVILY_API_KEY),
        }


# 전역 설정 인스턴스
settings = Settings()

# LangSmith 환경 변수 설정 (LangChain에서 자동으로 인식)
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

# 필요한 디렉토리 생성
settings.ensure_data_directories()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Settings loaded: {settings.get_info()}")