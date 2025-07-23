# config/llm_config.py
"""
LLM 제공자별 설정 및 클라이언트 생성
"""

from typing import Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from .settings import settings
import logging

logger = logging.getLogger(__name__)

class LLMConfig:
    """LLM 제공자별 설정 관리 클래스"""
    
    @staticmethod
    def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
        """
        현재 설정된 LLM 제공자의 채팅 모델 반환
        
        Args:
            temperature: 모델의 창의성 설정 (0.0 = 결정적, 1.0 = 창의적)
        
        Returns:
            BaseChatModel: 설정된 채팅 모델
        """
        if settings.LLM_PROVIDER == "gpt":
            return ChatOpenAI(
                model=settings.OPENAI_CHAT_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=temperature,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=2,
            )
        
        elif settings.LLM_PROVIDER == "gemini":
            return ChatGoogleGenerativeAI(
                model=settings.GOOGLE_CHAT_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=2,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    
    @staticmethod
    def get_embedding_model() -> Embeddings:
        """
        현재 설정된 LLM 제공자의 임베딩 모델 반환
        
        Returns:
            Embeddings: 설정된 임베딩 모델
        """
        if settings.LLM_PROVIDER == "gpt":
            return OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=2,
            )
        
        elif settings.LLM_PROVIDER == "gemini":
            return GoogleGenerativeAIEmbeddings(
                model=settings.GOOGLE_EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    
    @staticmethod
    def get_fast_chat_model(temperature: float = 0.1) -> BaseChatModel:
        """
        빠른 처리를 위한 경량 모델 반환 (간단한 태스크용)
        
        Args:
            temperature: 모델의 창의성 설정
            
        Returns:
            BaseChatModel: 경량 채팅 모델
        """
        if settings.LLM_PROVIDER == "gpt":
            # gpt-4o-mini가 이미 경량 모델이므로 동일한 모델 사용
            return ChatOpenAI(
                model=settings.OPENAI_CHAT_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=temperature,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=2,
            )
        
        elif settings.LLM_PROVIDER == "gemini":
            # gemini-2.5-flash가 이미 빠른 모델이므로 동일한 모델 사용
            return ChatGoogleGenerativeAI(
                model=settings.GOOGLE_CHAT_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=2,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    
    @staticmethod
    def get_model_info() -> dict:
        """
        현재 설정된 모델 정보 반환
        
        Returns:
            dict: 모델 정보 딕셔너리
        """
        return {
            "provider": settings.LLM_PROVIDER,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "api_key_configured": bool(settings.api_key),
        }


class ModelTemperatureConfig:
    """태스크별 최적 Temperature 설정"""
    
    # 정확성이 중요한 태스크 (낮은 창의성)
    ANALYTICAL = 0.0      # 질문 분석, 검증
    FACTUAL = 0.1         # 문서 요약, 정보 추출
    
    # 균형이 필요한 태스크 (중간 창의성)
    BALANCED = 0.3        # 인사이트 생성
    
    # 창의성이 필요한 태스크 (높은 창의성)
    CREATIVE = 0.7        # 응답 생성 (구조화, 표현력)
    
    @classmethod
    def get_temperature_for_agent(cls, agent_type: str) -> float:
        """
        에이전트 타입에 따른 최적 Temperature 반환
        
        Args:
            agent_type: 에이전트 타입 문자열
            
        Returns:
            float: 해당 에이전트에 최적화된 Temperature 값
        """
        temperature_map = {
            "question_analyzer": cls.ANALYTICAL,
            "doc_summarizer": cls.FACTUAL,
            "insight_generator": cls.BALANCED,
            "response_generator": cls.CREATIVE,
            "validator": cls.ANALYTICAL,
        }
        
        return temperature_map.get(agent_type, cls.BALANCED)


# 전역 LLM 인스턴스들 (재사용을 위한 캐싱)
_chat_model_cache = {}
_embedding_model_cache = None

def get_cached_chat_model(agent_type: str = "default") -> BaseChatModel:
    """
    캐시된 채팅 모델 반환 (성능 최적화)
    
    Args:
        agent_type: 에이전트 타입 (Temperature 최적화용)
        
    Returns:
        BaseChatModel: 캐시된 채팅 모델
    """
    global _chat_model_cache
    
    temperature = ModelTemperatureConfig.get_temperature_for_agent(agent_type)
    cache_key = f"{agent_type}_{temperature}"
    
    if cache_key not in _chat_model_cache:
        _chat_model_cache[cache_key] = LLMConfig.get_chat_model(temperature)
        logger.info(f"Created new chat model for {agent_type} with temperature {temperature}")
    
    return _chat_model_cache[cache_key]

def get_cached_embedding_model() -> Embeddings:
    """
    캐시된 임베딩 모델 반환 (성능 최적화)
    
    Returns:
        Embeddings: 캐시된 임베딩 모델
    """
    global _embedding_model_cache
    
    if _embedding_model_cache is None:
        _embedding_model_cache = LLMConfig.get_embedding_model()
        logger.info("Created new embedding model")
    
    return _embedding_model_cache

def clear_model_cache():
    """모델 캐시 초기화 (메모리 정리용)"""
    global _chat_model_cache, _embedding_model_cache
    _chat_model_cache.clear()
    _embedding_model_cache = None
    logger.info("Model cache cleared")

# 초기화 시 모델 정보 로깅
logger.info(f"LLM Configuration: {LLMConfig.get_model_info()}")