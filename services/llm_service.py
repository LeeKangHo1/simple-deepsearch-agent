# services/llm_service.py
"""
LLM 호출 서비스

LangChain을 통해 다양한 LLM 제공자(OpenAI, Google)와 통신하는 서비스.
프롬프트 템플릿 관리, 응답 파싱, 오류 처리, 토큰 사용량 추적 등을 제공합니다.

주요 기능:
- OpenAI GPT 및 Google Gemini 모델 지원
- 에이전트별 최적화된 프롬프트 템플릿
- 구조화된 출력 파싱 (JSON, 리스트 등)
- 토큰 사용량 추적 및 비용 계산
- 재시도 로직 및 오류 처리
- 응답 품질 검증
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time
import re

# LangChain imports
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, ListOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel

# 프로젝트 모듈들
from config.llm_config import get_cached_chat_model, get_cached_embedding_model, ModelTemperatureConfig
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """
    LLM 응답 정보를 담는 데이터 클래스
    
    LLM 호출 결과와 메타데이터를 함께 저장하여
    품질 평가, 비용 추적, 디버깅 등에 활용합니다.
    """
    content: str
    """LLM이 생성한 실제 응답 내용"""
    
    tokens_used: int = 0
    """사용된 토큰 수 (입력 + 출력)"""
    
    input_tokens: int = 0
    """입력 토큰 수"""
    
    output_tokens: int = 0
    """출력 토큰 수"""
    
    model_name: str = ""
    """사용된 모델명"""
    
    provider: str = ""
    """LLM 제공자 (openai, google)"""
    
    response_time: float = 0.0
    """응답 시간 (초)"""
    
    temperature: float = 0.0
    """사용된 Temperature 설정"""
    
    success: bool = True
    """호출 성공 여부"""
    
    error_message: str = ""
    """오류 발생 시 오류 메시지"""
    
    def get_estimated_cost(self) -> float:
        """
        예상 비용 계산 (USD)
        
        현재 설정된 모델과 토큰 사용량을 기반으로
        예상 비용을 계산합니다.
        
        Returns:
            float: 예상 비용 (USD)
        """
        if self.provider == "openai":
            if "gpt-4o-mini" in self.model_name.lower():
                # GPT-4o-mini 가격: $0.15/1M input, $0.60/1M output
                input_cost = (self.input_tokens / 1_000_000) * 0.15
                output_cost = (self.output_tokens / 1_000_000) * 0.60
                return input_cost + output_cost
        
        # 기본값 (정확한 가격 정보가 없는 경우)
        return (self.tokens_used / 1_000_000) * 0.50

class LLMService:
    """
    LLM 호출을 관리하는 중앙 서비스 클래스
    
    다양한 에이전트에서 필요한 LLM 호출을 표준화된 방식으로 처리.
    프롬프트 템플릿 관리, 응답 파싱, 품질 검증 등을 담당합니다.
    
    사용 예시:
        llm_service = LLMService()
        response = await llm_service.generate_sub_queries(
            "오픈소스 LLM 트렌드 알려줘"
        )
    """
    
    def __init__(self):
        """LLM 서비스 초기화"""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0
        
        # 출력 파서들 초기화
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        self.list_parser = ListOutputParser()
    
    async def generate_sub_queries(self, user_question: str, num_queries: int = 4) -> LLMResponse:
        """
        사용자 질문에서 하위 검색 쿼리들을 생성
        
        질문 분석 에이전트에서 사용하는 메서드.
        사용자의 복잡한 질문을 여러 개의 구체적인 검색 쿼리로 분해합니다.
        
        Args:
            user_question: 사용자의 원본 질문
            num_queries: 생성할 하위 쿼리 개수 (기본: 4개)
            
        Returns:
            LLMResponse: 생성된 하위 쿼리 리스트가 포함된 응답
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 검색 쿼리 전문가입니다. 사용자의 질문을 분석하여 효과적인 검색 쿼리들을 생성해주세요.

규칙:
1. 원본 질문을 {num_queries}개의 구체적인 검색 쿼리로 분해
2. 각 쿼리는 서로 다른 관점이나 측면을 다룰 것
3. 검색 엔진에서 좋은 결과를 얻을 수 있는 키워드 중심으로 작성
4. 너무 길거나 복잡하지 않게 간결하게 작성
5. JSON 배열 형태로 반환: ["쿼리1", "쿼리2", ...]

예시:
질문: "인공지능 발전이 일자리에 미치는 영향은?"
출력: ["인공지능 자동화 일자리 대체", "AI 기술 새로운 직업 창출", "인공지능 고용시장 변화 전망", "로봇 자동화 실업률 통계"]"""),
            ("user", "질문: {question}")
        ])
        
        return await self._call_llm(
            prompt_template=prompt_template,
            input_variables={"question": user_question, "num_queries": num_queries},
            agent_type="question_analyzer",
            output_parser=self.list_parser,
            expected_type=list
        )
    
    async def summarize_document(self, title: str, content: str, max_length: int = 300) -> LLMResponse:
        """
        문서 내용을 요약
        
        문서 요약 에이전트에서 사용하는 메서드.
        긴 문서 내용을 핵심 정보만 추출하여 간결하게 요약합니다.
        
        Args:
            title: 문서 제목
            content: 문서 전체 내용
            max_length: 최대 요약 길이 (기본: 300자)
            
        Returns:
            LLMResponse: 요약된 텍스트가 포함된 응답
        """
        # 너무 긴 내용은 앞부분만 사용 (토큰 절약)
        truncated_content = content[:2000] if len(content) > 2000 else content
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 문서 요약 전문가입니다. 주어진 문서를 {max_length}자 이내로 요약해주세요.

요약 규칙:
1. 문서의 핵심 내용과 주요 포인트만 포함
2. 객관적이고 정확한 정보만 추출
3. 개인적인 의견이나 추측은 제외
4. 명확하고 간결한 문장으로 작성
5. {max_length}자를 초과하지 말 것

출력 형식: 요약문만 반환 (부가 설명 없이)"""),
            ("user", "제목: {title}\n\n내용:\n{content}")
        ])
        
        return await self._call_llm(
            prompt_template=prompt_template,
            input_variables={"title": title, "content": truncated_content},
            agent_type="doc_summarizer",
            output_parser=self.str_parser,
            expected_type=str
        )
    
    async def generate_insights(self, summaries: List[str], user_question: str) -> LLMResponse:
        """
        문서 요약들로부터 인사이트 생성
        
        인사이트 생성 에이전트에서 사용하는 메서드.
        여러 문서 요약을 종합 분석하여 깊이 있는 통찰과 시사점을 도출합니다.
        
        Args:
            summaries: 문서 요약 리스트
            user_question: 사용자의 원본 질문
            
        Returns:
            LLMResponse: 생성된 인사이트 리스트가 포함된 응답
        """
        summaries_text = "\n\n".join([f"문서 {i+1}: {summary}" for i, summary in enumerate(summaries)])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 정보 분석 전문가입니다. 여러 문서 요약을 종합하여 깊이 있는 인사이트를 도출해주세요.

인사이트 생성 규칙:
1. 단순한 정보 나열이 아닌 의미 있는 통찰 제공
2. 여러 문서에서 나타나는 공통 패턴이나 트렌드 파악
3. 원인과 결과, 상관관계 등을 분석
4. 미래 전망이나 시사점 포함
5. 3-5개의 핵심 인사이트를 JSON 배열로 반환

출력 형식:
["인사이트1: 구체적인 통찰 내용", "인사이트2: 패턴이나 트렌드 분석", ...]

예시:
["기업들이 오픈소스 LLM을 채택하는 이유는 비용 절감보다 유연성과 커스터마이징 가능성 때문이다", "상업용 모델과 오픈소스 모델의 성능 격차가 빠르게 줄어들면서 시장 판도가 변화하고 있다"]"""),
            ("user", "원본 질문: {question}\n\n문서 요약들:\n{summaries}")
        ])
        
        return await self._call_llm(
            prompt_template=prompt_template,
            input_variables={"question": user_question, "summaries": summaries_text},
            agent_type="insight_generator",
            output_parser=self.list_parser,
            expected_type=list
        )
    
    async def generate_response(self, insights: List[str], user_question: str, sources: List[Dict[str, str]]) -> LLMResponse:
        """
        최종 마크다운 응답 생성
        
        응답 생성 에이전트에서 사용하는 메서드.
        인사이트와 출처 정보를 바탕으로 구조화된 마크다운 응답을 생성합니다.
        
        Args:
            insights: 인사이트 리스트
            user_question: 사용자의 원본 질문
            sources: 출처 정보 리스트 [{"title": "제목", "url": "URL"}]
            
        Returns:
            LLMResponse: 마크다운 형식의 최종 응답
        """
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        sources_text = "\n".join([f"- {source['title']}: {source['url']}" for source in sources])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 리서치 보고서 작성 전문가입니다. 인사이트와 출처를 바탕으로 마크다운 형식의 응답을 작성해주세요.

작성 규칙:
1. 마크다운 형식으로 구조화 (제목, 소제목, 불릿 포인트 등)
2. 사용자 질문에 직접적으로 답변
3. 각 핵심 내용 옆에 출처 표시: *(출처: domain.com)*
4. 논리적이고 읽기 쉬운 구조로 구성
5. 최대 {settings.MAX_RESPONSE_LENGTH}자 이내
6. 객관적이고 정확한 정보만 포함

구조 예시:
# [질문 관련 제목]

## 주요 현황
- 핵심 내용 1 *(출처: example.com)*
- 핵심 내용 2 *(출처: news.com)*

## 주요 인사이트
- 인사이트 1
- 인사이트 2

## 향후 전망
- 예상되는 변화나 트렌드

---
출처: 참고한 주요 자료들"""),
            ("user", "질문: {question}\n\n주요 인사이트:\n{insights}\n\n참고 출처:\n{sources}")
        ])
        
        return await self._call_llm(
            prompt_template=prompt_template,
            input_variables={
                "question": user_question, 
                "insights": insights_text, 
                "sources": sources_text
            },
            agent_type="response_generator",
            output_parser=self.str_parser,
            expected_type=str
        )
    
    async def validate_response(self, response_content: str, user_question: str) -> LLMResponse:
        """
        응답 품질 검증
        
        검증 에이전트에서 사용하는 메서드.
        생성된 응답의 품질, 논리성, 완성도를 검증합니다.
        
        Args:
            response_content: 검증할 응답 내용
            user_question: 사용자의 원본 질문
            
        Returns:
            LLMResponse: 검증 결과가 포함된 응답
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 응답 품질 검증 전문가입니다. 주어진 응답을 검토하여 품질을 평가해주세요.

검증 기준:
1. 사용자 질문에 적절히 답변했는가?
2. 논리적 일관성이 있는가?
3. 출처 표시가 적절히 되어 있는가?
4. 마크다운 형식이 올바른가?
5. 내용이 객관적이고 정확한가?

출력 형식 (JSON):
{
    "is_valid": true/false,
    "feedback": "구체적인 피드백 메시지",
    "issues": ["문제점1", "문제점2", ...],
    "suggestions": ["개선사항1", "개선사항2", ...],
    "confidence_score": 0.0-1.0
}"""),
            ("user", "원본 질문: {question}\n\n검증할 응답:\n{response}")
        ])
        
        return await self._call_llm(
            prompt_template=prompt_template,
            input_variables={"question": user_question, "response": response_content},
            agent_type="validator",
            output_parser=self.json_parser,
            expected_type=dict
        )
    
    async def _call_llm(
        self, 
        prompt_template: ChatPromptTemplate,
        input_variables: Dict[str, Any],
        agent_type: str,
        output_parser: Any,
        expected_type: type,
        max_retries: int = 3
    ) -> LLMResponse:
        """
        LLM 호출을 수행하는 내부 메서드
        
        프롬프트 실행, 응답 파싱, 오류 처리, 메트릭 수집을 담당하는
        핵심 메서드입니다. 모든 LLM 호출은 이 메서드를 통해 수행됩니다.
        
        Args:
            prompt_template: 실행할 프롬프트 템플릿
            input_variables: 프롬프트에 전달할 변수들
            agent_type: 에이전트 타입 (Temperature 설정용)
            output_parser: 응답 파싱을 위한 파서
            expected_type: 예상되는 응답 타입
            max_retries: 최대 재시도 횟수
            
        Returns:
            LLMResponse: LLM 응답 및 메타데이터
        """
        start_time = time.time()
        
        # 에이전트 타입에 맞는 모델 가져오기 (Temperature 이미 적용됨)
        model = get_cached_chat_model(agent_type)
        
        for attempt in range(max_retries):
            try:
                # 프롬프트 + 모델 + 파서로 체인 구성
                chain = prompt_template | model | output_parser
                
                # LLM 호출 실행
                result = await chain.ainvoke(input_variables)
                
                # 응답 타입 검증
                if not isinstance(result, expected_type):
                    if attempt < max_retries - 1:
                        logger.warning(f"Unexpected response type. Expected {expected_type}, got {type(result)}. Retrying...")
                        continue
                    else:
                        # 마지막 시도에서도 실패하면 문자열로 변환
                        result = str(result) if expected_type == str else result
                
                # 성공 응답 생성
                response_time = time.time() - start_time
                self._update_metrics(response_time)
                
                return LLMResponse(
                    content=json.dumps(result) if isinstance(result, (list, dict)) else str(result),
                    model_name=settings.chat_model,
                    provider=settings.LLM_PROVIDER,
                    response_time=response_time,
                    temperature=ModelTemperatureConfig.get_temperature_for_agent(agent_type),  # 여기서만 조회
                    success=True
                )
                
            except OutputParserException as e:
                logger.warning(f"Output parsing failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # 파싱 실패 시 원본 텍스트 반환
                    return LLMResponse(
                        content="응답 파싱에 실패했습니다.",
                        success=False,
                        error_message=str(e),
                        response_time=time.time() - start_time
                    )
            
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return LLMResponse(
                        content="LLM 호출에 실패했습니다.",
                        success=False,
                        error_message=str(e),
                        response_time=time.time() - start_time
                    )
        
        # 모든 시도 실패
        return LLMResponse(
            content="최대 재시도 횟수를 초과했습니다.",
            success=False,
            error_message="Max retries exceeded",
            response_time=time.time() - start_time
        )
    
    def _update_metrics(self, response_time: float):
        """메트릭 업데이트"""
        self.call_count += 1
        # 향후 토큰 사용량 추적 로직 추가 예정
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        사용량 통계 반환
        
        Returns:
            Dict[str, Any]: 사용량 통계 정보
        """
        return {
            "total_calls": self.call_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost,
            "provider": settings.LLM_PROVIDER,
            "model": settings.chat_model
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0


# 전역 LLM 서비스 인스턴스
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """
    전역 LLM 서비스 인스턴스 반환
    
    Returns:
        LLMService: LLM 서비스 인스턴스
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
        logger.info("LLM service instance created")
    
    return _llm_service_instance

def reset_llm_service():
    """LLM 서비스 인스턴스 리셋 (테스트용)"""
    global _llm_service_instance
    _llm_service_instance = None
    logger.info("LLM service instance reset")