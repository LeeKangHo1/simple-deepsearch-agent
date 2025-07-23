# services/llm_service.py
"""
LLM 호출 서비스 (최적화 버전)

LangChain을 통해 다양한 LLM 제공자(OpenAI, Google)와 통신하는 서비스.
프롬프트 템플릿 관리, 응답 파싱, 오류 처리, 토큰 사용량 추적 등을 제공합니다.

주요 기능:
- OpenAI GPT 및 Google Gemini 모델 지원
- 에이전트별 최적화된 프롬프트 템플릿
- 구조화된 출력 파싱 (JSON, 리스트 등)
- 토큰 사용량 추적 및 비용 계산
- 재시도 로직 및 오류 처리
- 응답 품질 검증

최적화 내용:
- Dead Code 제거
- 불필요한 중복 계산 제거
- 프롬프트 템플릿 캐싱
- 토큰 계산 로직 개선
- 예외 처리 간소화
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time
import re
from functools import lru_cache

# LangChain imports
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, ListOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel

# 프로젝트 모듈들
from config.llm_config import get_cached_chat_model, ModelTemperatureConfig
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
    
    @property
    def estimated_cost_usd(self) -> float:
        """
        예상 비용 계산 (USD) - Property로 변경하여 캐싱
        
        Returns:
            float: 예상 비용 (USD)
        """
        if self.provider == "openai" and "gpt-4o-mini" in self.model_name.lower():
            # GPT-4o-mini 가격: $0.15/1M input, $0.60/1M output
            input_cost = (self.input_tokens / 1_000_000) * 0.15
            output_cost = (self.output_tokens / 1_000_000) * 0.60
            return input_cost + output_cost
        
        # 기본값 (정확한 가격 정보가 없는 경우)
        return (self.tokens_used / 1_000_000) * 0.50

class PromptTemplateCache:
    """프롬프트 템플릿 캐싱 클래스"""
    
    _templates = {}
    
    @classmethod
    def get_template(cls, template_name: str) -> ChatPromptTemplate:
        """캐시된 프롬프트 템플릿 반환"""
        if template_name not in cls._templates:
            cls._templates[template_name] = cls._create_template(template_name)
        return cls._templates[template_name]
    
    @classmethod
    def _create_template(cls, template_name: str) -> ChatPromptTemplate:
        """프롬프트 템플릿 생성"""
        templates = {
            "sub_queries": ChatPromptTemplate.from_messages([
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
            ]),
            
            "summarize": ChatPromptTemplate.from_messages([
                ("system", """당신은 문서 요약 전문가입니다. 주어진 문서를 {max_length}자 이내로 요약해주세요.

요약 규칙:
1. 문서의 핵심 내용과 주요 포인트만 포함
2. 객관적이고 정확한 정보만 추출
3. 개인적인 의견이나 추측은 제외
4. 명확하고 간결한 문장으로 작성
5. {max_length}자를 초과하지 말 것

출력 형식: 요약문만 반환 (부가 설명 없이)"""),
                ("user", "제목: {title}\n\n내용:\n{content}")
            ]),
            
            "insights": ChatPromptTemplate.from_messages([
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
            ]),
            
            "response": ChatPromptTemplate.from_messages([
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
            ]),
            
            "validate": ChatPromptTemplate.from_messages([
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
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        return templates[template_name]

class TokenEstimator:
    """토큰 사용량 추정 클래스"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        텍스트의 대략적인 토큰 수 추정
        실제 토큰화와는 차이가 있지만 빠른 추정용으로 사용
        
        Args:
            text: 토큰 수를 추정할 텍스트
            
        Returns:
            int: 추정된 토큰 수
        """
        # 영어: 단어 수 * 1.3, 한국어: 글자 수 * 1.5 (경험적 수치)
        english_chars = len(re.findall(r'[a-zA-Z\s]', text))
        korean_chars = len(text) - english_chars
        
        english_tokens = (english_chars / 4) * 1.3  # 영어 평균 단어 길이 4
        korean_tokens = korean_chars * 1.5
        
        return int(english_tokens + korean_tokens)

class LLMService:
    """
    LLM 호출을 관리하는 중앙 서비스 클래스 (최적화 버전)
    """
    
    def __init__(self):
        """LLM 서비스 초기화"""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0
        
        # 출력 파서들 초기화 (재사용을 위해 인스턴스 변수로)
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        self.list_parser = ListOutputParser()
        
        # 파서 매핑 (switch 역할)
        self.parser_map = {
            list: self.list_parser,
            dict: self.json_parser,
            str: self.str_parser
        }
    
    async def generate_sub_queries(self, user_question: str, num_queries: int = 4) -> LLMResponse:
        """사용자 질문에서 하위 검색 쿼리들을 생성"""
        return await self._call_llm(
            template_name="sub_queries",
            input_variables={"question": user_question, "num_queries": num_queries},
            agent_type="question_analyzer",
            expected_type=list
        )
    
    async def summarize_document(self, title: str, content: str, max_length: int = 300) -> LLMResponse:
        """문서 내용을 요약"""
        # 토큰 절약을 위한 내용 자르기 (더 정확한 계산)
        max_content_tokens = 1500  # 약 2000자 정도
        if TokenEstimator.estimate_tokens(content) > max_content_tokens:
            # 토큰 기준으로 자르기 (단순히 글자 수가 아닌)
            truncated_content = content[:int(max_content_tokens * 0.7)]  # 안전 마진
        else:
            truncated_content = content
        
        return await self._call_llm(
            template_name="summarize",
            input_variables={"title": title, "content": truncated_content, "max_length": max_length},
            agent_type="doc_summarizer",
            expected_type=str
        )
    
    async def generate_insights(self, summaries: List[str], user_question: str) -> LLMResponse:
        """문서 요약들로부터 인사이트 생성"""
        summaries_text = "\n\n".join([f"문서 {i+1}: {summary}" for i, summary in enumerate(summaries)])
        
        return await self._call_llm(
            template_name="insights",
            input_variables={"question": user_question, "summaries": summaries_text},
            agent_type="insight_generator",
            expected_type=list
        )
    
    async def generate_response(self, insights: List[str], user_question: str, sources: List[Dict[str, str]]) -> LLMResponse:
        """최종 마크다운 응답 생성"""
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        sources_text = "\n".join([f"- {source['title']}: {source['url']}" for source in sources])
        
        return await self._call_llm(
            template_name="response",
            input_variables={
                "question": user_question, 
                "insights": insights_text, 
                "sources": sources_text
            },
            agent_type="response_generator",
            expected_type=str
        )
    
    async def validate_response(self, response_content: str, user_question: str) -> LLMResponse:
        """응답 품질 검증"""
        return await self._call_llm(
            template_name="validate",
            input_variables={"question": user_question, "response": response_content},
            agent_type="validator",
            expected_type=dict
        )
    
    async def _call_llm(
        self, 
        template_name: str,
        input_variables: Dict[str, Any],
        agent_type: str,
        expected_type: type,
        max_retries: int = 3
    ) -> LLMResponse:
        """
        LLM 호출을 수행하는 내부 메서드 (최적화 버전)
        
        최적화 내용:
        - 프롬프트 템플릿 캐싱
        - 불필요한 변수 계산 제거
        - Dead Code 제거
        - 예외 처리 간소화
        """
        start_time = time.time()
        
        # 캐시된 템플릿과 모델 가져오기
        prompt_template = PromptTemplateCache.get_template(template_name)
        model = get_cached_chat_model(agent_type)
        output_parser = self.parser_map[expected_type]
        
        # 토큰 사용량 추정 (입력)
        prompt_text = str(input_variables)
        estimated_input_tokens = TokenEstimator.estimate_tokens(prompt_text)
        
        for attempt in range(max_retries):
            try:
                # 프롬프트 + 모델 + 파서로 체인 구성
                chain = prompt_template | model | output_parser
                
                # LLM 호출 실행
                result = await chain.ainvoke(input_variables)
                
                # 타입 검증 (마지막 시도가 아닌 경우만 재시도)
                if not isinstance(result, expected_type) and attempt < max_retries - 1:
                    logger.warning(f"Unexpected response type. Expected {expected_type}, got {type(result)}. Retrying...")
                    continue
                
                # 타입이 다르면 강제 변환 (마지막 시도)
                if not isinstance(result, expected_type):
                    result = self._convert_type(result, expected_type)
                
                # 성공 응답 생성
                response_time = time.time() - start_time
                estimated_output_tokens = TokenEstimator.estimate_tokens(str(result))
                
                response = LLMResponse(
                    content=json.dumps(result, ensure_ascii=False) if isinstance(result, (list, dict)) else str(result),
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    tokens_used=estimated_input_tokens + estimated_output_tokens,
                    model_name=settings.chat_model,
                    provider=settings.LLM_PROVIDER,
                    response_time=response_time,
                    temperature=ModelTemperatureConfig.get_temperature_for_agent(agent_type),
                    success=True
                )
                
                # 메트릭 업데이트
                self._update_metrics(response)
                return response
                
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # 마지막 시도에서 실패하면 실패 응답 반환
                if attempt == max_retries - 1:
                    return LLMResponse(
                        content=f"LLM 호출에 실패했습니다: {type(e).__name__}",
                        success=False,
                        error_message=str(e),
                        response_time=time.time() - start_time
                    )
    
    def _convert_type(self, value: Any, target_type: type) -> Any:
        """타입 변환 헬퍼 메서드"""
        try:
            if target_type == str:
                return str(value)
            elif target_type == list:
                if isinstance(value, str):
                    # JSON 파싱 시도
                    try:
                        parsed = json.loads(value)
                        return parsed if isinstance(parsed, list) else [str(value)]
                    except:
                        return [str(value)]
                return list(value) if hasattr(value, '__iter__') else [value]
            elif target_type == dict:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        return parsed if isinstance(parsed, dict) else {"content": str(value)}
                    except:
                        return {"content": str(value)}
                return dict(value) if hasattr(value, 'items') else {"content": str(value)}
            else:
                return value
        except:
            return str(value) if target_type == str else value
    
    def _update_metrics(self, response: LLMResponse):
        """메트릭 업데이트 (최적화)"""
        self.call_count += 1
        self.total_tokens_used += response.tokens_used
        self.total_cost += response.estimated_cost_usd
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계 반환"""
        return {
            "total_calls": self.call_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_tokens_per_call": round(self.total_tokens_used / max(self.call_count, 1), 2),
            "avg_cost_per_call": round(self.total_cost / max(self.call_count, 1), 4),
            "provider": settings.LLM_PROVIDER,
            "model": settings.chat_model
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0


# 전역 LLM 서비스 인스턴스 (싱글톤)
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """전역 LLM 서비스 인스턴스 반환"""
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