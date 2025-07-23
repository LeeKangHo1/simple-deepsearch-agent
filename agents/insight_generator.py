# agents/insight_generator.py
"""
인사이트 생성 에이전트

문서 요약들을 종합 분석하여 깊이 있는 통찰과 시사점을 도출합니다.
단순한 정보 나열이 아닌 의미 있는 패턴, 트렌드, 상관관계를 파악하여
사용자에게 가치 있는 인사이트를 제공합니다.

주요 기능:
- 다중 문서 요약 종합 분석
- 패턴 및 트렌드 도출
- 원인-결과 관계 분석
- 미래 전망 및 시사점 생성
- 신뢰도 기반 가중치 처리
"""

import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 모듈들
from models.data_models import Insight, DocumentSummary
from models.state import ResearchState, StateManager
from services.llm_service import get_llm_service, LLMResponse
from utils.text_processing import clean_text, is_valid_text, extract_keywords
from utils.validators import validate_insights
from utils.logger import create_agent_logger, log_execution_time

logger = logging.getLogger(__name__)
agent_logger = create_agent_logger("insight_generator")

class InsightGenerator:
    """
    인사이트 생성 에이전트 클래스
    
    여러 문서 요약을 종합적으로 분석하여 의미 있는 통찰을 도출합니다.
    신뢰도 점수를 고려한 가중치 처리와 구분자 기반 파싱을 지원합니다.
    
    처리 흐름:
    1. 문서 요약 전처리 및 신뢰도 기반 가중치 계산
    2. LLM을 통한 인사이트 생성
    3. JSON/구분자 기반 파싱
    4. Insight 객체 생성 및 카테고리 분류
    5. 품질 검증 및 후처리
    """
    
    def __init__(self, min_insight_length: int = 30, target_insight_count: int = 4):
        """
        인사이트 생성기 초기화
        
        Args:
            min_insight_length: 인사이트 최소 길이 (기본: 30자)
            target_insight_count: 목표 인사이트 개수 (기본: 4개)
        """
        self.min_insight_length = min_insight_length
        self.target_insight_count = target_insight_count
        self.llm_service = get_llm_service()
        
        # 통계 추적
        self.total_summaries_processed = 0
        self.total_insights_generated = 0
        self.json_parse_failures = 0
        self.separator_parse_count = 0
    
    @log_execution_time
    async def generate_insights(
        self, 
        summaries: List[DocumentSummary], 
        user_question: str = ""
    ) -> List[Insight]:
        """
        문서 요약들로부터 인사이트 생성
        
        Args:
            summaries: DocumentSummary 객체 리스트
            user_question: 사용자의 원본 질문
            
        Returns:
            List[Insight]: 생성된 인사이트 리스트
        """
        if not summaries:
            logger.warning("인사이트 생성할 요약이 없습니다.")
            return []
        
        agent_logger.start_step(f"인사이트 생성 시작 ({len(summaries)}개 요약)")
        
        try:
            # 1단계: 요약 전처리 및 가중치 계산
            processed_summaries = self._preprocess_summaries(summaries)
            
            if not processed_summaries:
                agent_logger.end_step("요약 전처리", False, "유효한 요약이 없음")
                return []
            
            # 2단계: LLM을 통한 인사이트 생성
            llm_response = await self._generate_insights_llm(processed_summaries, user_question)
            
            if not llm_response.success:
                agent_logger.end_step("인사이트 생성", False, f"LLM 호출 실패: {llm_response.error_message}")
                return []
            
            # 3단계: 응답 파싱 및 Insight 객체 생성
            insights = self._parse_insights_response(llm_response.content, summaries, user_question)
            
            # 4단계: 통계 업데이트
            self.total_summaries_processed += len(summaries)
            self.total_insights_generated += len(insights)
            
            agent_logger.end_step(
                "인사이트 생성 완료",
                True,
                f"{len(insights)}개 인사이트 생성"
            )
            
            return insights
            
        except Exception as e:
            agent_logger.end_step("인사이트 생성", False, f"오류 발생: {str(e)}")
            logger.error(f"인사이트 생성 중 오류 발생: {e}")
            return []
    
    def _preprocess_summaries(self, summaries: List[DocumentSummary]) -> List[Dict[str, Any]]:
        """
        요약 전처리 및 신뢰도 기반 가중치 계산
        
        Args:
            summaries: DocumentSummary 객체 리스트
            
        Returns:
            List[Dict[str, Any]]: 전처리된 요약 정보 리스트
        """
        processed = []
        
        for i, summary in enumerate(summaries):
            try:
                # 요약 유효성 검증
                if not summary.summary or len(summary.summary.strip()) < 10:
                    logger.debug(f"요약 {i+1} 제외: 내용이 너무 짧음")
                    continue
                
                # 신뢰도 점수 기반 가중치 계산
                confidence = max(0.1, summary.confidence_score)  # 최소 0.1 보장
                weight = confidence  # 신뢰도가 곧 가중치
                
                processed_summary = {
                    "index": i + 1,
                    "content": summary.summary.strip(),
                    "confidence_score": confidence,
                    "weight": weight,
                    "document_hash": summary.document_hash,
                    "key_points": summary.key_points,
                    "word_count": summary.word_count
                }
                
                processed.append(processed_summary)
                
            except Exception as e:
                logger.warning(f"요약 {i+1} 전처리 실패: {e}")
                continue
        
        # 가중치 기준으로 정렬 (높은 신뢰도 우선)
        processed.sort(key=lambda x: x["weight"], reverse=True)
        
        logger.info(f"요약 전처리 완료: {len(processed)}/{len(summaries)}개 유효")
        return processed
    
    async def _generate_insights_llm(
        self, 
        processed_summaries: List[Dict[str, Any]], 
        user_question: str
    ) -> LLMResponse:
        """
        LLM을 통한 인사이트 생성
        
        Args:
            processed_summaries: 전처리된 요약 리스트
            user_question: 사용자 질문
            
        Returns:
            LLMResponse: LLM 응답
        """
        # 신뢰도 기반 가중치를 고려한 요약 텍스트 구성
        weighted_summaries = []
        for summary in processed_summaries:
            weight_indicator = "⭐" * min(5, int(summary["weight"] * 5))  # 신뢰도 시각화
            weighted_text = f"문서 {summary['index']} {weight_indicator} (신뢰도: {summary['confidence_score']:.2f}): {summary['content']}"
            weighted_summaries.append(weighted_text)
        
        summaries_text = "\n\n".join(weighted_summaries)
        
        # 사용자 질문 컨텍스트 강화
        question_context = ""
        if user_question.strip():
            question_context = f"\n\n**중요**: 특히 '{user_question}'와 관련된 통찰에 우선순위를 두세요."
        
        # 인사이트 생성 프롬프트
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 정보 분석 전문가입니다. 여러 문서 요약을 종합하여 깊이 있는 인사이트를 도출해주세요.

인사이트 생성 규칙:
1. 단순한 정보 나열이 아닌 의미 있는 통찰 제공
2. 여러 문서에서 나타나는 공통 패턴이나 트렌드 파악
3. 원인과 결과, 상관관계 등을 분석
4. 미래 전망이나 시사점 포함
5. {self.target_insight_count}개의 핵심 인사이트를 생성
6. 각 인사이트는 최소 {self.min_insight_length}자 이상
7. 신뢰도 높은 문서(⭐ 많은)의 내용에 더 중점을 둘 것{question_context}

출력 형식 - 각 인사이트를 구분자로 분리:
===INSIGHT_1===
첫 번째 핵심 인사이트: 구체적이고 깊이 있는 통찰 내용
===INSIGHT_2===
두 번째 핵심 인사이트: 패턴이나 트렌드 분석
===INSIGHT_3===
세 번째 핵심 인사이트: 원인과 결과 관계 분석
===INSIGHT_4===
네 번째 핵심 인사이트: 미래 전망이나 시사점

인사이트 예시:
- "기업들이 오픈소스 LLM을 채택하는 이유는 비용 절감보다 유연성과 커스터마이징 가능성 때문이며, 이는 기술 자립도를 높이려는 전략적 판단으로 보인다"
- "상업용 모델과 오픈소스 모델의 성능 격차가 빠르게 줄어들면서 시장 판도가 변화하고 있으며, 향후 2-3년 내 오픈소스가 주류가 될 가능성이 높다" """),
            ("user", "원본 질문: {question}\n\n문서 요약들 (신뢰도 기반 가중치 적용):\n{summaries}")
        ])
        
        return await self.llm_service._call_llm(
            prompt_template=prompt_template,
            input_variables={"question": user_question, "summaries": summaries_text},
            agent_type="insight_generator",
            output_parser=self.llm_service.str_parser,
            expected_type=str
        )
    
    def _parse_insights_response(
        self, 
        response_content: str, 
        original_summaries: List[DocumentSummary],
        user_question: str
    ) -> List[Insight]:
        """
        LLM 응답을 파싱하여 Insight 객체 생성
        
        Args:
            response_content: LLM 응답 내용
            original_summaries: 원본 DocumentSummary 리스트
            user_question: 사용자 질문
            
        Returns:
            List[Insight]: 파싱된 인사이트 객체 리스트
        """
        insights = []
        
        try:
            # 1단계: JSON 파싱 시도
            insight_texts = self._try_json_parsing(response_content)
            
            # 2단계: JSON 실패 시 구분자 기반 파싱 (폴백)
            if not insight_texts:
                self.json_parse_failures += 1
                insight_texts = self._parse_with_separators(response_content)
                self.separator_parse_count += 1
                logger.debug("구분자 기반 파싱으로 폴백")
            
            # 3단계: Insight 객체 생성
            for i, insight_text in enumerate(insight_texts):
                insight_text = insight_text.strip()
                
                # 최소 길이 검증
                if len(insight_text) < self.min_insight_length:
                    logger.debug(f"인사이트 {i+1} 제외: 너무 짧음 ({len(insight_text)}자)")
                    continue
                
                # 인사이트 객체 생성
                insight = self._create_insight_object(
                    content=insight_text,
                    index=i,
                    original_summaries=original_summaries,
                    user_question=user_question
                )
                
                insights.append(insight)
            
            logger.info(f"인사이트 파싱 완료: {len(insights)}개 생성")
            return insights
            
        except Exception as e:
            logger.error(f"인사이트 응답 파싱 실패: {e}")
            return []
    
    def _try_json_parsing(self, content: str) -> List[str]:
        """
        JSON 파싱 시도
        
        Args:
            content: 파싱할 내용
            
        Returns:
            List[str]: 파싱된 인사이트 텍스트 리스트 (실패 시 빈 리스트)
        """
        try:
            # JSON 배열 패턴 찾기
            json_pattern = r'\[.*?\]'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return [str(item) for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    continue
            
            # 직접 JSON 파싱 시도
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            
            return []
            
        except Exception as e:
            logger.debug(f"JSON 파싱 실패: {e}")
            return []
    
    def _parse_with_separators(self, content: str) -> List[str]:
        """
        구분자를 사용한 응답 파싱 (폴백 전략)
        
        Args:
            content: 파싱할 응답 내용
            
        Returns:
            List[str]: 분리된 인사이트 텍스트 리스트
        """
        insights = []
        
        try:
            # ===INSIGHT_N=== 패턴으로 분리
            pattern = r'===INSIGHT_\d+===(.*?)(?====INSIGHT_\d+===|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            if matches:
                insights = [match.strip() for match in matches if match.strip()]
                logger.debug(f"===INSIGHT_N=== 패턴 파싱: {len(insights)}개")
            else:
                # 대안 구분자 시도: --- 또는 ===
                separators = ['---', '===', '***']
                for sep in separators:
                    parts = content.split(sep)
                    if len(parts) > 1:
                        insights = [part.strip() for part in parts if part.strip() and len(part.strip()) > 20]
                        if insights:
                            logger.debug(f"'{sep}' 구분자 파싱: {len(insights)}개")
                            break
            
            # 마지막 폴백: 줄바꿈 기반 분리
            if not insights:
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    # 번호나 불릿 포인트로 시작하는 라인 찾기
                    if (re.match(r'^\d+\.', line) or 
                        re.match(r'^[-*•]', line) or 
                        len(line) > self.min_insight_length):
                        # 번호나 기호 제거
                        cleaned = re.sub(r'^[\d\.\-\*•\s]+', '', line).strip()
                        if len(cleaned) >= self.min_insight_length:
                            insights.append(cleaned)
                
                if insights:
                    logger.debug(f"줄바꿈 기반 파싱: {len(insights)}개")
            
            return insights[:self.target_insight_count * 2]  # 최대 개수 제한
            
        except Exception as e:
            logger.error(f"구분자 파싱 실패: {e}")
            return []
    
    def _create_insight_object(
        self, 
        content: str, 
        index: int,
        original_summaries: List[DocumentSummary],
        user_question: str
    ) -> Insight:
        """
        Insight 객체 생성
        
        Args:
            content: 인사이트 내용
            index: 인사이트 인덱스
            original_summaries: 원본 요약 리스트
            user_question: 사용자 질문
            
        Returns:
            Insight: 생성된 인사이트 객체
        """
        # 카테고리 자동 분류
        category = self._classify_insight_category(content, user_question)
        
        # 신뢰도 점수 계산 (요약들의 평균 신뢰도 기반)
        confidence_score = self._calculate_insight_confidence(content, original_summaries)
        
        # 뒷받침하는 문서 해시 수집
        supporting_documents = [summary.document_hash for summary in original_summaries]
        
        return Insight(
            content=content,
            category=category,
            confidence_score=confidence_score,
            supporting_documents=supporting_documents,
            created_at=datetime.now()
        )
    
    def _classify_insight_category(self, content: str, user_question: str) -> str:
        """
        인사이트 내용을 분석하여 카테고리 분류
        
        Args:
            content: 인사이트 내용
            user_question: 사용자 질문
            
        Returns:
            str: 분류된 카테고리
        """
        content_lower = content.lower()
        
        # 키워드 기반 카테고리 분류
        trend_keywords = ['트렌드', '증가', '감소', '변화', '성장', '확대', '축소', '급증', '급감']
        analysis_keywords = ['분석', '원인', '이유', '때문', '결과', '영향', '상관관계', '관계']
        prediction_keywords = ['전망', '예상', '예측', '미래', '향후', '될 것', '할 것', '가능성']
        recommendation_keywords = ['권장', '추천', '제안', '해야', '필요', '중요', '고려']
        
        # 각 카테고리별 키워드 매칭 점수 계산
        scores = {
            'trend': sum(1 for keyword in trend_keywords if keyword in content_lower),
            'analysis': sum(1 for keyword in analysis_keywords if keyword in content_lower),
            'prediction': sum(1 for keyword in prediction_keywords if keyword in content_lower),
            'recommendation': sum(1 for keyword in recommendation_keywords if keyword in content_lower)
        }
        
        # 최고 점수 카테고리 반환
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "general"
    
    def _calculate_insight_confidence(
        self, 
        insight_content: str, 
        summaries: List[DocumentSummary]
    ) -> float:
        """
        인사이트 신뢰도 점수 계산
        
        Args:
            insight_content: 인사이트 내용
            summaries: 원본 요약 리스트
            
        Returns:
            float: 신뢰도 점수 (0.0 ~ 1.0)
        """
        if not summaries:
            return 0.5
        
        # 요약들의 평균 신뢰도
        avg_confidence = sum(s.confidence_score for s in summaries) / len(summaries)
        
        # 인사이트 길이 보너스 (더 긴 인사이트가 더 신뢰도 높음)
        length_bonus = min(0.2, len(insight_content) / 500)  # 최대 0.2 보너스
        
        # 키워드 다양성 보너스 (더 다양한 키워드가 더 신뢰도 높음)
        keywords = extract_keywords(insight_content, max_keywords=10)
        diversity_bonus = min(0.1, len(keywords) / 20)  # 최대 0.1 보너스
        
        # 최종 신뢰도 계산
        final_confidence = avg_confidence + length_bonus + diversity_bonus
        
        return min(1.0, max(0.1, final_confidence))  # 0.1 ~ 1.0 범위로 제한
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            "total_summaries_processed": self.total_summaries_processed,
            "total_insights_generated": self.total_insights_generated,
            "json_parse_failures": self.json_parse_failures,
            "separator_parse_count": self.separator_parse_count,
            "success_rate_percent": round(
                (self.total_insights_generated / max(1, self.total_summaries_processed)) * 100, 2
            ),
            "min_insight_length": self.min_insight_length,
            "target_insight_count": self.target_insight_count
        }


# LangGraph 노드 함수
async def generate_insights_node(state: ResearchState) -> ResearchState:
    """
    인사이트 생성 LangGraph 노드
    
    State의 summaries를 받아서 인사이트를 생성하고 insights에 저장합니다.
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 인사이트가 추가된 업데이트된 상태
    """
    logger.info("=== 인사이트 생성 노드 시작 ===")
    
    try:
        # 현재 단계 설정
        state = StateManager.set_step(state, "generating_insights", "🧠 인사이트 생성 중...")
        
        # 요약 확인
        if not state.get("summaries"):
            error_msg = "인사이트 생성할 요약이 없습니다."
            logger.warning(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # DocumentSummary 객체 생성 (요약이 문자열 리스트로 저장되어 있는 경우)
        summaries = []
        for i, summary_text in enumerate(state["summaries"]):
            try:
                # 간단한 DocumentSummary 객체 생성
                summary = DocumentSummary(
                    document_hash=f"summary_{i}",
                    summary=summary_text,
                    confidence_score=0.7,  # 기본 신뢰도
                    word_count=len(summary_text)
                )
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"DocumentSummary 생성 실패: {e}")
                continue
        
        if not summaries:
            error_msg = "유효한 DocumentSummary 객체를 생성할 수 없습니다."
            logger.error(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 인사이트 생성기 생성 및 실행
        generator = InsightGenerator()
        insights = await generator.generate_insights(
            summaries=summaries,
            user_question=state.get("user_input", "")
        )
        
        if not insights:
            error_msg = "인사이트 생성에 실패했습니다."
            logger.error(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 인사이트를 문자열 리스트로 변환하여 상태에 저장
        insight_texts = [insight.content for insight in insights]
        
        # 상태 업데이트
        new_state = state.copy()
        new_state["insights"] = insight_texts
        
        # 처리 통계 로깅
        stats = generator.get_processing_stats()
        logger.info(f"인사이트 생성 완료: {stats}")
        
        new_state = StateManager.add_log(
            new_state,
            f"🧠 인사이트 생성 완료: {len(insight_texts)}개 인사이트 생성"
        )
        
        # 인사이트 내용 요약 로깅 (디버깅용)
        for i, insight in enumerate(insights, 1):
            logger.debug(f"인사이트 {i} ({insight.category}): {insight.content[:100]}...")
        
        logger.info("=== 인사이트 생성 노드 완료 ===")
        return new_state
        
    except Exception as e:
        error_msg = f"인사이트 생성 노드에서 오류 발생: {str(e)}"
        logger.error(error_msg)
        return StateManager.set_error(state, error_msg)


# 유틸리티 함수들
def create_insight_generator(target_count: int = 4) -> InsightGenerator:
    """
    인사이트 생성기 인스턴스 생성 헬퍼 함수
    
    Args:
        target_count: 목표 인사이트 개수
        
    Returns:
        InsightGenerator: 설정된 인사이트 생성기 인스턴스
    """
    return InsightGenerator(target_insight_count=target_count)

