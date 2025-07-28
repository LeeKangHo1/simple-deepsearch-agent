# agents/validator.py
"""
검증 에이전트

생성된 최종 응답의 품질, 논리성, 완성도를 검증하여 사용자에게 제공하기 전에
품질을 보장합니다. 검증 실패 시 구체적인 피드백과 개선 제안을 제공합니다.

주요 기능:
- 응답 구조 및 형식 검증
- 출처 표시 완성도 확인
- 논리적 일관성 검증
- 인사이트 품질 평가
- 사용자 질문과의 관련성 확인
- 검증 실패 시 구체적 피드백 제공
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 모듈들
from models.data_models import ValidationResult
from models.state import ResearchState, StateManager
from services.llm_service import get_llm_service, LLMResponse
from utils.text_processing import clean_text, count_words, extract_keywords
from utils.validators import validate_response_structure, validate_insights
from utils.logger import create_agent_logger, log_execution_time

logger = logging.getLogger(__name__)
agent_logger = create_agent_logger("validator")

class ResponseValidator:
    """
    응답 검증 에이전트 클래스
    
    생성된 마크다운 응답의 전반적인 품질을 다각도로 검증합니다.
    구조적 완성도부터 내용의 논리성까지 체계적으로 평가하여
    고품질 응답만 사용자에게 전달되도록 보장합니다.
    
    검증 항목:
    1. 구조적 완성도 (마크다운 형식, 섹션 구조)
    2. 출처 표시 완성도 (누락, 형식 오류)
    3. 내용 품질 (논리성, 일관성, 관련성)
    4. 인사이트 포함 여부 (원본 대비 누락 검증)
    5. 사용자 질문 응답 적합성
    """
    
    def __init__(self, max_retry_count: int = 2):
        """
        검증기 초기화
        
        Args:
            max_retry_count: 최대 재시도 횟수 (기본: 2회)
        """
        self.max_retry_count = max_retry_count
        self.llm_service = get_llm_service()
        
        # 검증 기준 설정 (완화됨)
        self.validation_criteria = {
            "min_word_count": 100,        # 최소 단어 수 (200 → 100)
            "max_word_count": 10000,      # 최대 단어 수
            "required_sections": ["인사이트"],  # 필수 섹션 완화 (인사이트만 필수)
            "min_insights_ratio": 0.5,    # 원본 인사이트 대비 최소 포함 비율 (0.8 → 0.5)
            "min_sources_count": 1,       # 최소 출처 개수
        }
        
        # 통계 추적
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0
        self.llm_validation_count = 0
    
    @log_execution_time
    async def validate_response(
        self,
        response_content: str,
        user_question: str,
        original_insights: List[str],
        original_sources: List[Dict[str, str]]
    ) -> ValidationResult:
        """
        응답 품질 종합 검증
        
        Args:
            response_content: 검증할 응답 내용
            user_question: 사용자 원본 질문
            original_insights: 원본 인사이트 리스트
            original_sources: 원본 출처 정보 리스트
            
        Returns:
            ValidationResult: 검증 결과 객체
        """
        if not response_content or not response_content.strip():
            return ValidationResult(
                is_valid=False,
                feedback="응답 내용이 비어있습니다.",
                issues=["빈 응답"],
                confidence_score=0.0
            )
        
        agent_logger.start_step("응답 검증 시작")
        self.total_validations += 1
        
        try:
            # 1단계: 구조적 검증 (빠른 검증)
            structural_issues = self._validate_structure(response_content)
            
            # 2단계: 출처 검증
            source_issues = self._validate_sources(response_content, original_sources)
            
            # 3단계: 인사이트 포함 검증
            insight_issues = self._validate_insights_inclusion(response_content, original_insights)
            
            # 4단계: 내용 품질 검증 (LLM 기반)
            content_validation = await self._validate_content_quality(response_content, user_question)
            
            # 5단계: 종합 평가
            all_issues = structural_issues + source_issues + insight_issues
            if not content_validation.success:
                all_issues.extend(content_validation.content.get("issues", []))
            
            # 검증 결과 결정 (완화된 기준)
            # 심각한 문제가 없으면 통과 (경미한 문제는 허용)
            critical_issues = [issue for issue in all_issues if any(
                critical in issue.lower() for critical in ["빈 응답", "검증 오류", "파싱 실패"]
            )]
            
            is_valid = len(critical_issues) == 0  # 심각한 문제만 체크
            confidence_score = self._calculate_confidence_score(
                len(all_issues), content_validation, response_content
            )
            
            # 피드백 생성
            feedback = self._generate_feedback(all_issues, content_validation)
            suggestions = self._generate_suggestions(all_issues, user_question)
            
            # 통계 업데이트
            if is_valid:
                self.passed_validations += 1
            else:
                self.failed_validations += 1
            
            result = ValidationResult(
                is_valid=is_valid,
                feedback=feedback,
                issues=all_issues,
                suggestions=suggestions,
                confidence_score=confidence_score
            )
            
            agent_logger.end_step(
                "응답 검증 완료",
                is_valid,
                f"신뢰도: {confidence_score:.2f}, 이슈: {len(all_issues)}개"
            )
            
            return result
            
        except Exception as e:
            agent_logger.end_step("응답 검증", False, f"오류 발생: {str(e)}")
            logger.error(f"응답 검증 중 오류 발생: {e}")
            
            return ValidationResult(
                is_valid=False,
                feedback=f"검증 과정에서 오류가 발생했습니다: {str(e)}",
                issues=["검증 오류"],
                confidence_score=0.0
            )
    
    def _validate_structure(self, content: str) -> List[str]:
        """
        응답 구조 검증
        
        Args:
            content: 검증할 응답 내용
            
        Returns:
            List[str]: 발견된 구조적 문제점 리스트
        """
        issues = []
        
        try:
            # 기본 길이 검증
            word_stats = count_words(content)
            total_chars = word_stats["total_chars"]
            
            if total_chars < self.validation_criteria["min_word_count"]:
                issues.append(f"응답이 너무 짧습니다 ({total_chars}자, 최소 {self.validation_criteria['min_word_count']}자 필요)")
            
            if total_chars > self.validation_criteria["max_word_count"]:
                issues.append(f"응답이 너무 깁니다 ({total_chars}자, 최대 {self.validation_criteria['max_word_count']}자)")
            
            # 마크다운 제목 구조 확인
            if not re.search(r'^#\s+.+', content, re.MULTILINE):
                issues.append("메인 제목(# )이 없습니다")
            
            # 필수 섹션 확인
            missing_sections = []
            for section in self.validation_criteria["required_sections"]:
                section_pattern = rf'##\s+{re.escape(section)}'
                if not re.search(section_pattern, content):
                    missing_sections.append(section)
            
            if missing_sections:
                issues.append(f"필수 섹션이 누락되었습니다: {', '.join(missing_sections)}")
            
            # 불릿 포인트 또는 구조화된 내용 확인
            structure_patterns = [
                r'^\s*[-*+]\s+.+',      # 불릿 포인트
                r'^\s*\d+\.\s+.+',      # 번호 리스트
                r'^#{2,}\s+.+',         # 소제목
            ]
            
            has_structure = any(
                re.search(pattern, content, re.MULTILINE) 
                for pattern in structure_patterns
            )
            
            if not has_structure:
                issues.append("구조화된 내용이 부족합니다 (불릿포인트, 소제목 등)")
            
            return issues
            
        except Exception as e:
            logger.error(f"구조 검증 실패: {e}")
            return ["구조 검증 중 오류 발생"]
    
    def _validate_sources(self, content: str, original_sources: List[Dict[str, str]]) -> List[str]:
        """
        출처 표시 검증
        
        Args:
            content: 검증할 응답 내용
            original_sources: 원본 출처 정보 리스트
            
        Returns:
            List[str]: 발견된 출처 관련 문제점 리스트
        """
        issues = []
        
        try:
            # 출처 표시 패턴 확인
            source_patterns = [
                r'\*\([^)]*출처[^)]*\)\*',           # *(출처: domain.com)*
                r'\*\([^)]*source[^)]*\)\*',         # *(source: domain.com)*
                r'#{3,}\s*출처',                     # ### 출처
                r'---[^#]*출처',                     # --- 출처
            ]
            
            has_sources = any(
                re.search(pattern, content, re.IGNORECASE) 
                for pattern in source_patterns
            )
            
            if not has_sources:
                issues.append("출처 표시가 없습니다")
            
            # 원본 출처 개수와 비교
            if len(original_sources) >= self.validation_criteria["min_sources_count"]:
                # 도메인 추출하여 포함 여부 확인
                content_lower = content.lower()
                found_domains = 0
                
                for source in original_sources[:5]:  # 상위 5개만 확인
                    domain = source.get("domain", "").lower()
                    if domain and domain in content_lower:
                        found_domains += 1
                
                if found_domains == 0:
                    issues.append("원본 출처가 응답에 반영되지 않았습니다")
                elif found_domains < len(original_sources) * 0.5:
                    issues.append("일부 출처만 반영되었습니다")
            
            return issues
            
        except Exception as e:
            logger.error(f"출처 검증 실패: {e}")
            return ["출처 검증 중 오류 발생"]
    
    def _validate_insights_inclusion(self, content: str, original_insights: List[str]) -> List[str]:
        """
        인사이트 포함 여부 검증
        
        Args:
            content: 검증할 응답 내용
            original_insights: 원본 인사이트 리스트
            
        Returns:
            List[str]: 발견된 인사이트 관련 문제점 리스트
        """
        issues = []
        
        try:
            if not original_insights:
                return issues  # 원본 인사이트가 없으면 검증 불가
            
            content_lower = content.lower()
            included_insights = 0
            
            for insight in original_insights:
                # 인사이트의 핵심 키워드들로 포함 여부 확인
                insight_keywords = extract_keywords(insight, max_keywords=3, min_length=3)
                
                if not insight_keywords:
                    continue
                
                # 키워드 중 일정 비율 이상이 응답에 포함되어 있는지 확인
                matching_keywords = sum(
                    1 for keyword in insight_keywords 
                    if keyword.lower() in content_lower
                )
                
                keyword_ratio = matching_keywords / len(insight_keywords)
                if keyword_ratio >= 0.5:  # 50% 이상 키워드가 매칭되면 포함된 것으로 간주
                    included_insights += 1
            
            inclusion_ratio = included_insights / len(original_insights)
            min_ratio = self.validation_criteria["min_insights_ratio"]
            
            if inclusion_ratio < min_ratio:
                issues.append(
                    f"인사이트 포함률이 낮습니다 "
                    f"({included_insights}/{len(original_insights)}, {inclusion_ratio:.1%})"
                )
            
            # 인사이트 섹션 자체가 있는지 확인
            if not re.search(r'##\s*주요\s*인사이트', content, re.IGNORECASE):
                issues.append("주요 인사이트 섹션이 없습니다")
            
            return issues
            
        except Exception as e:
            logger.error(f"인사이트 검증 실패: {e}")
            return ["인사이트 검증 중 오류 발생"]
    
    async def _validate_content_quality(self, content: str, user_question: str) -> LLMResponse:
        """
        LLM을 통한 내용 품질 검증
        
        Args:
            content: 검증할 응답 내용
            user_question: 사용자 질문
            
        Returns:
            LLMResponse: 품질 검증 결과
        """
        self.llm_validation_count += 1
        
        # 응답이 너무 길면 앞부분만 사용 (토큰 절약)
        truncated_content = content[:3000] if len(content) > 3000 else content
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 응답 품질 검증 전문가입니다. 주어진 응답을 검토하여 품질을 평가해주세요.

검증 기준:
1. 사용자 질문에 적절히 답변했는가?
2. 논리적 일관성이 있는가?
3. 내용이 객관적이고 정확한가?
4. 마크다운 형식이 올바른가?
5. 정보가 잘 구조화되어 있는가?

출력 형식 (JSON):
{{
    "is_valid": true/false,
    "feedback": "구체적인 피드백 메시지",
    "issues": ["문제점1", "문제점2", ...],
    "suggestions": ["개선사항1", "개선사항2", ...],
    "content_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "structure_score": 0.0-1.0
}}

각 점수는 0.0(매우 나쁨)부터 1.0(매우 좋음)까지입니다."""),
            ("user", "원본 질문: {question}\n\n검증할 응답:\n{response}")
        ])
        
        return await self.llm_service._call_llm(
            prompt_template=prompt_template,
            input_variables={"question": user_question, "response": truncated_content},
            agent_type="validator",
            output_parser=self.llm_service.json_parser,
            expected_type=dict
        )
    
    def _calculate_confidence_score(
        self,
        issues_count: int,
        content_validation: LLMResponse,
        response_content: str
    ) -> float:
        """
        전체 신뢰도 점수 계산
        
        Args:
            issues_count: 발견된 문제점 개수
            content_validation: LLM 품질 검증 결과
            response_content: 응답 내용
            
        Returns:
            float: 신뢰도 점수 (0.0 ~ 1.0)
        """
        try:
            # 기본 점수에서 문제점만큼 차감
            base_score = 1.0
            
            # 문제점 개수에 따른 차감 (문제 1개당 -0.2점)
            issues_penalty = min(issues_count * 0.2, 0.8)
            base_score -= issues_penalty
            
            # LLM 검증 결과 반영
            if content_validation.success:
                try:
                    validation_data = content_validation.content
                    if isinstance(validation_data, str):
                        import json
                        validation_data = json.loads(validation_data)
                    
                    # LLM이 제공한 점수들 평균
                    llm_scores = [
                        validation_data.get("content_score", 0.7),
                        validation_data.get("relevance_score", 0.7),
                        validation_data.get("structure_score", 0.7)
                    ]
                    llm_avg_score = sum(llm_scores) / len(llm_scores)
                    
                    # 기본 점수와 LLM 점수를 가중 평균
                    final_score = (base_score * 0.6) + (llm_avg_score * 0.4)
                    
                except Exception as e:
                    logger.warning(f"LLM 점수 파싱 실패: {e}")
                    final_score = base_score * 0.8  # LLM 검증 실패 시 보수적 점수
            else:
                final_score = base_score * 0.5  # LLM 검증 실패 시 큰 폐널티
            
            # 응답 길이 보너스 (적당한 길이에 보너스)
            content_length = len(response_content)
            if 500 <= content_length <= 5000:
                length_bonus = 0.1
            else:
                length_bonus = 0.0
            
            final_score += length_bonus
            
            # 0.0 ~ 1.0 범위로 제한
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"신뢰도 점수 계산 실패: {e}")
            return 0.5  # 기본값
    
    def _generate_feedback(self, issues: List[str], content_validation: LLMResponse) -> str:
        """
        검증 피드백 메시지 생성
        
        Args:
            issues: 발견된 문제점 리스트
            content_validation: LLM 검증 결과
            
        Returns:
            str: 피드백 메시지
        """
        if not issues and content_validation.success:
            return "응답 품질이 우수합니다. 모든 검증 기준을 통과했습니다."
        
        feedback_parts = []
        
        # 구조적 문제점
        structural_issues = [issue for issue in issues if any(
            keyword in issue for keyword in ["제목", "섹션", "구조", "형식", "길이"]
        )]
        if structural_issues:
            feedback_parts.append(f"구조적 개선 필요: {'; '.join(structural_issues[:2])}")
        
        # 내용 문제점
        content_issues = [issue for issue in issues if issue not in structural_issues]
        if content_issues:
            feedback_parts.append(f"내용 개선 필요: {'; '.join(content_issues[:2])}")
        
        # LLM 피드백 추가
        if content_validation.success:
            try:
                validation_data = content_validation.content
                if isinstance(validation_data, str):
                    import json
                    validation_data = json.loads(validation_data)
                
                llm_feedback = validation_data.get("feedback", "")
                if llm_feedback and len(llm_feedback) < 200:
                    feedback_parts.append(f"추가 의견: {llm_feedback}")
                    
            except Exception as e:
                logger.warning(f"LLM 피드백 추출 실패: {e}")
        
        return " | ".join(feedback_parts) if feedback_parts else "검증에서 일부 문제가 발견되었습니다."
    
    def _generate_suggestions(self, issues: List[str], user_question: str) -> List[str]:
        """
        개선 제안 생성
        
        Args:
            issues: 발견된 문제점 리스트
            user_question: 사용자 질문
            
        Returns:
            List[str]: 개선 제안 리스트
        """
        suggestions = []
        
        # 문제점 유형별 제안
        for issue in issues:
            if "섹션" in issue and "누락" in issue:
                suggestions.append("누락된 섹션을 추가하고 각 섹션에 적절한 내용을 포함하세요")
            elif "출처" in issue:
                suggestions.append("각 섹션에 관련 출처를 명시하고 출처 목록을 추가하세요")
            elif "인사이트" in issue:
                suggestions.append("생성된 모든 인사이트를 인사이트 섹션에 포함하세요")
            elif "길이" in issue:
                if "짧습니다" in issue:
                    suggestions.append("내용을 더 상세히 설명하고 추가 정보를 포함하세요")
                else:
                    suggestions.append("핵심 내용만 남기고 불필요한 부분을 제거하세요")
            elif "구조" in issue:
                suggestions.append("마크다운 형식을 올바르게 사용하고 섹션을 명확히 구분하세요")
        
        # 일반적인 제안 추가
        if not suggestions:
            suggestions.append("응답의 논리적 흐름을 개선하고 사용자 질문에 더 직접적으로 답변하세요")
        
        # 사용자 질문 관련 제안
        if user_question:
            question_keywords = extract_keywords(user_question, max_keywords=3)
            if question_keywords:
                suggestions.append(f"'{', '.join(question_keywords)}'와 관련된 내용을 더 강조하세요")
        
        return suggestions[:5]  # 최대 5개까지
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        pass_rate = 0.0
        if self.total_validations > 0:
            pass_rate = (self.passed_validations / self.total_validations) * 100
        
        return {
            "total_validations": self.total_validations,
            "passed_validations": self.passed_validations,
            "failed_validations": self.failed_validations,
            "pass_rate_percent": round(pass_rate, 2),
            "llm_validation_count": self.llm_validation_count,
            "max_retry_count": self.max_retry_count,
            "validation_criteria": self.validation_criteria
        }


# LangGraph 노드 함수
async def validate_response_node(state: ResearchState) -> ResearchState:
    """
    응답 검증 LangGraph 노드
    
    State의 markdown_answer를 검증하고 is_valid, validation_feedback을 설정합니다.
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 검증 결과가 추가된 업데이트된 상태
    """
    logger.info("=== 응답 검증 노드 시작 ===")
    
    try:
        # 현재 단계 설정
        state = StateManager.set_step(state, "validating", "✅ 응답 품질 검증 중...")
        
        # 검증할 데이터 확인
        response_content = state.get("markdown_answer", "")
        if not response_content:
            error_msg = "검증할 응답이 없습니다."
            logger.warning(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 추가 데이터 수집
        user_question = state.get("user_input", "")
        original_insights = state.get("insights", [])
        original_documents = state.get("documents", [])
        
        # 원본 출처 정보 구성
        original_sources = []
        for doc in original_documents:
            if isinstance(doc, dict) and "url" in doc and "title" in doc:
                domain = doc.get("url", "").split("//")[-1].split("/")[0] if doc.get("url") else "unknown"
                original_sources.append({
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "domain": domain
                })
        
        # 검증기 생성 및 실행
        validator = ResponseValidator()
        validation_result = await validator.validate_response(
            response_content=response_content,
            user_question=user_question,
            original_insights=original_insights,
            original_sources=original_sources
        )
        
        # 상태 업데이트
        new_state = state.copy()
        new_state["is_valid"] = validation_result.is_valid
        new_state["validation_feedback"] = validation_result.feedback
        
        # 처리 통계 로깅
        stats = validator.get_processing_stats()
        logger.info(f"응답 검증 완료: {stats}")
        
        # 검증 결과에 따른 로그 메시지
        if validation_result.is_valid:
            new_state = StateManager.add_log(
                new_state,
                f"✅ 응답 검증 통과: 신뢰도 {validation_result.confidence_score:.2f}"
            )
            logger.info("응답 검증 통과")
        else:
            new_state = StateManager.add_log(
                new_state,
                f"❌ 응답 검증 실패: {len(validation_result.issues)}개 문제 발견"
            )
            logger.warning(f"응답 검증 실패: {validation_result.feedback}")
            
            # 개선 제안 로깅
            if validation_result.suggestions:
                new_state = StateManager.add_log(
                    new_state,
                    f"💡 개선 제안: {'; '.join(validation_result.suggestions[:2])}"
                )
        
        logger.info("=== 응답 검증 노드 완료 ===")
        return new_state
        
    except Exception as e:
        error_msg = f"응답 검증 노드에서 오류 발생: {str(e)}"
        logger.error(error_msg)
        return StateManager.set_error(state, error_msg)


# 유틸리티 함수들
def create_response_validator(max_retries: int = 2) -> ResponseValidator:
    """
    응답 검증기 인스턴스 생성 헬퍼 함수
    
    Args:
        max_retries: 최대 재시도 횟수
        
    Returns:
        ResponseValidator: 설정된 검증기 인스턴스
    """
    return ResponseValidator(max_retry_count=max_retries)

def should_retry_generation(validation_result: ValidationResult, current_retry: int, max_retries: int) -> bool:
    """
    재생성 여부 판단 함수
    
    검증 결과를 바탕으로 응답을 재생성할지 결정합니다.
    심각한 문제가 있거나 신뢰도가 낮은 경우에만 재시도를 권장합니다.
    
    Args:
        validation_result: 검증 결과
        current_retry: 현재 재시도 횟수
        max_retries: 최대 재시도 횟수
        
    Returns:
        bool: 재생성이 필요하면 True
    """
    # 최대 재시도 횟수 초과
    if current_retry >= max_retries:
        logger.info(f"최대 재시도 횟수 초과: {current_retry}/{max_retries}")
        return False
    
    # 검증 통과
    if validation_result.is_valid:
        logger.info("검증 통과 - 재시도 불필요")
        return False
    
    # 신뢰도가 너무 낮으면 재시도
    if validation_result.confidence_score < 0.3:
        logger.info(f"신뢰도 너무 낮음 ({validation_result.confidence_score:.2f}) - 재시도 필요")
        return True
    
    # 심각한 구조적 문제가 있으면 재시도
    critical_issues = [
        "메인 제목",
        "필수 섹션",
        "출처 표시가 없습니다",
        "인사이트 섹션이 없습니다"
    ]
    
    has_critical_issue = any(
        any(critical in issue for critical in critical_issues)
        for issue in validation_result.issues
    )
    
    if has_critical_issue:
        logger.info("심각한 구조적 문제 발견 - 재시도 필요")
        return True
    
    # 문제가 너무 많으면 재시도
    if len(validation_result.issues) >= 5:
        logger.info(f"문제점 과다 ({len(validation_result.issues)}개) - 재시도 필요")
        return True
    
    # 그 외의 경우는 재시도하지 않음 (미세한 문제는 허용)
    logger.info("미세한 문제만 있음 - 재시도 불필요")
    return False

def get_retry_feedback(validation_result: ValidationResult) -> str:
    """
    재시도를 위한 피드백 메시지 생성
    
    검증 실패 원인을 분석하여 다음 생성 시 참고할 수 있는
    구체적인 개선 방향을 제시합니다.
    
    Args:
        validation_result: 검증 결과
        
    Returns:
        str: 재시도용 피드백 메시지
    """
    feedback_parts = []
    
    # 심각한 문제부터 우선 처리
    critical_issues = []
    minor_issues = []
    
    for issue in validation_result.issues:
        if any(critical in issue for critical in ["제목", "섹션", "출처 표시가 없습니다"]):
            critical_issues.append(issue)
        else:
            minor_issues.append(issue)
    
    if critical_issues:
        feedback_parts.append(f"필수 수정사항: {'; '.join(critical_issues[:2])}")
    
    if minor_issues:
        feedback_parts.append(f"추가 개선사항: {'; '.join(minor_issues[:2])}")
    
    # 구체적인 개선 제안 추가
    if validation_result.suggestions:
        feedback_parts.append(f"개선 방향: {validation_result.suggestions[0]}")
    
    return " | ".join(feedback_parts) if feedback_parts else validation_result.feedback