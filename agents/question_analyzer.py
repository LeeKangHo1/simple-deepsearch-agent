# agents/question_analyzer.py
"""
질문 분석 에이전트

사용자의 복잡한 질문을 여러 개의 구체적인 검색 쿼리로 분해하는 에이전트입니다.
LLM을 활용하여 다양한 관점에서 질문을 분석하고, 효과적인 검색을 위한 하위 쿼리들을 생성합니다.

주요 기능:
- 사용자 질문의 의도 파악 및 키워드 추출
- 다각도 관점에서 하위 질문 생성 (3-5개)
- 검색에 최적화된 쿼리 형태로 변환
- 중복 쿼리 제거 및 품질 검증
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
import time

# 프로젝트 모듈들
from models.state import ResearchState, StateManager
from services.llm_service import get_llm_service, LLMResponse
from utils.validators import validate_search_query, validate_query_list, sanitize_input
from utils.logger import get_agent_logger, log_agent_start, log_agent_end

logger = get_agent_logger("question_analyzer")


class QuestionAnalyzer:
    """
    질문 분석 에이전트 클래스
    
    사용자의 자연어 질문을 받아서 효과적인 검색을 위한
    여러 개의 하위 쿼리로 분해하는 역할을 담당합니다.
    
    사용 예시:
        analyzer = QuestionAnalyzer()
        result = await analyzer.analyze("오픈소스 LLM 트렌드 알려줘")
        # result: ["오픈소스 LLM 최신 동향", "LLaMA Mistral 성능 비교", ...]
    """
    
    def __init__(self, max_queries: int = 4, enable_keyword_extraction: bool = True):
        """
        질문 분석 에이전트 초기화
        
        Args:
            max_queries: 생성할 최대 쿼리 개수 (기본 4개)
            enable_keyword_extraction: 키워드 추출 기능 활성화 여부
        """
        self.llm_service = get_llm_service()
        self.max_queries = max_queries
        self.enable_keyword_extraction = enable_keyword_extraction
        
        # 성능 통계
        self.total_requests = 0
        self.total_queries_generated = 0
        self.avg_processing_time = 0.0
        
        logger.info(f"질문 분석 에이전트 초기화 완료 (최대 쿼리: {max_queries}개)")
    
    async def analyze_question(self, user_input: str) -> List[str]:
        """
        사용자 질문을 분석하여 하위 검색 쿼리 생성
        
        이 메서드는 단독으로 호출 가능한 공개 인터페이스입니다.
        LangGraph 워크플로우 외부에서도 사용할 수 있습니다.
        
        Args:
            user_input: 사용자의 원본 질문
            
        Returns:
            List[str]: 생성된 하위 검색 쿼리 리스트
            
        Raises:
            ValueError: 입력이 유효하지 않은 경우
            Exception: LLM 호출 실패 등 기타 오류
        """
        log_agent_start("question_analyzer", user_input)
        start_time = time.time()
        
        try:
            # 1단계: 입력 검증 및 정제
            validated_input = self._validate_and_clean_input(user_input)
            logger.debug(f"입력 검증 완료: '{validated_input[:50]}...'")
            
            # 2단계: LLM을 통한 하위 쿼리 생성
            queries = await self._generate_sub_queries(validated_input)
            logger.debug(f"LLM 쿼리 생성 완료: {len(queries)}개")
            
            # 3단계: 쿼리 후처리 (검증, 중복 제거, 품질 개선)
            final_queries = self._post_process_queries(queries, validated_input)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_statistics(len(final_queries), processing_time)
            
            log_agent_end("question_analyzer", success=True, output_data=final_queries)
            logger.info(f"질문 분석 완료: {len(final_queries)}개 쿼리 생성 ({processing_time:.2f}초)")
            
            return final_queries
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_statistics(0, processing_time)
            
            log_agent_end("question_analyzer", success=False, error=str(e))
            logger.error(f"질문 분석 실패 ({processing_time:.2f}초): {e}")
            raise
    
    async def process_state(self, state: ResearchState) -> ResearchState:
        """
        LangGraph 워크플로우용 상태 처리 메서드
        
        ResearchState를 입력받아 질문 분석 결과를 추가하고 반환합니다.
        이 메서드는 LangGraph 노드에서 호출됩니다.
        
        Args:
            state: 현재 연구 상태
            
        Returns:
            ResearchState: 질문 분석 결과가 추가된 상태
        """
        # 진행 상태 업데이트
        new_state = StateManager.set_step(
            state, 
            "analyzing", 
            "사용자 질문을 분석하여 검색 쿼리를 생성 중..."
        )
        
        try:
            user_input = state["user_input"]
            if not user_input:
                raise ValueError("사용자 입력이 비어있습니다.")
            
            # 질문 분석 수행
            sub_queries = await self.analyze_question(user_input)
            
            # 상태에 결과 저장
            new_state = new_state.copy()
            new_state["sub_queries"] = sub_queries
            
            # 성공 로그 추가
            new_state = StateManager.add_log(
                new_state, 
                f"✅ 질문 분석 완료: {len(sub_queries)}개 검색 쿼리 생성"
            )
            
            logger.info(f"상태 처리 완료: {len(sub_queries)}개 쿼리를 상태에 저장")
            return new_state
            
        except Exception as e:
            # 오류 상태 설정
            error_state = StateManager.set_error(new_state, f"질문 분석 실패: {e}")
            logger.error(f"상태 처리 실패: {e}")
            return error_state
    
    def _validate_and_clean_input(self, user_input: str) -> str:
        """
        사용자 입력을 검증하고 정제합니다.
        
        Args:
            user_input: 원본 사용자 입력
            
        Returns:
            str: 검증되고 정제된 입력
            
        Raises:
            ValueError: 입력이 유효하지 않은 경우
        """
        if not user_input or not isinstance(user_input, str):
            raise ValueError("사용자 입력이 비어있거나 유효하지 않습니다.")
        
        # 입력 정제 (보안 및 품질 향상)
        cleaned_input = sanitize_input(user_input, max_length=1000)
        
        if len(cleaned_input.strip()) < 3:
            raise ValueError("질문이 너무 짧습니다. 최소 3자 이상 입력해주세요.")
        
        if len(cleaned_input) > 500:
            raise ValueError("질문이 너무 깁니다. 500자 이내로 입력해주세요.")
        
        return cleaned_input
    
    async def _generate_sub_queries(self, question: str) -> List[str]:
        """
        LLM을 사용하여 하위 검색 쿼리 생성
        
        Args:
            question: 분석할 질문
            
        Returns:
            List[str]: 생성된 하위 쿼리 리스트
            
        Raises:
            Exception: LLM 호출 실패 시
        """
        logger.debug(f"LLM 하위 쿼리 생성 시작: '{question[:50]}...'")
        
        try:
            # LLM 서비스를 통한 쿼리 생성
            response: LLMResponse = await self.llm_service.generate_sub_queries(
                user_question=question,
                num_queries=self.max_queries
            )
            
            if not response.success:
                raise Exception(f"LLM 호출 실패: {response.error_message}")
            
            # 응답 파싱 (JSON 배열 형태로 받음)
            try:
                queries = json.loads(response.content)
                if not isinstance(queries, list):
                    raise ValueError("LLM 응답이 배열 형태가 아닙니다.")
                
                # 문자열이 아닌 항목 제거
                queries = [str(q).strip() for q in queries if q and str(q).strip()]
                
                logger.debug(f"LLM 응답 파싱 완료: {len(queries)}개 쿼리")
                return queries
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 문자열을 직접 처리
                logger.warning("JSON 파싱 실패, 문자열 직접 처리 시도")
                return self._parse_queries_from_text(response.content)
                
        except Exception as e:
            logger.error(f"하위 쿼리 생성 실패: {e}")
            # 폴백: 키워드 기반 쿼리 생성
            return self._generate_fallback_queries(question)
    
    def _parse_queries_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 쿼리를 추출하는 폴백 메서드
        
        LLM이 JSON 형태가 아닌 텍스트로 응답한 경우 사용합니다.
        
        Args:
            text: 파싱할 텍스트
            
        Returns:
            List[str]: 추출된 쿼리 리스트
        """
        queries = []
        
        # 여러 가지 형태의 리스트를 파싱 시도
        patterns = [
            r'"([^"]+)"',           # "쿼리" 형태
            r"'([^']+)'",           # '쿼리' 형태
            r'^\d+\.\s*(.+)$',      # 1. 쿼리 형태
            r'^[-*]\s*(.+)$',       # - 쿼리, * 쿼리 형태
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                import re
                matches = re.findall(pattern, line, re.MULTILINE)
                for match in matches:
                    query = match.strip()
                    if len(query) > 5 and query not in queries:  # 중복 제거
                        queries.append(query)
                        break
        
        # 패턴 매칭 실패 시 줄바꿈 기준으로 단순 분할
        if not queries:
            for line in lines:
                line = line.strip()
                if len(line) > 5:
                    queries.append(line)
        
        logger.debug(f"텍스트에서 {len(queries)}개 쿼리 추출")
        return queries[:self.max_queries]
    
    def _generate_fallback_queries(self, question: str) -> List[str]:
        """
        LLM 호출 실패 시 사용하는 폴백 쿼리 생성
        
        키워드 추출과 간단한 규칙을 사용하여 기본적인 쿼리를 생성합니다.
        
        Args:
            question: 원본 질문
            
        Returns:
            List[str]: 폴백 쿼리 리스트
        """
        logger.warning("폴백 쿼리 생성 모드 실행")
        
        queries = []
        
        # 원본 질문을 그대로 첫 번째 쿼리로 사용
        queries.append(question)
        
        if self.enable_keyword_extraction:
            try:
                from utils.text_processing import extract_keywords
                
                # 키워드 추출
                keywords = extract_keywords(question, max_keywords=8)
                
                if keywords:
                    # 키워드 조합으로 추가 쿼리 생성
                    if len(keywords) >= 2:
                        queries.append(' '.join(keywords[:3]))  # 상위 3개 키워드
                    
                    if len(keywords) >= 4:
                        queries.append(' '.join(keywords[2:5]))  # 중간 3개 키워드
                    
                    # 개별 중요 키워드들
                    for keyword in keywords[:2]:
                        if len(keyword) > 2:
                            queries.append(keyword)
                
            except ImportError:
                logger.warning("키워드 추출 모듈을 찾을 수 없습니다.")
        
        # 간단한 패턴 기반 쿼리 생성
        if '트렌드' in question or '동향' in question:
            base = question.replace('트렌드', '').replace('동향', '').strip()
            queries.append(f"{base} 최신 동향")
            queries.append(f"{base} 시장 전망")
        
        if '비교' in question or 'vs' in question.lower():
            queries.append(question.replace('비교', '차이점'))
        
        # 중복 제거 및 길이 제한
        unique_queries = []
        for query in queries:
            if query not in unique_queries and len(query.strip()) > 3:
                unique_queries.append(query.strip())
        
        result = unique_queries[:self.max_queries]
        logger.info(f"폴백 쿼리 생성 완료: {len(result)}개")
        
        return result
    
    def _post_process_queries(self, queries: List[str], original_question: str) -> List[str]:
        """
        생성된 쿼리들을 후처리합니다.
        
        검증, 중복 제거, 품질 개선 등을 수행하여 최종 쿼리 리스트를 생성합니다.
        
        Args:
            queries: 원본 쿼리 리스트
            original_question: 사용자의 원본 질문
            
        Returns:
            List[str]: 후처리된 최종 쿼리 리스트
        """
        if not queries:
            logger.warning("생성된 쿼리가 없음, 원본 질문 사용")
            return [original_question]
        
        logger.debug(f"쿼리 후처리 시작: {len(queries)}개 → 검증 및 정제")
        
        # 1단계: 검증 및 중복 제거
        valid_queries, removed_count = validate_query_list(queries)
        
        if removed_count > 0:
            logger.debug(f"검증 단계에서 {removed_count}개 쿼리 제거됨")
        
        # 2단계: 원본 질문과의 관련성 확인
        filtered_queries = self._filter_by_relevance(valid_queries, original_question)
        
        # 3단계: 길이 및 품질 조정
        optimized_queries = self._optimize_query_quality(filtered_queries)
        
        # 4단계: 최대 개수 제한
        final_queries = optimized_queries[:self.max_queries]
        
        # 최소 1개는 보장 (원본 질문 사용)
        if not final_queries:
            final_queries = [original_question]
            logger.warning("후처리 결과 쿼리가 없어 원본 질문 사용")
        
        logger.debug(f"쿼리 후처리 완료: {len(final_queries)}개 최종 쿼리")
        return final_queries
    
    def _filter_by_relevance(self, queries: List[str], original_question: str) -> List[str]:
        """
        원본 질문과의 관련성을 기준으로 쿼리를 필터링합니다.
        
        Args:
            queries: 필터링할 쿼리 리스트
            original_question: 원본 질문
            
        Returns:
            List[str]: 관련성이 높은 쿼리들
        """
        if not queries:
            return []
        
        try:
            from utils.text_processing import extract_keywords, calculate_text_similarity
            
            # 원본 질문의 키워드 추출
            original_keywords = set(extract_keywords(original_question.lower(), max_keywords=10))
            
            relevant_queries = []
            for query in queries:
                # 키워드 기반 관련성 확인
                query_keywords = set(extract_keywords(query.lower(), max_keywords=10))
                
                # 공통 키워드가 있거나 텍스트 유사도가 높은 경우 유지
                common_keywords = original_keywords & query_keywords
                similarity = calculate_text_similarity(original_question, query)
                
                if len(common_keywords) > 0 or similarity > 0.2:
                    relevant_queries.append(query)
                else:
                    logger.debug(f"관련성 낮은 쿼리 제거: '{query}'")
            
            return relevant_queries if relevant_queries else queries  # 폴백
            
        except Exception as e:
            logger.warning(f"관련성 필터링 실패, 원본 쿼리 유지: {e}")
            return queries
    
    def _optimize_query_quality(self, queries: List[str]) -> List[str]:
        """
        쿼리의 품질을 최적화합니다.
        
        검색에 더 효과적인 형태로 쿼리를 조정합니다.
        
        Args:
            queries: 최적화할 쿼리 리스트
            
        Returns:
            List[str]: 최적화된 쿼리 리스트
        """
        optimized = []
        
        for query in queries:
            # 기본 정제
            cleaned = query.strip()
            
            # 너무 긴 쿼리 단축
            if len(cleaned) > 100:
                # 문장 단위로 자르기
                sentences = cleaned.split('.')
                cleaned = sentences[0].strip()
                if len(cleaned) > 100:
                    cleaned = cleaned[:100].strip()
            
            # 검색에 불필요한 표현 제거
            remove_patterns = [
                r'알려주세요?\??',
                r'궁금합니다?\??',
                r'어떻게 되나요?\??',
                r'설명해주세요?\??',
                r'무엇인가요?\??',
                r'찾아주세요?\??',
            ]
            
            for pattern in remove_patterns:
                import re
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # 검색 효율을 위한 키워드 중심으로 조정
            cleaned = cleaned.strip().rstrip('.,!?')
            
            if len(cleaned) > 5:
                optimized.append(cleaned)
        
        return optimized
    
    def _update_statistics(self, queries_generated: int, processing_time: float):
        """성능 통계 업데이트"""
        self.total_requests += 1
        self.total_queries_generated += queries_generated
        
        # 이동 평균 계산
        if self.total_requests == 1:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = (
                (self.avg_processing_time * (self.total_requests - 1) + processing_time) 
                / self.total_requests
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        에이전트 성능 통계 반환
        
        Returns:
            Dict[str, Any]: 성능 통계 정보
        """
        return {
            "total_requests": self.total_requests,
            "total_queries_generated": self.total_queries_generated,
            "avg_queries_per_request": (
                self.total_queries_generated / max(self.total_requests, 1)
            ),
            "avg_processing_time": round(self.avg_processing_time, 3),
            "max_queries": self.max_queries,
            "keyword_extraction_enabled": self.enable_keyword_extraction
        }
    
    def reset_statistics(self):
        """통계 초기화"""
        self.total_requests = 0
        self.total_queries_generated = 0
        self.avg_processing_time = 0.0
        logger.info("질문 분석 에이전트 통계 초기화됨")


# 전역 인스턴스 (싱글톤 패턴)
_question_analyzer_instance = None

def get_question_analyzer() -> QuestionAnalyzer:
    """
    전역 질문 분석 에이전트 인스턴스 반환
    
    Returns:
        QuestionAnalyzer: 질문 분석 에이전트 인스턴스
    """
    global _question_analyzer_instance
    
    if _question_analyzer_instance is None:
        _question_analyzer_instance = QuestionAnalyzer()
        logger.info("질문 분석 에이전트 전역 인스턴스 생성됨")
    
    return _question_analyzer_instance

def reset_question_analyzer():
    """전역 인스턴스 리셋 (테스트용)"""
    global _question_analyzer_instance
    _question_analyzer_instance = None
    logger.info("질문 분석 에이전트 전역 인스턴스 리셋됨")


# LangGraph 노드 함수 (워크플로우에서 직접 사용)
async def analyze_question_node(state: ResearchState) -> ResearchState:
    """
    LangGraph 워크플로우용 질문 분석 노드 함수
    
    이 함수는 LangGraph에서 직접 호출되며, 
    전역 QuestionAnalyzer 인스턴스를 사용합니다.
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 질문 분석 결과가 추가된 상태
    """
    analyzer = get_question_analyzer()
    return await analyzer.process_state(state)


# 개발/테스트용 헬퍼 함수들
async def test_question_analyzer(test_question: str = None):
    """
    질문 분석 에이전트 테스트 함수
    
    Args:
        test_question: 테스트할 질문 (기본값: 샘플 질문)
    """
    if test_question is None:
        test_question = "오픈소스 LLM 트렌드와 상업용 모델과의 성능 차이점 알려줘"
    
    print(f"🧪 질문 분석 에이전트 테스트")
    print(f"📝 테스트 질문: {test_question}")
    print("-" * 50)
    
    try:
        analyzer = QuestionAnalyzer()
        
        start_time = time.time()
        queries = await analyzer.analyze_question(test_question)
        end_time = time.time()
        
        print(f"✅ 성공! {len(queries)}개 쿼리 생성 ({end_time - start_time:.2f}초)")
        print("\n📋 생성된 검색 쿼리들:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        
        print(f"\n📊 에이전트 통계:")
        stats = analyzer.get_statistics()
        for key, value in stats.items():
            print(f"  - {key}: {value}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        raise





if __name__ == "__main__":
    # 직접 실행 시 테스트 수행
    asyncio.run(test_question_analyzer())