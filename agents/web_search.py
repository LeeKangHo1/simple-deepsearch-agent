# agents/web_search.py
"""
웹 검색 에이전트 (재시도 로직 포함)

여러 검색 엔진을 병렬로 활용하여 하위 쿼리들에 대한 문서를 수집하고,
벡터 DB를 통한 중복 제거 및 관련성 정렬을 수행하는 에이전트입니다.

주요 기능:
- DuckDuckGo + Tavily 병렬 검색 실행
- 검증 실패 시 다른 관점의 쿼리로 재검색
- 기존 결과와 새 결과 병합
- 벡터 DB 기반 중복 제거 (URL + 내용 유사도)
- 사용자 질문과의 의미적 유사도 기반 정렬
- 검색 결과 품질 검증 및 필터링
- 진행 상태 실시간 추적
"""

import logging
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import hashlib

# 프로젝트 모듈들
from models.state import ResearchState, StateManager
from models.data_models import Document, SearchQuery, SearchResult, remove_duplicate_documents
from services.search_service import get_search_service, SearchService
from services.vector_db import get_vector_db_service, VectorDBService
from services.llm_service import get_llm_service
from utils.validators import validate_search_query, validate_document, sanitize_input, validate_query_list
from utils.logger import get_agent_logger, log_agent_start, log_agent_end
from utils.text_processing import extract_keywords, calculate_text_similarity

logger = get_agent_logger("web_search")

class WebSearchAgent:
    """
    웹 검색 에이전트 클래스 (재시도 로직 포함)
    
    여러 검색 엔진을 활용하여 효율적이고 포괄적인 문서 수집을 담당합니다.
    벡터 DB를 통한 지능적 중복 제거와 관련성 기반 정렬을 제공하며,
    검증 실패 시 다른 관점의 쿼리로 재검색하는 기능을 포함합니다.
    """
    
    def __init__(self, 
                 max_results_per_query: int = 3,  # 5 → 3으로 축소
                 max_total_results: int = 10,     # 15 → 10으로 축소
                 similarity_threshold: float = 0.85,
                 enable_content_similarity: bool = False):  # 중복 제거 비활성화로 속도 향상
        """
        웹 검색 에이전트 초기화
        
        Args:
            max_results_per_query: 쿼리당 최대 검색 결과 수
            max_total_results: 전체 최대 결과 수
            similarity_threshold: 중복 판정 임계값 (0.0-1.0)
            enable_content_similarity: 내용 유사도 검사 활성화
        """
        self.search_service = get_search_service()
        self.vector_db = get_vector_db_service()
        
        self.max_results_per_query = max_results_per_query
        self.max_total_results = max_total_results
        self.similarity_threshold = similarity_threshold
        self.enable_content_similarity = enable_content_similarity
        
        # 성능 통계
        self.total_searches = 0
        self.total_documents_found = 0
        self.total_documents_after_dedup = 0
        self.avg_search_time = 0.0
        self.search_engine_stats = defaultdict(int)
        self.retry_stats = {"total_retries": 0, "successful_retries": 0}
        
        logger.info(f"웹 검색 에이전트 초기화 완료 "
                   f"(쿼리당 최대 {max_results_per_query}개, "
                   f"전체 최대 {max_total_results}개, "
                   f"유사도 임계값 {similarity_threshold})")
    
    async def process_state(self, state: ResearchState) -> ResearchState:
        """
        LangGraph 워크플로우용 상태 처리 메서드 (재시도 로직 포함)
        
        Args:
            state: 현재 연구 상태
            
        Returns:
            ResearchState: 검색 결과가 추가된 상태
        """
        log_agent_start("web_search", {"retry_count": state.get("retry_count", 0)})
        
        # 재시도 여부 확인
        retry_count = state.get("retry_count", 0)
        is_retry = retry_count > 0
        
        # 진행 상태 업데이트
        if is_retry:
            new_state = StateManager.set_step(
                state, 
                "searching", 
                f"검증 피드백을 바탕으로 추가 검색을 수행 중... (재시도 {retry_count}회)"
            )
        else:
            new_state = StateManager.set_step(
                state, 
                "searching", 
                "웹 검색을 통해 관련 문서를 수집 중..."
            )
        
        try:
            if is_retry:
                # 재시도: 새로운 관점의 쿼리로 추가 검색
                documents = await self._handle_retry_search(state)
            else:
                # 초기 검색: 기존 로직
                documents = await self._handle_initial_search(state)
            
            # 상태에 결과 저장 (빈 리스트도 허용)
            new_state = new_state.copy()
            if documents:
                new_state["documents"] = [doc.to_dict() for doc in documents]
            else:
                new_state["documents"] = []
            
            # 성공 로그 추가
            if is_retry:
                new_state = StateManager.add_log(
                    new_state, 
                    f"✅ 추가 검색 완료: 총 {len(documents)}개 문서 (기존 + 새 검색 결과)"
                )
                self.retry_stats["successful_retries"] += 1
            else:
                new_state = StateManager.add_log(
                    new_state, 
                    f"✅ 웹 검색 완료: {len(documents)}개 관련 문서 수집"
                )
            
            # 검색 엔진별 통계 로그 추가
            stats_msg = self._get_search_stats_message()
            new_state = StateManager.add_log(new_state, stats_msg)
            
            log_agent_end("web_search", success=True, 
                         output_data={"document_count": len(documents), "is_retry": is_retry})
            
            logger.info(f"상태 처리 완료: {len(documents)}개 문서를 상태에 저장 (재시도: {is_retry})")
            return new_state
            
        except Exception as e:
            # 오류 상태 설정
            error_msg = f"웹 검색 실패 ({'재시도' if is_retry else '초기검색'}): {e}"
            error_state = StateManager.set_error(new_state, error_msg)
            
            log_agent_end("web_search", success=False, error=str(e))
            logger.error(f"상태 처리 실패: {e}")
            return error_state
    
    async def _handle_initial_search(self, state: ResearchState) -> List[Document]:
        """
        초기 검색 처리
        
        Args:
            state: 연구 상태
            
        Returns:
            List[Document]: 검색된 문서 리스트
        """
        # 하위 쿼리 확인
        sub_queries = state.get("sub_queries", [])
        if not sub_queries:
            logger.warning("검색할 하위 쿼리가 없습니다. 빈 결과를 반환합니다.")
            return []
        
        user_input = state.get("user_input", "")
        
        logger.info(f"초기 웹 검색 시작: {len(sub_queries)}개 쿼리")
        
        # 웹 검색 수행
        documents = await self.search_multiple_queries(sub_queries, user_input)
        
        return documents
    
    async def _handle_retry_search(self, state: ResearchState) -> List[Document]:
        """
        재시도 검색 처리 (새로운 관점의 쿼리 + 기존 결과 병합)
        
        Args:
            state: 연구 상태
            
        Returns:
            List[Document]: 기존 + 새 검색 결과가 병합된 문서 리스트
        """
        self.retry_stats["total_retries"] += 1
        
        # 기존 데이터 가져오기
        user_input = state.get("user_input", "")
        existing_queries = state.get("sub_queries", [])
        validation_feedback = state.get("validation_feedback", "")
        existing_documents_data = state.get("documents", [])
        
        logger.info(f"재시도 검색 시작: {len(existing_documents_data)}개 기존 문서")
        
        # 기존 문서를 Document 객체로 변환
        existing_documents = []
        for doc_data in existing_documents_data:
            try:
                doc = Document(
                    title=doc_data.get("title", ""),
                    url=doc_data.get("url", ""),
                    content=doc_data.get("content", ""),
                    source=doc_data.get("source", "unknown"),
                    relevance_score=doc_data.get("relevance_score", 0.0)
                )
                existing_documents.append(doc)
            except Exception as e:
                logger.warning(f"기존 문서 변환 실패: {e}")
                continue
        
        # 1단계: LLM으로 새로운 관점의 쿼리 생성
        new_queries = await self._generate_retry_queries(
            user_input, existing_queries, validation_feedback
        )
        
        if not new_queries:
            logger.warning("새로운 쿼리 생성 실패, 기존 문서 반환")
            return existing_documents
        
        logger.info(f"새로운 관점의 쿼리 {len(new_queries)}개 생성: {new_queries}")
        
        # 2단계: 새 쿼리로 검색 수행
        new_documents = await self.search_multiple_queries(new_queries, user_input)
        
        logger.info(f"새 검색 결과: {len(new_documents)}개 문서")
        
        # 3단계: 기존 + 새 문서 병합 및 중복 제거
        all_documents = existing_documents + new_documents
        merged_documents = remove_duplicate_documents(all_documents, threshold=self.similarity_threshold)
        
        logger.info(f"문서 병합 완료: {len(existing_documents)} + {len(new_documents)} → {len(merged_documents)}개")
        
        # 4단계: 상태에 새 쿼리도 추가 (추적용)
        # Note: 이 부분은 상위에서 처리하거나 별도 필드로 관리 필요
        
        return merged_documents
    
    async def _generate_retry_queries(self, 
                                    user_question: str, 
                                    existing_queries: List[str], 
                                    validation_feedback: str) -> List[str]:
        """
        검증 피드백을 바탕으로 새로운 관점의 검색 쿼리 생성
        
        Args:
            user_question: 사용자 원본 질문
            existing_queries: 기존 검색 쿼리들
            validation_feedback: 검증 에이전트 피드백
            
        Returns:
            List[str]: 새로운 관점의 검색 쿼리 리스트
        """
        try:
            logger.debug("LLM을 통한 재시도 쿼리 생성 시작")
            
            llm_service = get_llm_service()
            
            # LLM으로 새로운 관점의 쿼리 생성
            response = await llm_service.generate_retry_queries(
                user_question=user_question,
                existing_queries=existing_queries,
                validation_feedback=validation_feedback
            )
            
            if not response.success:
                logger.error(f"재시도 쿼리 생성 실패: {response.error_message}")
                return []
            
            # 응답 파싱
            try:
                new_queries = json.loads(response.content)
                if not isinstance(new_queries, list):
                    logger.error("재시도 쿼리 응답이 리스트가 아님")
                    return []
                
                # 쿼리 검증 및 정제
                validated_queries, removed_count = validate_query_list(new_queries)
                
                if removed_count > 0:
                    logger.debug(f"재시도 쿼리 검증: {removed_count}개 제거됨")
                
                logger.info(f"재시도 쿼리 생성 완료: {len(validated_queries)}개")
                return validated_queries
                
            except json.JSONDecodeError as e:
                logger.error(f"재시도 쿼리 JSON 파싱 실패: {e}")
                return []
                
        except Exception as e:
            logger.error(f"재시도 쿼리 생성 중 오류: {e}")
            return []
    
    async def search_multiple_queries(self, 
                                    queries: List[str], 
                                    original_question: str = None) -> List[Document]:
        """
        여러 쿼리에 대해 병렬 검색 수행 후 중복 제거 및 정렬
        
        이 메서드는 단독으로 호출 가능한 공개 인터페이스입니다.
        
        Args:
            queries: 검색할 쿼리 리스트
            original_question: 원본 질문 (관련성 정렬용)
            
        Returns:
            List[Document]: 중복 제거되고 정렬된 문서 리스트
            
        Raises:
            ValueError: 쿼리가 유효하지 않은 경우
            Exception: 검색 실패 등 기타 오류
        """
        start_time = time.time()
        
        try:
            # 1단계: 쿼리 검증 및 전처리
            validated_queries = self._validate_and_prepare_queries(queries)
            if not validated_queries:
                raise ValueError("유효한 검색 쿼리가 없습니다.")
            
            logger.info(f"병렬 검색 시작: {len(validated_queries)}개 쿼리")
            
            # 2단계: 병렬 검색 실행
            raw_results = await self._execute_parallel_search(validated_queries)
            logger.debug(f"원본 검색 결과: {len(raw_results)}개 문서")
            
            # 3단계: 문서 검증 및 기본 필터링
            valid_documents = self._validate_and_filter_documents(raw_results)
            logger.debug(f"검증된 문서: {len(valid_documents)}개")
            
            # 4단계: 간단한 중복 제거 (URL 기반만)
            deduplicated_docs = self._simple_deduplication(valid_documents)
            logger.debug(f"중복 제거 후: {len(deduplicated_docs)}개")
            
            # 5단계: 관련성 정렬 생략 (속도 향상)
            sorted_docs = deduplicated_docs
            
            # 6단계: 최대 개수 제한
            final_results = sorted_docs[:self.max_total_results]
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_statistics(len(queries), len(raw_results), len(final_results), processing_time)
            
            logger.info(f"웹 검색 완료: {len(final_results)}개 최종 문서 ({processing_time:.2f}초)")
            return final_results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_statistics(len(queries) if queries else 0, 0, 0, processing_time)
            
            logger.error(f"웹 검색 실패 ({processing_time:.2f}초): {e}")
            raise
    
    def _validate_and_prepare_queries(self, queries: List[str]) -> List[str]:
        """
        검색 쿼리들을 검증하고 전처리합니다.
        
        Args:
            queries: 원본 쿼리 리스트
            
        Returns:
            List[str]: 검증되고 전처리된 쿼리 리스트
        """
        if not queries:
            return []
        
        validated_queries = []
        
        for query in queries:
            if not query or not isinstance(query, str):
                logger.warning(f"유효하지 않은 쿼리 스킵: {query}")
                continue
            
            # 입력 정제
            cleaned_query = sanitize_input(query.strip(), max_length=200)
            
            # 쿼리 검증
            is_valid, error_msg = validate_search_query(cleaned_query)
            if is_valid:
                validated_queries.append(cleaned_query)
            else:
                logger.warning(f"검증 실패 쿼리 스킵: '{cleaned_query}' - {error_msg}")
        
        logger.debug(f"쿼리 검증 완료: {len(queries)}개 → {len(validated_queries)}개")
        return validated_queries
    
    async def _execute_parallel_search(self, queries: List[str]) -> List[Document]:
        """
        여러 쿼리에 대해 병렬 검색을 실행합니다.
        
        Args:
            queries: 검색할 쿼리 리스트
            
        Returns:
            List[Document]: 모든 검색 결과를 통합한 문서 리스트
        """
        all_documents = []
        
        # 쿼리별 병렬 검색 태스크 생성
        search_tasks = []
        for query in queries:
            task = self._search_single_query(query)
            search_tasks.append(task)
        
        logger.debug(f"병렬 검색 태스크 {len(search_tasks)}개 실행 중...")
        
        # 병렬 실행
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # 결과 통합
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"쿼리 '{queries[i]}' 검색 실패: {result}")
                    continue
                
                if isinstance(result, list):
                    all_documents.extend(result)
                    logger.debug(f"쿼리 '{queries[i]}': {len(result)}개 문서")
        
        except Exception as e:
            logger.error(f"병렬 검색 실행 중 오류: {e}")
            # 개별 쿼리 순차 실행으로 폴백
            all_documents = await self._fallback_sequential_search(queries)
        
        logger.info(f"병렬 검색 완료: 총 {len(all_documents)}개 문서 수집")
        return all_documents
    
    async def _search_single_query(self, query: str) -> List[Document]:
        """
        단일 쿼리에 대해 모든 검색 엔진에서 검색 수행
        
        Args:
            query: 검색할 쿼리
            
        Returns:
            List[Document]: 해당 쿼리의 검색 결과 문서들
        """
        query_start_time = time.time()
        documents = []
        
        try:
            # SearchService를 통한 통합 검색 (DuckDuckGo + Tavily)
            search_results = await self.search_service.search_all_engines(
                query=query,
                max_results=self.max_results_per_query
            )
            
            # SearchService에서 이미 Document 객체를 반환하므로 그대로 사용
            for document in search_results:
                try:
                    # 메타데이터에 쿼리 정보 추가
                    if not hasattr(document, 'metadata') or document.metadata is None:
                        document.metadata = {}
                    
                    document.metadata.update({
                        "query": query,
                        "content_length": len(document.content) if document.content else 0
                    })
                    
                    documents.append(document)
                    
                    # 검색 엔진별 통계 업데이트 (source는 SearchEngine enum)
                    engine_name = document.source.value if hasattr(document.source, 'value') else str(document.source)
                    self.search_engine_stats[engine_name] += 1
                    
                except Exception as e:
                    logger.warning(f"문서 변환 실패 (쿼리: {query}): {e}")
                    continue
            
            query_time = time.time() - query_start_time
            logger.debug(f"쿼리 '{query}' 검색 완료: {len(documents)}개 문서 ({query_time:.2f}초)")
            
        except Exception as e:
            logger.error(f"쿼리 '{query}' 검색 실패: {e}")
            # 빈 리스트 반환으로 전체 프로세스 중단 방지
        
        return documents
    
    async def _fallback_sequential_search(self, queries: List[str]) -> List[Document]:
        """
        병렬 검색 실패 시 순차 검색으로 폴백
        
        Args:
            queries: 검색할 쿼리 리스트
            
        Returns:
            List[Document]: 순차 검색 결과
        """
        logger.warning("병렬 검색 실패, 순차 검색으로 전환")
        
        all_documents = []
        
        for query in queries:
            try:
                docs = await self._search_single_query(query)
                all_documents.extend(docs)
                
                # 순차 실행이므로 약간의 지연 추가 (API 제한 고려)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"순차 검색에서 쿼리 '{query}' 실패: {e}")
                continue
        
        logger.info(f"순차 검색 완료: {len(all_documents)}개 문서")
        return all_documents
    
    def _validate_and_filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        수집된 문서들을 검증하고 기본 필터링을 수행합니다.
        
        Args:
            documents: 원본 문서 리스트
            
        Returns:
            List[Document]: 검증된 문서 리스트
        """
        if not documents:
            return []
        
        valid_documents = []
        filtered_count = 0
        
        seen_urls = set()  # URL 기반 기본 중복 제거
        
        for doc in documents:
            try:
                # 문서 유효성 검증
                is_valid, errors = validate_document(doc)
                if not is_valid:
                    filtered_count += 1
                    logger.debug(f"문서 검증 실패: {errors}")
                    continue
                
                # URL 기반 기본 중복 제거
                if doc.url in seen_urls:
                    filtered_count += 1
                    logger.debug(f"URL 중복 제거: {doc.url}")
                    continue
                
                # 내용 길이 기본 검증
                if not doc.content or len(doc.content.strip()) < 50:
                    filtered_count += 1
                    logger.debug(f"내용 부족으로 제거: {doc.title}")
                    continue
                
                # 내용 품질 기본 검증
                if self._is_low_quality_content(doc.content):
                    filtered_count += 1
                    logger.debug(f"저품질 내용으로 제거: {doc.title}")
                    continue
                
                seen_urls.add(doc.url)
                valid_documents.append(doc)
                
            except Exception as e:
                logger.warning(f"문서 검증 중 오류: {e}")
                filtered_count += 1
                continue
        
        logger.debug(f"문서 검증 완료: {len(documents)}개 → {len(valid_documents)}개 "
                    f"(필터링됨: {filtered_count}개)")
        
        return valid_documents
    
    def _simple_deduplication(self, documents: List[Document]) -> List[Document]:
        """
        간단한 URL 기반 중복 제거 (속도 최적화)
        
        Args:
            documents: 중복 제거할 문서 리스트
            
        Returns:
            List[Document]: 중복 제거된 문서 리스트
        """
        seen_urls = set()
        unique_docs = []
        
        for doc in documents:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _is_low_quality_content(self, content: str) -> bool:
        """
        저품질 내용인지 판단합니다.
        
        Args:
            content: 검사할 내용
            
        Returns:
            bool: 저품질 내용 여부
        """
        if not content:
            return True
        
        content_lower = content.lower()
        
        # 저품질 패턴들
        low_quality_patterns = [
            "페이지를 찾을 수 없",
            "access denied",
            "404 not found",
            "error occurred",
            "javascript를 활성화",
            "enable javascript",
            "쿠키를 허용",
            "allow cookies",
            "로그인이 필요",
            "login required",
            "구독이 필요",
            "subscription required"
        ]
        
        for pattern in low_quality_patterns:
            if pattern in content_lower:
                return True
        
        # 중복 문자/단어 비율 검사
        words = content.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # 중복 비율 70% 이상
                return True
        
        return False
    
    async def _remove_duplicates_with_vector_db(self, documents: List[Document]) -> List[Document]:
        """
        벡터 DB를 활용한 고급 중복 제거
        
        URL 중복 제거는 이미 완료된 상태에서 내용 유사도 기반 중복 제거 수행
        
        Args:
            documents: 중복 제거할 문서 리스트
            
        Returns:
            List[Document]: 중복 제거된 문서 리스트
        """
        if not documents or not self.enable_content_similarity:
            return documents
        
        if len(documents) <= 1:
            return documents
        
        try:
            logger.debug(f"벡터 DB 중복 제거 시작: {len(documents)}개 문서")
            
            # 단순한 중복 제거 사용 (벡터 DB 의존성 줄이기)
            deduplicated_docs = remove_duplicate_documents(documents, threshold=self.similarity_threshold)
            
            removed_count = len(documents) - len(deduplicated_docs)
            if removed_count > 0:
                logger.info(f"내용 유사도 기반 중복 제거: {removed_count}개 문서 제거")
            
            return deduplicated_docs
            
        except Exception as e:
            logger.warning(f"중복 제거 실패, 원본 반환: {e}")
            return documents
    
    async def _sort_by_relevance(self, documents: List[Document], original_question: str) -> List[Document]:
        """
        원본 질문과의 관련성을 기준으로 문서를 정렬합니다.
        
        Args:
            documents: 정렬할 문서 리스트
            original_question: 원본 질문
            
        Returns:
            List[Document]: 관련성 순으로 정렬된 문서 리스트
        """
        if not documents or not original_question:
            return documents
        
        try:
            logger.debug(f"관련성 기반 정렬 시작: {len(documents)}개 문서")
            
            # 키워드 기반 간단한 관련성 계산
            question_keywords = set(extract_keywords(original_question, max_keywords=10))
            
            for doc in documents:
                # 제목과 내용에서 키워드 추출
                title_keywords = set(extract_keywords(doc.title, max_keywords=5))
                content_keywords = set(extract_keywords(doc.content[:500], max_keywords=10))  # 첫 500자만
                doc_keywords = title_keywords | content_keywords
                
                # 키워드 일치도 계산 (제목 가중치 높게)
                title_overlap = len(question_keywords & title_keywords)
                content_overlap = len(question_keywords & content_keywords)
                
                # 관련성 점수 계산 (0.0 ~ 1.0)
                title_score = title_overlap / max(len(question_keywords), 1) * 0.7
                content_score = content_overlap / max(len(question_keywords), 1) * 0.3
                doc.relevance_score = title_score + content_score
            
            # 관련성 점수 순으로 정렬
            sorted_docs = sorted(documents, key=lambda x: x.relevance_score, reverse=True)
            
            logger.debug(f"관련성 정렬 완료: 상위 문서 점수 {sorted_docs[0].relevance_score:.3f}")
            return sorted_docs
            
        except Exception as e:
            logger.warning(f"관련성 정렬 실패, 원본 순서 유지: {e}")
            return documents
    
    def _update_statistics(self, query_count: int, raw_docs: int, final_docs: int, processing_time: float):
        """성능 통계 업데이트"""
        self.total_searches += 1
        self.total_documents_found += raw_docs
        self.total_documents_after_dedup += final_docs
        
        # 이동 평균 계산
        if self.total_searches == 1:
            self.avg_search_time = processing_time
        else:
            self.avg_search_time = (
                (self.avg_search_time * (self.total_searches - 1) + processing_time) 
                / self.total_searches
            )
    
    def _get_search_stats_message(self) -> str:
        """검색 통계 메시지 생성"""
        engine_stats = []
        for engine, count in self.search_engine_stats.items():
            engine_stats.append(f"{engine}: {count}개")
        
        retry_info = f"재시도: {self.retry_stats['total_retries']}회"
        
        return f"📊 검색 결과 - {', '.join(engine_stats)}, {retry_info}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        에이전트 성능 통계 반환
        
        Returns:
            Dict[str, Any]: 성능 통계 정보
        """
        return {
            "total_searches": self.total_searches,
            "total_documents_found": self.total_documents_found,
            "total_documents_after_dedup": self.total_documents_after_dedup,
            "deduplication_rate": (
                (self.total_documents_found - self.total_documents_after_dedup) 
                / max(self.total_documents_found, 1) * 100
            ),
            "avg_documents_per_search": (
                self.total_documents_after_dedup / max(self.total_searches, 1)
            ),
            "avg_search_time": round(self.avg_search_time, 3),
            "search_engine_stats": dict(self.search_engine_stats),
            "retry_stats": dict(self.retry_stats),
            "max_results_per_query": self.max_results_per_query,
            "max_total_results": self.max_total_results,
            "similarity_threshold": self.similarity_threshold,
            "content_similarity_enabled": self.enable_content_similarity
        }
    
    def reset_statistics(self):
        """통계 초기화"""
        self.total_searches = 0
        self.total_documents_found = 0
        self.total_documents_after_dedup = 0
        self.avg_search_time = 0.0
        self.search_engine_stats.clear()
        self.retry_stats = {"total_retries": 0, "successful_retries": 0}
        logger.info("웹 검색 에이전트 통계 초기화됨")


# LangGraph 노드 함수
async def web_search_node(state: ResearchState) -> ResearchState:
    """
    LangGraph 워크플로우용 웹 검색 노드 함수
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 웹 검색 결과가 추가된 상태
    """
    web_search_agent = get_web_search_agent()
    return await web_search_agent.process_state(state)


# 전역 인스턴스 (싱글톤 패턴)
_web_search_agent_instance = None

def get_web_search_agent() -> WebSearchAgent:
    """
    전역 웹 검색 에이전트 인스턴스 반환
    
    Returns:
        WebSearchAgent: 웹 검색 에이전트 인스턴스
    """
    global _web_search_agent_instance
    
    if _web_search_agent_instance is None:
        _web_search_agent_instance = WebSearchAgent()
        logger.info("웹 검색 에이전트 전역 인스턴스 생성됨")
    
    return _web_search_agent_instance

def reset_web_search_agent():
    """전역 인스턴스 리셋 (테스트용)"""
    global _web_search_agent_instance
    _web_search_agent_instance = None
    logger.info("웹 검색 에이전트 전역 인스턴스 리셋됨")