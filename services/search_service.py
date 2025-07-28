# services/search_service.py
"""
검색 API 통합 서비스

DuckDuckGo와 Tavily 검색 API를 통합하여 관리하는 서비스 클래스.
병렬 검색 실행, 결과 정규화, 오류 처리 등의 기능을 제공합니다.

주요 기능:
- DuckDuckGo 무료 검색 (duckduckgo-search 라이브러리)
- Tavily AI 검색 (유료 API)
- 병렬 검색 실행으로 성능 최적화
- 검색 결과 표준화 및 정규화
- 오류 처리 및 재시도 로직
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 검색 라이브러리들
from duckduckgo_search import DDGS
from tavily import TavilyClient

# 프로젝트 모듈들
from models.data_models import Document, SearchQuery, SearchEngine
from config.settings import settings

logger = logging.getLogger(__name__)

class SearchService:
    """
    통합 검색 서비스 클래스
    
    여러 검색 엔진을 통합하여 관리하고, 병렬 검색을 통해
    효율적으로 다양한 소스에서 정보를 수집합니다.
    
    지원하는 검색 엔진:
    - DuckDuckGo: 무료, 개인정보 보호 중심
    - Tavily: 유료, AI 최적화된 검색 결과
    
    사용 예시:
        service = SearchService()
        queries = [
            SearchQuery("AI 트렌드", SearchEngine.DUCKDUCKGO),
            SearchQuery("머신러닝 발전", SearchEngine.TAVILY)
        ]
        documents = await service.search_parallel(queries)
    """
    
    def __init__(self):
        """검색 서비스 초기화"""
        self.tavily_client = None
        self.ddgs_client = DDGS()
        
        # Tavily 클라이언트 초기화 (API 키가 있는 경우만)
        if settings.TAVILY_API_KEY:
            try:
                self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                logger.info("Tavily client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily client: {e}")
        else:
            logger.warning("Tavily API key not found. Only DuckDuckGo search available.")
    
    async def search_parallel(self, queries: List[SearchQuery]) -> List[Document]:
        """
        여러 검색 쿼리를 병렬로 실행
        
        각 쿼리를 별도 스레드에서 동시에 실행하여 성능을 최적화.
        모든 검색이 완료된 후 결과를 통합하여 반환합니다.
        
        Args:
            queries: 실행할 검색 쿼리 리스트
            
        Returns:
            List[Document]: 모든 검색 결과를 통합한 문서 리스트
        """
        if not queries:
            return []
        
        start_time = time.time()
        all_documents = []
        
        # ThreadPoolExecutor를 사용한 병렬 검색
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            # 각 쿼리를 별도 스레드에서 실행
            future_to_query = {
                executor.submit(self._search_single, query): query 
                for query in queries
            }
            
            # 완료된 검색 결과들을 수집 (타임아웃 단축)
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    documents = future.result(timeout=10)  # 30초 → 10초로 단축
                    all_documents.extend(documents)
                    logger.info(f"Search completed for '{query.query}' via {query.engine.value}: {len(documents)} documents")
                except Exception as e:
                    logger.error(f"Search failed for '{query.query}' via {query.engine.value}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel search completed: {len(all_documents)} total documents in {elapsed_time:.2f}s")
        
        return all_documents
    
    async def search_all_engines(self, query: str, max_results: int = 3) -> List[Document]:
        """
        DuckDuckGo + Tavily를 사용하여 단일 쿼리에 대한 병렬 검색 수행 (최적화됨)

        Args:
            query (str): 검색어
            max_results (int): 엔진별 최대 결과 수 (기본값 3으로 축소)

        Returns:
            List[Document]: 통합된 검색 결과 문서 리스트
        """
        search_queries = []

        # 빠른 검색을 위해 DuckDuckGo만 사용 (Tavily는 느림)
        search_queries.append(SearchQuery(
            query=query,
            engine=SearchEngine.DUCKDUCKGO,
            max_results=max_results,
            language="ko"
        ))

        # 병렬 검색 실행
        results = await self.search_parallel(search_queries)
        return results


    
    def _search_single(self, query: SearchQuery) -> List[Document]:
        """
        단일 검색 쿼리 실행
        
        검색 엔진 타입에 따라 적절한 검색 메서드를 호출하고,
        결과를 표준화된 Document 객체로 변환합니다.
        
        Args:
            query: 실행할 검색 쿼리
            
        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        try:
            if query.engine == SearchEngine.DUCKDUCKGO:
                return self._search_duckduckgo(query)
            elif query.engine == SearchEngine.TAVILY:
                return self._search_tavily(query)
            else:
                logger.error(f"Unsupported search engine: {query.engine}")
                return []
        except Exception as e:
            logger.error(f"Search error for '{query.query}' via {query.engine.value}: {e}")
            return []
    
    def _search_duckduckgo(self, query: SearchQuery) -> List[Document]:
        """
        DuckDuckGo 검색 실행
        
        duckduckgo-search 라이브러리를 사용하여 무료 검색을 수행.
        개인정보 보호가 중요하고, 일반적인 웹 검색 결과가 필요한 경우 사용.
        
        Args:
            query: DuckDuckGo 검색 쿼리
            
        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        try:
            # DuckDuckGo 검색 실행 (최적화)
            results = self.ddgs_client.text(
                keywords=query.query,
                max_results=min(query.max_results, 3),  # 최대 3개로 제한
                region=query.language,
                safesearch='moderate',
                timelimit=None  # 시간 제한 제거로 속도 향상
            )
            
            documents = []
            for result in results:
                # DuckDuckGo 결과를 표준 형식으로 변환
                normalized_result = self._normalize_duckduckgo_result(result)
                if normalized_result:
                    doc = Document.from_search_result(normalized_result, SearchEngine.DUCKDUCKGO)
                    documents.append(doc)
            
            logger.debug(f"DuckDuckGo search for '{query.query}': {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed for '{query.query}': {e}")
            return []
    
    def _search_tavily(self, query: SearchQuery) -> List[Document]:
        """
        Tavily AI 검색 실행
        
        Tavily API를 사용하여 AI 최적화된 검색을 수행.
        더 정확하고 관련성 높은 결과를 제공하지만 유료 서비스.
        
        Args:
            query: Tavily 검색 쿼리
            
        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        if not self.tavily_client:
            logger.warning("Tavily client not available. Skipping Tavily search.")
            return []
        
        try:
            # Tavily 검색 실행
            response = self.tavily_client.search(
                query=query.query,
                search_depth="advanced",  # 심층 검색
                max_results=query.max_results,
                include_answer=False,    # 답변 생성은 비활성화 (원본 문서만 수집)
                include_raw_content=True, # 전체 내용 포함
                include_domains=None,    # 모든 도메인 허용
                exclude_domains=['social_media']  # 소셜미디어 제외
            )
            
            documents = []
            for result in response.get('results', []):
                # Tavily 결과를 표준 형식으로 변환
                normalized_result = self._normalize_tavily_result(result)
                if normalized_result:
                    doc = Document.from_search_result(normalized_result, SearchEngine.TAVILY)
                    documents.append(doc)
            
            logger.debug(f"Tavily search for '{query.query}': {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Tavily search failed for '{query.query}': {e}")
            return []
    
    def _normalize_duckduckgo_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        DuckDuckGo 검색 결과를 표준 형식으로 정규화
        
        DuckDuckGo의 원시 응답을 Document 클래스에서 사용할 수 있는
        표준 딕셔너리 형식으로 변환합니다.
        
        DuckDuckGo 응답 구조:
        {
            'title': '제목',
            'href': 'URL', 
            'body': '내용 요약'
        }
        
        Args:
            result: DuckDuckGo 원시 검색 결과
            
        Returns:
            Dict[str, Any]: 표준화된 검색 결과 또는 None
        """
        try:
            # 필수 필드 확인
            if not all(key in result for key in ['title', 'href', 'body']):
                return None
            
            # 빈 내용 필터링
            if not result['title'].strip() or not result['body'].strip():
                return None
            
            return {
                'title': result['title'].strip(),
                'url': result['href'].strip(),
                'content': result['body'].strip(),
                'snippet': result['body'].strip()[:200] + "..." if len(result['body']) > 200 else result['body'].strip(),
                'published_date': None  # DuckDuckGo는 날짜 정보 제공하지 않음
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize DuckDuckGo result: {e}")
            return None
    
    def _normalize_tavily_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Tavily 검색 결과를 표준 형식으로 정규화
        
        Tavily의 원시 응답을 Document 클래스에서 사용할 수 있는
        표준 딕셔너리 형식으로 변환합니다.
        
        Tavily 응답 구조:
        {
            'title': '제목',
            'url': 'URL',
            'content': '전체 내용',
            'score': 관련성_점수,
            'published_date': '날짜'
        }
        
        Args:
            result: Tavily 원시 검색 결과
            
        Returns:
            Dict[str, Any]: 표준화된 검색 결과 또는 None
        """
        try:
            # 필수 필드 확인
            if not all(key in result for key in ['title', 'url', 'content']):
                return None
            
            # 빈 내용 필터링
            if not result['title'].strip() or not result['content'].strip():
                return None
            
            return {
                'title': result['title'].strip(),
                'url': result['url'].strip(),
                'content': result['content'].strip(),
                'snippet': result['content'].strip()[:200] + "..." if len(result['content']) > 200 else result['content'].strip(),
                'published_date': result.get('published_date'),
                'relevance_score': result.get('score', 0.0)  # Tavily는 관련성 점수 제공
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize Tavily result: {e}")
            return None
    
    def create_search_queries(self, sub_queries: List[str]) -> List[SearchQuery]:
        """
        하위 질문들을 검색 쿼리 객체로 변환
        
        질문 분석 에이전트에서 생성된 하위 질문들을 실제 검색 쿼리로 변환.
        각 쿼리를 여러 검색 엔진에 분산하여 다양한 소스에서 정보 수집.
        
        분배 전략:
        - Tavily 사용 가능시: 절반은 Tavily, 절반은 DuckDuckGo
        - Tavily 불가시: 모두 DuckDuckGo
        
        Args:
            sub_queries: 질문 분석 에이전트에서 생성된 하위 질문 리스트
            
        Returns:
            List[SearchQuery]: 실행 준비된 검색 쿼리 리스트
        """
        if not sub_queries:
            return []
        
        queries = []
        
        for i, query_text in enumerate(sub_queries):
            # 검색 엔진 선택 (교대로 사용)
            if self.tavily_client and i % 2 == 0:
                engine = SearchEngine.TAVILY
                max_results = min(settings.MAX_SEARCH_RESULTS // 2, 10)
            else:
                engine = SearchEngine.DUCKDUCKGO  
                max_results = min(settings.MAX_SEARCH_RESULTS // 2, settings.DUCKDUCKGO_MAX_RESULTS)
            
            search_query = SearchQuery(
                query=query_text.strip(),
                engine=engine,
                max_results=max_results,
                language="ko"  # 한국어 우선
            )
            queries.append(search_query)
        
        logger.info(f"Created {len(queries)} search queries from {len(sub_queries)} sub-queries")
        return queries
    
    def get_available_engines(self) -> List[SearchEngine]:
        """
        사용 가능한 검색 엔진 목록 반환
        
        현재 설정과 API 키 상태에 따라 실제로 사용할 수 있는
        검색 엔진들의 목록을 반환합니다.
        
        Returns:
            List[SearchEngine]: 사용 가능한 검색 엔진 리스트
        """
        available = [SearchEngine.DUCKDUCKGO]  # DuckDuckGo는 항상 사용 가능
        
        if self.tavily_client:
            available.append(SearchEngine.TAVILY)
        
        return available
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        검색 서비스 상태 정보 반환
        
        현재 검색 서비스의 상태와 설정 정보를 반환.
        디버깅이나 모니터링 목적으로 사용됩니다.
        
        Returns:
            Dict[str, Any]: 서비스 상태 정보
        """
        return {
            "available_engines": [engine.value for engine in self.get_available_engines()],
            "tavily_available": self.tavily_client is not None,
            "max_search_results": settings.MAX_SEARCH_RESULTS,
            "max_workers": settings.MAX_WORKERS,
            "request_timeout": settings.REQUEST_TIMEOUT,
            "duckduckgo_max_results": settings.DUCKDUCKGO_MAX_RESULTS
        }


# 전역 검색 서비스 인스턴스 (싱글톤 패턴)
_search_service_instance = None

def get_search_service() -> SearchService:
    """
    전역 검색 서비스 인스턴스 반환
    
    애플리케이션 전체에서 하나의 검색 서비스 인스턴스를 공유.
    초기화 비용을 절약하고 일관된 설정을 보장합니다.
    
    Returns:
        SearchService: 검색 서비스 인스턴스
    """
    global _search_service_instance
    
    if _search_service_instance is None:
        _search_service_instance = SearchService()
        logger.info("Search service instance created")
    
    return _search_service_instance

def reset_search_service():
    """
    검색 서비스 인스턴스 리셋 (테스트용)
    
    테스트나 설정 변경 시 검색 서비스를 다시 초기화할 때 사용.
    """
    global _search_service_instance
    _search_service_instance = None
    logger.info("Search service instance reset")