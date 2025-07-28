# models/data_models.py
"""
딥 리서치 챗봇에서 사용되는 데이터 모델 클래스들

이 모듈은 다음과 같은 주요 데이터 구조들을 정의합니다:
- Document: 검색으로 수집된 문서 정보
- SearchQuery: 검색 쿼리 정보
- DocumentSummary: 문서 요약 결과
- Insight: 인사이트 및 시사점
- ValidationResult: 검증 결과
- ResearchResponse: 최종 응답
- ProcessingStats: 처리 통계 및 성능 지표

모든 클래스는 @dataclass를 사용하여 타입 안정성을 보장하며,
자동 변환 메서드와 유틸리티 함수들을 제공합니다.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

class SearchEngine(Enum):
    """
    검색 엔진 타입 열거형
    
    현재 지원하는 검색 엔진:
    - DUCKDUCKGO: 개인정보 보호 중심의 검색 엔진 (무료)
    - TAVILY: AI 기반 검색 API (유료, 더 정확한 결과)
    """
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"

class DocumentType(Enum):
    """
    문서 타입 분류
    
    수집된 문서의 종류를 분류하여 처리 방식을 다르게 적용:
    - NEWS: 뉴스 기사 (최신성 중요)
    - BLOG: 블로그 포스트 (개인 의견 포함)
    - ACADEMIC: 학술 논문 (신뢰도 높음)
    - REPORT: 보고서 (공식 자료)
    - OTHER: 기타 문서
    """
    NEWS = "news"
    BLOG = "blog"
    ACADEMIC = "academic"
    REPORT = "report"
    OTHER = "other"
# models/data_models.py에 추가할 SearchResult 클래스

@dataclass
class SearchResult:
    """
    검색 엔진으로부터 받은 개별 검색 결과
    
    SearchService에서 반환되는 원시 검색 결과를 표현합니다.
    Document 클래스로 변환되기 전의 중간 데이터 구조입니다.
    """
    title: str                          # 검색 결과 제목
    url: str                           # 검색 결과 URL
    content: str                       # 검색 결과 내용/스니펫
    engine: str                        # 검색 엔진 이름 (duckduckgo, tavily)
    score: Optional[float] = None      # 검색 엔진에서 제공하는 관련성 점수
    published_date: Optional[str] = None  # 게시 날짜 (가능한 경우)
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not self.title or not self.url:
            raise ValueError("제목과 URL은 필수입니다.")
        
        if not self.content:
            self.content = ""
        
        # URL 정규화
        self.url = self.url.strip()
        if not self.url.startswith(('http://', 'https://')):
            if not self.url.startswith('//'):
                self.url = 'https://' + self.url
            else:
                self.url = 'https:' + self.url
    
    def to_document(self, query: str = None) -> 'Document':
        """
        SearchResult를 Document로 변환
        
        Args:
            query: 검색에 사용된 쿼리 (메타데이터에 포함)
            
        Returns:
            Document: 변환된 Document 객체
        """
        metadata = {
            "search_engine": self.engine,
            "relevance_score": self.score or 0.0,
            "content_length": len(self.content) if self.content else 0
        }
        
        if query:
            metadata["query"] = query
        
        if self.published_date:
            metadata["published_date"] = self.published_date
        
        return Document(
            title=self.title,
            url=self.url,
            content=self.content,
            source=self.engine,
            metadata=metadata
        )
    
@dataclass
class SearchQuery:
    """
    검색 쿼리 정보를 담는 데이터 클래스
    
    질문 분석 에이전트에서 생성된 하위 질문들을 실제 검색 쿼리로 변환할 때 사용.
    각 검색 엔진별로 최적화된 쿼리를 생성하고 관리합니다.
    
    사용 예시:
        query = SearchQuery(
            query="오픈소스 LLM 트렌드 2024",
            engine=SearchEngine.TAVILY,
            max_results=10
        )
    """
    query: str
    """검색할 쿼리 문자열 (질문 분석 에이전트에서 생성)"""
    
    engine: SearchEngine
    """사용할 검색 엔진 (DuckDuckGo 또는 Tavily)"""
    
    max_results: int = 10
    """최대 검색 결과 개수 (기본값: 10개)"""
    
    language: str = "ko"
    """검색 언어 코드 (ko: 한국어, en: 영어)"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """쿼리 생성 시간 (성능 추적용)"""

@dataclass
class Document:
    """
    검색으로 수집된 문서 정보를 담는 핵심 데이터 클래스
    
    웹 검색 에이전트에서 수집한 원시 데이터를 구조화하여 저장.
    중복 제거, 관련도 계산, 품질 평가 등의 기능을 포함합니다.
    
    주요 기능:
    - 자동 중복 감지 (content_hash 속성)
    - 질문과의 관련도 점수 계산
    - 벡터 임베딩 저장 (Chroma DB용)
    - 문서 품질 평가
    
    사용 예시:
        doc = Document(
            title="GPT-4o 성능 분석",
            url="https://example.com/article",
            content="GPT-4o는 최신 언어모델로...",
            source=SearchEngine.TAVILY
        )
    """
    title: str
    """문서 제목 (검색 결과에서 추출)"""
    
    url: str
    """문서 원본 URL (출처 표시용)"""
    
    content: str
    """문서 전체 내용 (요약 대상이 되는 텍스트)"""
    
    source: SearchEngine
    """수집한 검색 엔진 (DuckDuckGo 또는 Tavily)"""
    
    relevance_score: float = 0.0
    """질문과의 관련도 점수 (0.0 ~ 1.0, 높을수록 관련성 높음)"""
    
    doc_type: DocumentType = DocumentType.OTHER
    """문서 타입 분류 (뉴스, 블로그, 학술 등)"""
    
    embedding: Optional[List[float]] = None
    """문서 임베딩 벡터 (Chroma DB 저장용, 유사도 계산용)"""
    
    snippet: str = ""
    """문서 요약 스니펫 (검색 결과 미리보기)"""
    
    published_date: Optional[datetime] = None
    """문서 발행일 (최신성 평가용)"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """문서 수집 시간"""
    
    @property
    def content_hash(self) -> str:
        """
        내용 기반 해시값 생성 (중복 제거용)
        
        제목과 내용 일부를 조합하여 MD5 해시를 생성합니다.
        동일하거나 매우 유사한 문서를 자동으로 감지하여 중복 제거에 활용.
        
        Returns:
            str: 문서의 고유 해시값
        """
        content_for_hash = f"{self.title}|{self.content[:500]}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()
    
    @property
    def url_domain(self) -> str:
        """
        URL에서 도메인명 추출
        
        동일한 사이트에서 온 문서들을 그룹핑하거나
        신뢰도 평가에 활용할 수 있습니다.
        
        Returns:
            str: 도메인명 (예: "naver.com", "google.com")
        """
        try:
            from urllib.parse import urlparse
            return urlparse(self.url).netloc
        except:
            return "unknown"
    
    @classmethod
    def from_search_result(cls, result: Dict[str, Any], source: SearchEngine) -> 'Document':
        """
        검색 결과 딕셔너리로부터 Document 객체 생성
        
        Args:
            result: 정규화된 검색 결과 딕셔너리
            source: 검색 엔진 타입
            
        Returns:
            Document: 생성된 Document 객체
        """
        return cls(
            title=result.get('title', ''),
            url=result.get('url', ''),
            content=result.get('content', ''),
            source=source,
            snippet=result.get('snippet', result.get('content', '')[:200]),
            published_date=None  # 추후 파싱 로직 추가 가능
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Document 객체를 딕셔너리로 변환

        JSON 직렬화나 State 객체에 저장할 때 사용.
        모든 필드를 문자열 또는 기본 타입으로 변환합니다.

        Returns:
            Dict[str, Any]: 딕셔너리 형태의 문서 정보
        """
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source": self.source.value if isinstance(self.source, Enum) else str(self.source),
            "relevance_score": self.relevance_score,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, Enum) else str(self.doc_type),
            "snippet": self.snippet,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "created_at": self.created_at.isoformat(),
            "content_hash": self.content_hash,
            "url_domain": self.url_domain
        }

    
    @classmethod
    def from_search_result(cls, result: Dict[str, Any], source: SearchEngine) -> 'Document':
        """
        검색 결과 딕셔너리에서 Document 객체 생성
        
        DuckDuckGo나 Tavily 검색 결과를 표준화된 Document 객체로 변환.
        각 검색 엔진의 응답 형식이 달라도 동일한 구조로 처리할 수 있습니다.
        
        Args:
            result: 검색 엔진의 원시 결과 딕셔너리
            source: 검색 엔진 타입
            
        Returns:
            Document: 생성된 Document 객체
        """
        return cls(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            source=source,
            snippet=result.get("snippet", ""),
            published_date=cls._parse_date(result.get("published_date"))
        )
    
    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """
        다양한 형식의 날짜 문자열을 datetime 객체로 변환
        
        검색 엔진마다 다른 날짜 형식을 통일된 datetime으로 변환.
        변환 실패 시 None을 반환하여 오류를 방지합니다.
        
        Args:
            date_str: 날짜 문자열
            
        Returns:
            Optional[datetime]: 변환된 datetime 객체 또는 None
        """
        if not date_str:
            return None
        try:
            # 여러 날짜 형식 지원
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None

@dataclass
class DocumentSummary:
    """
    문서 요약 정보를 담는 데이터 클래스
    
    문서 요약 에이전트에서 각 Document를 분석한 결과를 저장.
    핵심 포인트 추출과 요약 품질 평가 기능을 포함합니다.
    
    사용 흐름:
    1. Document 객체를 문서 요약 에이전트에 전달
    2. LLM이 요약문과 핵심 포인트 생성
    3. DocumentSummary 객체로 결과 저장
    4. 인사이트 생성 에이전트에서 활용
    """
    document_hash: str
    """원본 문서의 해시값 (Document.content_hash와 연결)"""
    
    summary: str
    """요약 내용 (최대 300자 내외)"""
    
    key_points: List[str] = field(default_factory=list)
    """핵심 포인트들 (불릿 포인트 형태)"""
    
    confidence_score: float = 0.0
    """요약 신뢰도 점수 (0.0 ~ 1.0, LLM 응답 품질 평가)"""
    
    word_count: int = 0
    """요약 단어 수 (자동 계산됨)"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """요약 생성 시간"""
    
    def __post_init__(self):
        """요약 생성 후 단어 수 자동 계산"""
        self.word_count = len(self.summary.split()) if self.summary else 0

@dataclass
class Insight:
    """
    인사이트 및 시사점 정보를 담는 데이터 클래스
    
    인사이트 생성 에이전트에서 여러 문서 요약을 종합 분석한 결과.
    단순한 정보 나열이 아닌 깊은 통찰과 시사점을 제공합니다.
    
    인사이트 예시:
    - "기업들이 오픈소스 LLM을 채택하는 이유는 비용 절감보다 유연성 때문"
    - "상업용 모델과 오픈소스 모델의 성능 격차가 빠르게 줄어들고 있음"
    
    카테고리별 분류:
    - trend: 트렌드 분석
    - analysis: 심층 분석  
    - recommendation: 추천사항
    - prediction: 미래 전망
    """
    content: str
    """인사이트 내용 (핵심 통찰이나 시사점)"""
    
    category: str = "general"
    """인사이트 카테고리 (trend, analysis, recommendation, prediction 등)"""
    
    confidence_score: float = 0.0
    """인사이트 신뢰도 점수 (0.0 ~ 1.0, 뒷받침하는 근거의 강도)"""
    
    supporting_documents: List[str] = field(default_factory=list)
    """인사이트를 뒷받침하는 문서 해시값들 (출처 추적용)"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """인사이트 생성 시간"""

@dataclass
class ValidationResult:
    """
    검증 에이전트의 검증 결과를 담는 데이터 클래스
    
    최종 응답이 생성된 후, 논리적 오류나 부정확성을 자동으로 검증.
    검증 실패 시 구체적인 피드백과 개선 제안을 제공합니다.
    
    검증 항목:
    - 논리적 일관성 확인
    - 출처 누락 여부 체크
    - 인사이트와 문서 내용의 일치성
    - 응답 품질 평가
    
    사용 흐름:
    1. 응답 생성 에이전트에서 마크다운 응답 생성
    2. 검증 에이전트가 응답 품질 검증
    3. ValidationResult로 결과 반환
    4. 실패 시 재처리, 성공 시 최종 출력
    """
    is_valid: bool
    """검증 통과 여부 (True: 품질 좋음, False: 재처리 필요)"""
    
    feedback: str = ""
    """검증 피드백 메시지 (실패 시 구체적인 문제점 설명)"""
    
    issues: List[str] = field(default_factory=list)
    """발견된 문제점들 (논리 오류, 출처 누락 등)"""
    
    suggestions: List[str] = field(default_factory=list)
    """개선 제안사항들 (다음 재처리 시 참고용)"""
    
    confidence_score: float = 0.0
    """검증 신뢰도 점수 (0.0 ~ 1.0, 응답 품질 점수)"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """검증 수행 시간"""

@dataclass
class ResearchResponse:
    """최종 리서치 응답 정보"""
    markdown_content: str
    """마크다운 형식의 응답 내용"""
    
    sources: List[Dict[str, str]] = field(default_factory=list)
    """출처 정보 목록
    [{"title": "제목", "url": "URL", "domain": "도메인"}]
    """
    
    insights_count: int = 0
    """포함된 인사이트 개수"""
    
    documents_used: int = 0
    """사용된 문서 개수"""
    
    word_count: int = 0
    """응답 단어 수"""
    
    created_at: datetime = field(default_factory=datetime.now)
    """응답 생성 시간"""
    
    def __post_init__(self):
        """응답 생성 후 자동 계산"""
        self.word_count = len(self.markdown_content.split()) if self.markdown_content else 0

@dataclass
class ProcessingStats:
    """처리 통계 정보"""
    total_search_results: int = 0
    """총 검색 결과 수"""
    
    processed_documents: int = 0
    """처리된 문서 수"""
    
    duplicates_removed: int = 0
    """제거된 중복 문서 수"""
    
    total_processing_time: float = 0.0
    """총 처리 시간 (초)"""
    
    step_times: Dict[str, float] = field(default_factory=dict)
    """단계별 처리 시간"""
    
    retry_count: int = 0
    """재시도 횟수"""
    
    api_calls: Dict[str, int] = field(default_factory=dict)
    """API 호출 횟수"""
    
    def add_step_time(self, step: str, duration: float):
        """단계별 시간 추가"""
        self.step_times[step] = duration
        self.total_processing_time += duration
    
    def increment_api_call(self, api_name: str):
        """API 호출 횟수 증가"""
        self.api_calls[api_name] = self.api_calls.get(api_name, 0) + 1
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """효율성 지표 계산"""
        return {
            "documents_per_second": self.processed_documents / max(self.total_processing_time, 1),
            "duplicate_ratio": self.duplicates_removed / max(self.total_search_results, 1),
            "processing_success_rate": (self.processed_documents / max(self.total_search_results, 1)) * 100,
            "avg_step_time": sum(self.step_times.values()) / max(len(self.step_times), 1)
        }

# 유틸리티 함수들
def remove_duplicate_documents(documents: List[Document], threshold: float = 0.85) -> List[Document]:
    """
    중복 문서 제거 함수
    
    여러 검색 엔진에서 수집된 문서들 중 중복된 내용을 자동으로 제거.
    해시 기반과 제목 기반 두 가지 방식으로 중복을 감지합니다.
    
    중복 감지 방식:
    1. content_hash 기반: 제목 + 내용 일부의 MD5 해시 비교
    2. url_title_key 기반: 도메인 + 제목 조합 비교
    
    Args:
        documents: 중복 제거할 문서 리스트
        threshold: 유사도 임계값 (현재 미사용, 향후 임베딩 기반 유사도 비교용)
        
    Returns:
        List[Document]: 중복이 제거된 고유한 문서 리스트
        
    사용 예시:
        raw_docs = [doc1, doc2, doc3, doc2_duplicate]
        unique_docs = remove_duplicate_documents(raw_docs)
        # doc2_duplicate가 제거된 리스트 반환
    """
    if not documents:
        return []
    
    unique_documents = []
    seen_hashes = set()
    
    for doc in documents:
        # 해시 기반 중복 제거
        if doc.content_hash in seen_hashes:
            continue
        
        # URL 도메인 + 제목 기반 중복 제거
        url_title_key = f"{doc.url_domain}|{doc.title.lower()[:50]}"
        if url_title_key in seen_hashes:
            continue
        
        seen_hashes.add(doc.content_hash)
        seen_hashes.add(url_title_key)
        unique_documents.append(doc)
    
    return unique_documents

def sort_documents_by_relevance(documents: List[Document], query: str) -> List[Document]:
    """
    관련도 순으로 문서 정렬
    
    사용자의 원본 질문과 각 문서의 관련성을 계산하여 정렬.
    현재는 키워드 기반 유사도를 사용하며, 향후 임베딩 기반으로 개선 가능.
    
    관련도 계산 방식:
    - 제목 키워드 일치도: 70% 가중치
    - 내용 키워드 일치도: 30% 가중치
    - 제목에서 일치하는 키워드가 더 높은 점수
    
    Args:
        documents: 정렬할 문서 리스트
        query: 사용자의 원본 질문
        
    Returns:
        List[Document]: 관련도 높은 순으로 정렬된 문서 리스트
        
    주의사항:
        기존에 relevance_score가 설정된 문서는 재계산하지 않음
    """
    # 간단한 키워드 기반 관련도 계산 (실제로는 임베딩 기반으로 개선 가능)
    query_keywords = set(query.lower().split())
    
    def calculate_relevance(doc: Document) -> float:
        title_keywords = set(doc.title.lower().split())
        content_keywords = set(doc.content.lower().split()[:100])  # 첫 100단어만
        
        title_overlap = len(query_keywords & title_keywords) / max(len(query_keywords), 1)
        content_overlap = len(query_keywords & content_keywords) / max(len(query_keywords), 1)
        
        # 제목 가중치 높게 설정
        return (title_overlap * 0.7 + content_overlap * 0.3)
    
    # 관련도 점수 계산 및 정렬
    for doc in documents:
        if doc.relevance_score == 0.0:  # 이미 계산되지 않은 경우만
            doc.relevance_score = calculate_relevance(doc)
    
    return sorted(documents, key=lambda x: x.relevance_score, reverse=True)

def filter_documents_by_quality(documents: List[Document], min_content_length: int = 100) -> List[Document]:
    """
    문서 품질 기반 필터링
    
    수집된 문서들 중 품질이 낮은 문서들을 자동으로 제거.
    최소 품질 기준을 만족하는 문서만 요약 대상으로 선별합니다.
    
    품질 평가 기준:
    1. 최소 내용 길이: 100자 이상 (의미 있는 정보 포함 여부)
    2. 제목 존재: 빈 제목이 아닌 실제 제목 보유
    3. 유효한 URL: http/https로 시작하는 정상적인 URL
    4. 내용 유효성: 공백이 아닌 실제 텍스트 내용
    
    Args:
        documents: 필터링할 문서 리스트
        min_content_length: 최소 내용 길이 (기본값: 100자)
        
    Returns:
        List[Document]: 품질 기준을 만족하는 문서 리스트
        
    사용 예시:
        raw_docs = [good_doc, short_doc, no_title_doc, invalid_url_doc]
        quality_docs = filter_documents_by_quality(raw_docs, min_content_length=150)
        # good_doc만 반환됨
    """
    filtered = []
    
    for doc in documents:
        # 최소 내용 길이 확인
        if len(doc.content.strip()) < min_content_length:
            continue
            
        # 제목이 있는지 확인
        if not doc.title.strip():
            continue
            
        # URL이 유효한지 확인
        if not doc.url.strip() or not doc.url.startswith(('http://', 'https://')):
            continue
        
        filtered.append(doc)
    
    return filtered