# utils/validators.py
"""
데이터 검증 함수들

Deep Research Chatbot에서 사용되는 다양한 데이터의 유효성을 검증합니다.
사용자 입력, 검색 쿼리, 문서 내용, 응답 구조 등을 검증하여 품질을 보장합니다.
"""

import re
import urllib.parse
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime
import json
from enum import Enum

from models.data_models import Document, SearchEngine, DocumentType
from .text_processing import clean_text, is_valid_text

logger = logging.getLogger(__name__)

# 위험한 패턴들 (보안 검증용)
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # JavaScript
    r'javascript:',                # JavaScript URL
    r'vbscript:',                 # VBScript URL
    r'onload\s*=',                # 이벤트 핸들러
    r'onerror\s*=',
    r'onclick\s*=',
    r'<iframe[^>]*>',             # iframe 태그
    r'<object[^>]*>',             # object 태그
    r'<embed[^>]*>',              # embed 태그
]

def sanitize_input(text: str, max_length: int = 10000, allow_html: bool = False) -> str:
    """
    사용자 입력을 안전하게 정제합니다.
    
    XSS 공격 방지, 길이 제한, 특수문자 처리 등을 수행하여
    안전한 입력만 허용합니다.
    
    Args:
        text: 정제할 입력 텍스트
        max_length: 최대 길이 제한
        allow_html: HTML 태그 허용 여부
        
    Returns:
        str: 정제된 안전한 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 길이 제한
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"입력 텍스트가 최대 길이({max_length})를 초과하여 잘렸습니다.")
    
    # HTML 허용하지 않는 경우 정제
    if not allow_html:
        text = clean_text(text)
    
    # 위험한 패턴 제거
    for pattern in DANGEROUS_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 제어 문자 제거 (줄바꿈, 탭 제외)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()

def validate_search_query(query: str) -> Tuple[bool, str]:
    """
    검색 쿼리의 유효성을 검증합니다.
    
    검색 API에 전달하기 전에 쿼리가 적절한 형태인지 확인하고,
    품질이 낮은 쿼리를 걸러냅니다.
    
    Args:
        query: 검증할 검색 쿼리
        
    Returns:
        Tuple[bool, str]: (유효성 여부, 오류 메시지)
    """
    if not query or not isinstance(query, str):
        return False, "검색 쿼리가 비어있습니다."
    
    # 입력 정제
    cleaned_query = sanitize_input(query, max_length=500)
    
    if not cleaned_query:
        return False, "유효한 검색어가 없습니다."
    
    # 최소 길이 확인
    if len(cleaned_query.strip()) < 2:
        return False, "검색어가 너무 짧습니다. (최소 2자 이상)"
    
    # 최대 길이 확인
    if len(cleaned_query) > 200:
        return False, "검색어가 너무 깁니다. (최대 200자)"
    
    # 의미 있는 문자 확인
    meaningful_chars = re.findall(r'[가-힣a-zA-Z0-9]', cleaned_query)
    if len(meaningful_chars) < 2:
        return False, "검색어에 의미 있는 문자가 부족합니다."
    
    # 금지된 문자 패턴 확인
    forbidden_patterns = [
        r'^[^\w\s가-힣]*$',  # 특수문자만 있는 경우
        r'(.)\1{10,}',        # 같은 문자 10번 이상 반복
        r'^[\s]*$',           # 공백만 있는 경우
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, cleaned_query):
            return False, "부적절한 검색어 패턴입니다."
    
    # SQL 인젝션 패턴 확인 (기본적인 보안)
    sql_patterns = [
        r"('\s*(or|and)\s*')",
        r"(union\s+select)",
        r"(drop\s+table)",
        r"(delete\s+from)",
        r"(insert\s+into)",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            logger.warning(f"의심스러운 SQL 패턴 감지: {query}")
            return False, "안전하지 않은 검색어입니다."
    
    return True, ""

def validate_document(doc: Union[Document, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    문서 객체의 유효성을 검증합니다.

    웹 검색 결과로 받은 문서가 처리할 가치가 있는지 판단하고,
    필수 필드와 데이터 품질을 확인합니다.

    Args:
        doc: 검증할 문서 (Document 객체 또는 딕셔너리)

    Returns:
        Tuple[bool, List[str]]: (유효성 여부, 오류 메시지 리스트)
    """
    errors = []

    # Document 객체인 경우 딕셔너리로 변환
    if isinstance(doc, Document):
        doc_dict = doc.to_dict()
    elif isinstance(doc, dict):
        doc_dict = doc
    else:
        return False, ["문서가 올바른 형식이 아닙니다."]

    # 필수 필드 확인
    required_fields = ['title', 'url', 'content']
    for field in required_fields:
        if field not in doc_dict or not doc_dict[field]:
            errors.append(f"필수 필드 '{field}'가 없습니다.")

    if errors:
        return False, errors

    # 제목 검증
    title = doc_dict['title'].strip()
    if len(title) < 3:
        errors.append("제목이 너무 짧습니다. (최소 3자)")
    elif len(title) > 500:
        errors.append("제목이 너무 깁니다. (최대 500자)")

    # URL 검증
    url = doc_dict['url'].strip()
    if not is_valid_url(url):
        errors.append("URL이 유효하지 않습니다.")

    # 내용 검증
    content = doc_dict['content'].strip()
    if not is_valid_text(content, min_length=50):
        errors.append("문서 내용이 너무 짧거나 유효하지 않습니다. (최소 50자)")
    elif len(content) > 50000:
        errors.append("문서 내용이 너무 깁니다. (최대 50,000자)")

    # 중복 문장 패턴 확인
    if len(content) > 100:
        sentences = content.split('.')
        if len(sentences) > 5:
            unique_sentences = set(s.strip().lower() for s in sentences if len(s.strip()) > 10)
            if len(unique_sentences) < len(sentences) * 0.7:
                errors.append("문서에 반복되는 내용이 너무 많습니다.")

    # source 검증 (문자열이어야 함)
    source = doc_dict.get('source', '')
    if isinstance(source, Enum):
        source = source.value
    if source not in [e.value for e in SearchEngine]:
        errors.append(f"유효하지 않은 검색 엔진: {source}")

    # 관련성 점수 검증
    if 'relevance_score' in doc_dict:
        score = doc_dict['relevance_score']
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            errors.append("관련성 점수는 0.0~1.0 사이의 숫자여야 합니다.")

    return len(errors) == 0, errors


def validate_response_structure(response: str) -> Tuple[bool, List[str]]:
    """
    마크다운 응답의 구조를 검증합니다.
    
    생성된 응답이 PRD에서 요구하는 형식을 만족하는지 확인하고,
    출처 표시, 구조화, 내용 품질 등을 검증합니다.
    
    Args:
        response: 검증할 마크다운 응답
        
    Returns:
        Tuple[bool, List[str]]: (유효성 여부, 오류 메시지 리스트)
    """
    errors = []
    
    if not response or not isinstance(response, str):
        return False, ["응답이 비어있습니다."]
    
    response = response.strip()
    
    # 최소 길이 확인
    if len(response) < 100:
        errors.append("응답이 너무 짧습니다. (최소 100자)")
    
    # 최대 길이 확인
    if len(response) > 10000:
        errors.append("응답이 너무 깁니다. (최대 10,000자)")
    
    # 마크다운 제목 확인
    if not re.search(r'^#+\s+.+', response, re.MULTILINE):
        errors.append("마크다운 제목이 없습니다. (# 또는 ## 등)")
    
    # 불릿 포인트 또는 구조화된 내용 확인
    structure_patterns = [
        r'^[-*+]\s+.+',      # 불릿 포인트
        r'^\d+\.\s+.+',      # 번호 리스트
        r'^#{2,}\s+.+',      # 소제목
    ]
    
    has_structure = any(
        re.search(pattern, response, re.MULTILINE) 
        for pattern in structure_patterns
    )
    
    if not has_structure:
        errors.append("응답에 구조화된 내용이 없습니다. (불릿포인트, 소제목 등)")
    
    # 출처 표시 확인
    source_patterns = [
        r'\*\([^)]*출처[^)]*\)\*',           # *(출처: domain.com)*
        r'\*\([^)]*source[^)]*\)\*',         # *(source: domain.com)*
        r'\[[^\]]*\]\([^)]+\)',              # [텍스트](URL)
        r'https?://[^\s)]+',                 # 직접 URL
    ]
    
    has_sources = any(
        re.search(pattern, response, re.IGNORECASE) 
        for pattern in source_patterns
    )
    
    if not has_sources:
        errors.append("응답에 출처 표시가 없습니다.")
    
    # 내용 품질 확인
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    meaningful_lines = [line for line in lines if len(line) > 10 and not line.startswith('#')]
    
    if len(meaningful_lines) < 3:
        errors.append("응답에 의미 있는 내용이 부족합니다.")
    
    # 중복 문장 확인
    if len(meaningful_lines) > 5:
        unique_lines = set(line.lower() for line in meaningful_lines)
        if len(unique_lines) < len(meaningful_lines) * 0.8:
            errors.append("응답에 중복되는 내용이 너무 많습니다.")
    
    # HTML 태그 확인 (마크다운에는 없어야 함)
    html_tags = re.findall(r'<[^>]+>', response)
    if html_tags:
        errors.append(f"응답에 HTML 태그가 포함되어 있습니다: {html_tags[:3]}")
    
    return len(errors) == 0, errors

def is_valid_url(url: str) -> bool:
    """
    URL의 유효성을 검증합니다.
    
    Args:
        url: 검증할 URL
        
    Returns:
        bool: 유효한 URL이면 True
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urllib.parse.urlparse(url.strip())
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False

def extract_domain(url: str) -> str:
    """
    URL에서 도메인을 추출합니다.
    
    Args:
        url: 도메인을 추출할 URL
        
    Returns:
        str: 추출된 도메인 (실패 시 'unknown')
    """
    if not url:
        return 'unknown'
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # 포트 번호 제거
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain if domain else 'unknown'
    except Exception:
        return 'unknown'

def validate_json_structure(data: Union[str, dict], required_fields: List[str]) -> Tuple[bool, str]:
    """
    JSON 데이터의 구조를 검증합니다.
    
    API 응답이나 설정 파일의 유효성을 확인할 때 사용합니다.
    
    Args:
        data: 검증할 JSON 데이터 (문자열 또는 딕셔너리)
        required_fields: 필수 필드 리스트
        
    Returns:
        Tuple[bool, str]: (유효성 여부, 오류 메시지)
    """
    try:
        # 문자열인 경우 파싱
        if isinstance(data, str):
            data = json.loads(data)
        
        if not isinstance(data, dict):
            return False, "데이터가 JSON 객체가 아닙니다."
        
        # 필수 필드 확인
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"필수 필드가 없습니다: {missing_fields}"
        
        return True, ""
        
    except json.JSONDecodeError as e:
        return False, f"JSON 파싱 오류: {e}"
    except Exception as e:
        return False, f"검증 오류: {e}"

def validate_search_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    검색 결과 리스트를 검증하고 유효한 결과만 반환합니다.
    
    Args:
        results: 검증할 검색 결과 리스트
        
    Returns:
        Tuple[List[Dict[str, Any]], int]: (유효한 결과, 제거된 개수)
    """
    if not results:
        return [], 0
    
    valid_results = []
    removed_count = 0
    
    for result in results:
        try:
            # 기본 필드 확인
            if not all(key in result for key in ['title', 'url']):
                removed_count += 1
                continue
            
            # URL 유효성 확인
            if not is_valid_url(result['url']):
                removed_count += 1
                continue
            
            # 제목 유효성 확인
            title = result.get('title', '').strip()
            if not title or len(title) < 3:
                removed_count += 1
                continue
            
            # 내용 확인 (있는 경우)
            content = result.get('content', '').strip()
            if content and len(content) < 20:  # 너무 짧은 내용 제외
                result['content'] = ''  # 빈 내용으로 설정하되 결과는 유지
            
            valid_results.append(result)
            
        except Exception as e:
            logger.warning(f"검색 결과 검증 중 오류: {e}")
            removed_count += 1
    
    if removed_count > 0:
        logger.info(f"검색 결과 검증 완료: {len(valid_results)}개 유효, {removed_count}개 제거")
    
    return valid_results, removed_count

def validate_query_list(queries: List[str]) -> Tuple[List[str], int]:
    """
    쿼리 리스트를 검증하고 유효한 쿼리만 반환합니다.
    
    Args:
        queries: 검증할 쿼리 리스트
        
    Returns:
        Tuple[List[str], int]: (유효한 쿼리, 제거된 개수)
    """
    if not queries:
        return [], 0
    
    valid_queries = []
    removed_count = 0
    
    for query in queries:
        is_valid, error = validate_search_query(query)
        if is_valid:
            valid_queries.append(sanitize_input(query, max_length=200))
        else:
            logger.debug(f"쿼리 제거됨: '{query}' - {error}")
            removed_count += 1
    
    # 중복 제거
    unique_queries = []
    seen = set()
    for query in valid_queries:
        query_lower = query.lower().strip()
        if query_lower not in seen:
            seen.add(query_lower)
            unique_queries.append(query)
        else:
            removed_count += 1
    
    if removed_count > 0:
        logger.info(f"쿼리 검증 완료: {len(unique_queries)}개 유효, {removed_count}개 제거")
    
    return unique_queries, removed_count

def validate_insights(insights: List[str]) -> Tuple[List[str], int]:
    """
    인사이트 리스트를 검증하고 품질이 좋은 인사이트만 반환합니다.
    
    Args:
        insights: 검증할 인사이트 리스트
        
    Returns:
        Tuple[List[str], int]: (유효한 인사이트, 제거된 개수)
    """
    if not insights:
        return [], 0
    
    valid_insights = []
    removed_count = 0
    
    for insight in insights:
        if not insight or not isinstance(insight, str):
            removed_count += 1
            continue
        
        cleaned_insight = sanitize_input(insight, max_length=1000)
        
        # 최소 길이 확인
        if len(cleaned_insight) < 20:
            removed_count += 1
            continue
        
        # 의미 있는 내용인지 확인
        if not is_valid_text(cleaned_insight, min_length=20):
            removed_count += 1
            continue
        
        # 너무 단순한 나열인지 확인
        if cleaned_insight.count(',') > 10 or cleaned_insight.count('·') > 10:
            removed_count += 1
            continue
        
        valid_insights.append(cleaned_insight)
    
    # 중복 또는 너무 유사한 인사이트 제거
    unique_insights = []
    for insight in valid_insights:
        is_duplicate = False
        for existing in unique_insights:
            # 단순한 유사도 계산 (공통 단어 비율)
            words1 = set(insight.lower().split())
            words2 = set(existing.lower().split())
            
            if len(words1) > 0 and len(words2) > 0:
                common_words = words1 & words2
                similarity = len(common_words) / max(len(words1), len(words2))
                
                if similarity > 0.7:  # 70% 이상 유사하면 중복으로 판단
                    is_duplicate = True
                    removed_count += 1
                    break
        
        if not is_duplicate:
            unique_insights.append(insight)
    
    if removed_count > 0:
        logger.info(f"인사이트 검증 완료: {len(unique_insights)}개 유효, {removed_count}개 제거")
    
    return unique_insights, removed_count