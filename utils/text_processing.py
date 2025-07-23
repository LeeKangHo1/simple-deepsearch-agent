# utils/text_processing.py
"""
텍스트 처리 유틸리티 함수들

문서 내용 정제, 키워드 추출, 텍스트 정규화 등의 기능을 제공합니다.
웹 검색 결과나 LLM 응답의 품질을 향상시키기 위한 전처리에 사용됩니다.
"""

import re
import html
import unicodedata
from typing import List, Optional, Set, Dict
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """
    텍스트를 정제하여 깨끗한 상태로 만듭니다.
    
    HTML 태그 제거, 특수문자 정리, 공백 정규화 등을 수행합니다.
    웹 검색 결과에서 추출한 원시 텍스트를 처리할 때 사용됩니다.
    
    Args:
        text: 정제할 원본 텍스트
        remove_extra_whitespace: 불필요한 공백 제거 여부
        
    Returns:
        str: 정제된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # HTML 엔티티 디코딩 (&amp; → &, &lt; → < 등)
    cleaned = html.unescape(text)
    
    # HTML 태그 제거
    cleaned = remove_html_tags(cleaned)
    
    # 유니코드 정규화 (NFC 형식으로 통일)
    cleaned = unicodedata.normalize('NFC', cleaned)
    
    # 제어 문자 제거 (탭, 줄바꿈 제외)
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    # 불필요한 공백 정리
    if remove_extra_whitespace:
        cleaned = normalize_whitespace(cleaned)
    
    return cleaned.strip()

def remove_html_tags(text: str) -> str:
    """
    HTML 태그를 제거합니다.
    
    웹 페이지에서 추출한 텍스트에 포함된 HTML 마크업을 깨끗하게 제거합니다.
    단순한 정규식을 사용하여 빠르게 처리합니다.
    
    Args:
        text: HTML 태그가 포함된 텍스트
        
    Returns:
        str: HTML 태그가 제거된 순수 텍스트
    """
    if not text:
        return ""
    
    # HTML 태그 제거 (script, style 태그 내용도 함께 제거)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    공백 문자를 정규화합니다.
    
    연속된 공백을 하나로 줄이고, 줄바꿈을 적절히 처리합니다.
    문서 요약이나 응답 생성 시 깔끔한 형태로 만들기 위해 사용됩니다.
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        str: 공백이 정규화된 텍스트
    """
    if not text:
        return ""
    
    # 연속된 공백을 하나로 줄이기
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 연속된 줄바꿈을 최대 2개로 제한 (문단 구분 유지)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 줄 시작/끝의 공백 제거
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    return '\n'.join(lines)

def extract_keywords(text: str, max_keywords: int = 10, min_length: int = 2) -> List[str]:
    """
    텍스트에서 키워드를 추출합니다.
    
    간단한 빈도 기반 키워드 추출을 수행합니다.
    검색 쿼리 생성이나 문서 관련성 평가에 활용할 수 있습니다.
    
    Args:
        text: 키워드를 추출할 텍스트
        max_keywords: 최대 키워드 개수
        min_length: 키워드 최소 길이
        
    Returns:
        List[str]: 추출된 키워드 리스트 (빈도순)
    """
    if not text:
        return []
    
    # 텍스트 정제
    cleaned_text = clean_text(text).lower()
    
    # 불용어 목록 (한국어/영어 기본)
    stopwords = {
        # 한국어 불용어
        '이', '그', '저', '것', '들', '은', '는', '이', '가', '을', '를', 
        '에', '에서', '로', '으로', '와', '과', '하고', '도', '만', '까지',
        '부터', '의', '에게', '한테', '께', '한', '할', '하는', '했', '되',
        '된', '되는', '있', '없', '같', '다른', '새', '각', '모든', '어떤',
        # 영어 불용어  
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was',
        'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # 단어 추출 (한글, 영문, 숫자만)
    words = re.findall(r'[가-힣a-zA-Z0-9]+', cleaned_text)
    
    # 키워드 필터링 및 카운팅
    word_counts = {}
    for word in words:
        if (len(word) >= min_length and 
            word not in stopwords and 
            not word.isdigit()):  # 순수 숫자 제외
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # 빈도순으로 정렬하여 상위 키워드 반환
    sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [keyword for keyword, count in sorted_keywords[:max_keywords]]

def truncate_text(text: str, max_length: int, end_marker: str = "...") -> str:
    """
    텍스트를 지정된 길이로 자릅니다.
    
    문장 단위로 자르려고 시도하여 자연스러운 절단점을 찾습니다.
    문서 요약이나 스니펫 생성 시 사용됩니다.
    
    Args:
        text: 자를 텍스트
        max_length: 최대 길이
        end_marker: 생략 표시 (기본: "...")
        
    Returns:
        str: 잘린 텍스트
    """
    if not text or len(text) <= max_length:
        return text
    
    # end_marker 길이를 고려한 실제 자를 길이
    actual_max = max_length - len(end_marker)
    
    if actual_max <= 0:
        return end_marker
    
    # 문장 단위로 자르기 시도
    sentences = re.split(r'[.!?]\s+', text)
    result = ""
    
    for sentence in sentences:
        # 다음 문장을 추가했을 때 길이 체크
        next_result = result + sentence + ". " if result else sentence + ". "
        
        if len(next_result) <= actual_max:
            result = next_result
        else:
            break
    
    # 문장 단위로 자르기가 불가능하면 단어 단위로
    if not result.strip():
        words = text.split()
        result = ""
        
        for word in words:
            next_result = result + word + " " if result else word + " "
            if len(next_result) <= actual_max:
                result = next_result
            else:
                break
    
    # 마지막에 불필요한 공백과 구두점 정리
    result = result.strip().rstrip('.,!?')
    
    return result + end_marker if result else text[:actual_max] + end_marker

def is_valid_text(text: str, min_length: int = 5, max_length: int = 100000) -> bool:
    """
    텍스트가 유효한지 검증합니다.
    
    길이, 내용의 유의미성 등을 체크하여 처리할 가치가 있는 텍스트인지 판단합니다.
    문서 필터링이나 입력 검증에 사용됩니다.
    
    Args:
        text: 검증할 텍스트
        min_length: 최소 길이
        max_length: 최대 길이
        
    Returns:
        bool: 유효한 텍스트이면 True
    """
    if not text or not isinstance(text, str):
        return False
    
    cleaned = clean_text(text)
    
    # 길이 체크
    if not (min_length <= len(cleaned) <= max_length):
        return False
    
    # 의미 있는 문자가 충분히 있는지 체크
    meaningful_chars = re.findall(r'[가-힣a-zA-Z0-9]', cleaned)
    if len(meaningful_chars) < min_length * 0.5:  # 전체 길이의 50% 이상이 의미 있는 문자
        return False
    
    # 반복되는 문자나 패턴이 너무 많은지 체크
    unique_chars = set(cleaned.lower())
    if len(unique_chars) < min(10, len(cleaned) * 0.1):  # 너무 단조로운 텍스트
        return False
    
    return True

def extract_sentences(text: str, max_sentences: int = 5) -> List[str]:
    """
    텍스트에서 문장을 추출합니다.
    
    문서 요약이나 핵심 문장 추출에 사용됩니다.
    
    Args:
        text: 문장을 추출할 텍스트
        max_sentences: 최대 문장 수
        
    Returns:
        List[str]: 추출된 문장 리스트
    """
    if not text:
        return []
    
    # 문장 분리 (한국어와 영어 모두 고려)
    sentences = re.split(r'[.!?]+\s+', text)
    
    # 빈 문장 제거 및 정제
    valid_sentences = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if is_valid_text(cleaned, min_length=10):  # 최소 10자 이상의 문장만
            valid_sentences.append(cleaned)
    
    return valid_sentences[:max_sentences]

def count_words(text: str) -> Dict[str, int]:
    """
    텍스트의 단어 수 통계를 반환합니다.
    
    한국어와 영어를 구분하여 카운팅하며, 문서 분석에 유용한 정보를 제공합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        Dict[str, int]: 단어 수 통계 정보
    """
    if not text:
        return {"total_chars": 0, "korean_chars": 0, "english_words": 0, "numbers": 0}
    
    cleaned = clean_text(text)
    
    # 한글 문자 수
    korean_chars = len(re.findall(r'[가-힣]', cleaned))
    
    # 영어 단어 수
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', cleaned))
    
    # 숫자 개수
    numbers = len(re.findall(r'\b\d+\b', cleaned))
    
    return {
        "total_chars": len(cleaned),
        "korean_chars": korean_chars,
        "english_words": english_words,
        "numbers": numbers,
        "estimated_reading_time_minutes": max(1, (korean_chars + english_words * 4) // 200)  # 분당 200자 기준
    }

def extract_urls(text: str) -> List[str]:
    """
    텍스트에서 URL을 추출합니다.
    
    Args:
        text: URL을 추출할 텍스트
        
    Returns:
        List[str]: 추출된 URL 리스트
    """
    if not text:
        return []
    
    # URL 패턴 (http/https로 시작하는 URL)
    url_pattern = r'https?://[^\s<>"\'()[\]{}|\\^`~]*'
    urls = re.findall(url_pattern, text)
    
    # 중복 제거하면서 순서 유지
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls

def extract_emails(text: str) -> List[str]:
    """
    텍스트에서 이메일 주소를 추출합니다.
    
    Args:
        text: 이메일을 추출할 텍스트
        
    Returns:
        List[str]: 추출된 이메일 주소 리스트
    """
    if not text:
        return []
    
    # 이메일 패턴
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    # 중복 제거
    return list(set(emails))

def remove_patterns(text: str, patterns: List[str]) -> str:
    """
    텍스트에서 지정된 패턴들을 제거합니다.
    
    Args:
        text: 처리할 텍스트
        patterns: 제거할 정규식 패턴 리스트
        
    Returns:
        str: 패턴이 제거된 텍스트
    """
    if not text or not patterns:
        return text
    
    for pattern in patterns:
        try:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.warning(f"정규식 패턴 '{pattern}' 처리 실패: {e}")
    
    return text

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    긴 텍스트를 지정된 크기의 청크로 분할합니다.
    
    LLM 토큰 제한이나 벡터 임베딩을 위한 텍스트 분할에 사용됩니다.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 최대 크기
        overlap: 청크 간 겹치는 부분의 크기
        
    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 마지막 청크가 아니면 문장 경계에서 자르기 시도
        if end < len(text):
            # 다음 문장 경계 찾기
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 다음 시작 위치 (겹치는 부분 고려)
        start = end - overlap if end - overlap > start else end
        
        # 무한 루프 방지
        if start >= len(text):
            break
    
    return chunks

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트의 유사도를 계산합니다 (간단한 단어 기반).
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        float: 유사도 (0.0 ~ 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # 키워드 추출
    words1 = set(extract_keywords(text1, max_keywords=50))
    words2 = set(extract_keywords(text2, max_keywords=50))
    
    if not words1 or not words2:
        return 0.0
    
    # 자카드 유사도 계산
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def format_text_for_display(text: str, max_width: int = 80) -> str:
    """
    텍스트를 읽기 좋은 형태로 포맷팅합니다.
    
    Args:
        text: 포맷팅할 텍스트
        max_width: 최대 줄 너비
        
    Returns:
        str: 포맷팅된 텍스트
    """
    if not text:
        return ""
    
    # 기본 정제
    cleaned = clean_text(text)
    
    # 문단 단위로 분리
    paragraphs = cleaned.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        # 긴 줄을 적절히 분할
        words = paragraph.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # 현재 줄에 추가할 수 있는지 확인
            if current_length + word_length + 1 <= max_width:
                current_line.append(word)
                current_length += word_length + 1
            else:
                # 현재 줄 완성하고 새 줄 시작
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        # 마지막 줄 추가
        if current_line:
            lines.append(' '.join(current_line))
        
        formatted_paragraphs.append('\n'.join(lines))
    
    return '\n\n'.join(formatted_paragraphs)