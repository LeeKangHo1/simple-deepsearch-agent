# utils/__init__.py
"""
유틸리티 모듈 초기화

공통으로 사용되는 유틸리티 함수들을 제공하는 패키지입니다.
프로젝트 전반에서 사용되는 텍스트 처리, 로깅, 검증 함수들을 포함합니다.
"""

from .logger import (
    setup_logging,
    get_logger,
    log_execution_time
)

from .text_processing import (
    clean_text,
    extract_keywords,
    truncate_text,
    normalize_whitespace,
    remove_html_tags,
    is_valid_text
)

from .validators import (
    validate_search_query,
    validate_document,
    validate_response_structure,
    sanitize_input,
    is_valid_url,
    extract_domain
)

__all__ = [
    # logger
    'setup_logging',
    'get_logger', 
    'log_execution_time',
    
    # text_processing
    'clean_text',
    'extract_keywords',
    'truncate_text',
    'normalize_whitespace', 
    'remove_html_tags',
    'is_valid_text',
    
    # validators
    'validate_search_query',
    'validate_document',
    'validate_response_structure',
    'sanitize_input',
    'is_valid_url',
    'extract_domain'
]