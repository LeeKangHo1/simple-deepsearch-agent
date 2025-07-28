# utils/__init__.py
"""
유틸리티 모듈 초기화

Deep Research Chatbot에서 사용되는 공통 유틸리티 함수들을 제공합니다.
"""

from .text_processing import (
    clean_text,
    extract_keywords,
    truncate_text,
    is_valid_text,
    calculate_text_similarity
)

from .validators import (
    sanitize_input,
    validate_search_query,
    validate_document,
    validate_response_structure,
    validate_query_list,
    validate_insights
)

from .logger import (
    setup_logging,
    get_logger,
    get_agent_logger,
    log_execution_time,
    create_agent_logger
)

__all__ = [
    # text_processing
    'clean_text',
    'extract_keywords', 
    'truncate_text',
    'is_valid_text',
    'calculate_text_similarity',
    
    # validators
    'sanitize_input',
    'validate_search_query',
    'validate_document',
    'validate_response_structure',
    'validate_query_list',
    'validate_insights',
    
    # logger
    'setup_logging',
    'get_logger',
    'get_agent_logger',
    'log_execution_time',
    'create_agent_logger'
]