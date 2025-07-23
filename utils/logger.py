# utils/logger.py
"""
로깅 설정 및 유틸리티 함수들

Deep Research Chatbot의 로깅 시스템을 설정하고 관리합니다.
각 에이전트별 로그, 성능 측정, 오류 추적 등을 제공합니다.
"""

import logging
import logging.handlers
import time
import functools
from typing import Optional, Callable, Any
from pathlib import Path
import sys
from datetime import datetime

from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """
    콘솔 출력용 컬러 포맷터
    
    로그 레벨에 따라 다른 색상으로 출력하여 가독성을 높입니다.
    개발 환경에서 디버깅할 때 유용합니다.
    """
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청색
        'INFO': '\033[32m',     # 녹색  
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 자주색
        'RESET': '\033[0m'      # 색상 초기화
    }
    
    def format(self, record):
        """로그 레코드에 색상 적용"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # 로그 레벨에 색상 적용
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)

class AgentLogger:
    """
    에이전트별 로거 관리 클래스
    
    각 에이전트가 독립적인 로거를 가질 수 있도록 관리하며,
    에이전트별 성능 측정과 디버깅 정보를 제공합니다.
    """
    
    def __init__(self, agent_name: str):
        """
        에이전트 로거 초기화
        
        Args:
            agent_name: 에이전트 이름 (예: 'question_analyzer_logger', 'web_search_logger')
        """
        self.agent_name = agent_name
        self.logger = get_logger(f"{agent_name}_logger")
        self.start_time = None
        self.step_times = {}
    
    def start_step(self, step_name: str):
        """처리 단계 시작 로깅"""
        self.start_time = time.time()
        self.logger.info(f"[{self.agent_name}] {step_name} 시작")
    
    def end_step(self, step_name: str, success: bool = True, details: str = ""):
        """처리 단계 완료 로깅"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.step_times[step_name] = elapsed
            
            status = "완료" if success else "실패"
            log_msg = f"[{self.agent_name}] {step_name} {status} ({elapsed:.2f}초)"
            
            if details:
                log_msg += f" - {details}"
            
            if success:
                self.logger.info(log_msg)
            else:
                self.logger.error(log_msg)
        
        self.start_time = None
    
    def log_performance(self):
        """에이전트 성능 통계 로깅"""
        if self.step_times:
            total_time = sum(self.step_times.values())
            self.logger.info(f"[{self.agent_name}] 총 처리 시간: {total_time:.2f}초")
            
            for step, duration in self.step_times.items():
                percentage = (duration / total_time) * 100
                self.logger.debug(f"  - {step}: {duration:.2f}초 ({percentage:.1f}%)")

def setup_logging(
    log_category: str = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    로깅 시스템을 설정합니다.
    
    파일 로깅과 콘솔 로깅을 모두 지원하며, 로그 로테이션도 설정합니다.
    애플리케이션 시작 시 한 번만 호출하면 됩니다.
    
    Args:
        log_category: 로그 카테고리 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 기본 경로 사용)
        enable_console: 콘솔 출력 활성화 여부
        enable_colors: 콘솔 색상 출력 활성화 여부
        max_file_size: 로그 파일 최대 크기 (바이트)
        backup_count: 백업 파일 개수
    """
    
    # 설정에서 로그 카테고리 가져오기
    if log_category is None:
        log_category = settings.LOG_LEVEL
    
    # 로그 파일 경로 설정
    if log_file is None:
        log_file = "./logs/app.log"
    
    # 로그 디렉토리 생성
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_category.upper(), logging.INFO))
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 포맷터 설정
    detailed_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    simple_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 파일 핸들러 설정 (로테이션 포함)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
        file_formatter = logging.Formatter(detailed_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"로그 파일 설정 실패: {e}", file=sys.stderr)
    
    # 콘솔 핸들러 설정
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_category.upper(), logging.INFO))
        
        if enable_colors and sys.stdout.isatty():  # 터미널에서만 색상 사용
            console_formatter = ColoredFormatter(simple_format)
        else:
            console_formatter = logging.Formatter(simple_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 외부 라이브러리 로그 레벨 조정 (노이즈 감소)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # 설정 완료 로그
    root_logger.info(f"로깅 시스템 초기화 완료 (카테고리: {log_category})")

def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름의 로거를 반환합니다.
    
    에이전트별로 독립적인 로거를 사용할 수 있도록 합니다.
    로거 이름은 '{에이전트명}_logger' 형식을 사용합니다.
    
    Args:
        name: 로거 이름 (예: 'question_analyzer_logger')
        
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    return logging.getLogger(name)

def log_execution_time(func: Callable = None, *, logger_name: str = None):
    """
    함수 실행 시간을 측정하고 로그를 남기는 데코레이터
    
    에이전트 함수나 중요한 처리 함수의 성능을 모니터링할 때 사용합니다.
    
    사용 예시:
        @log_execution_time
        def process_documents(docs):
            # 처리 로직
            return results
        
        # 또는 특정 로거 사용
        @log_execution_time(logger_name="search_logger")
        def search_web(query):
            # 검색 로직
            return results
    
    Args:
        func: 래핑할 함수
        logger_name: 사용할 로거 이름 (기본값: 함수가 속한 모듈명_logger)
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # 로거 설정
            if logger_name:
                logger = get_logger(logger_name)
            else:
                # 모듈명에 _logger 추가
                module_name = f.__module__.split('.')[-1]  # 마지막 모듈명만 사용
                logger = get_logger(f"{module_name}_logger")
            
            # 실행 시간 측정
            start_time = time.time()
            function_name = f.__name__
            
            logger.debug(f"{function_name} 실행 시작")
            
            try:
                result = f(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                logger.info(f"{function_name} 실행 완료 ({elapsed_time:.3f}초)")
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"{function_name} 실행 실패 ({elapsed_time:.3f}초): {str(e)}"
                )
                raise
        
        return wrapper
    
    # 데코레이터가 인자 없이 사용된 경우
    if func is not None:
        return decorator(func)
    
    # 데코레이터가 인자와 함께 사용된 경우
    return decorator

class LogContext:
    """
    컨텍스트 매니저를 사용한 로그 그룹핑
    
    관련된 로그들을 그룹으로 묶어서 추적할 때 사용합니다.
    에이전트의 전체 처리 과정을 하나의 컨텍스트로 관리할 수 있습니다.
    """
    
    def __init__(self, logger: logging.Logger, context_name: str, log_category: int = logging.INFO):
        """
        로그 컨텍스트 초기화
        
        Args:
            logger: 사용할 로거
            context_name: 컨텍스트 이름
            log_category: 로그 카테고리 (logging.DEBUG, INFO 등)
        """
        self.logger = logger
        self.context_name = context_name
        self.log_category = log_category
        self.start_time = None
    
    def __enter__(self):
        """컨텍스트 시작"""
        self.start_time = time.time()
        self.logger.log(self.log_category, f"=== {self.context_name} 시작 ===")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.log(
                self.log_category, 
                f"=== {self.context_name} 완료 ({elapsed_time:.3f}초) ==="
            )
        else:
            self.logger.error(
                f"=== {self.context_name} 실패 ({elapsed_time:.3f}초): {exc_val} ==="
            )
    
    def log(self, message: str, category: int = None):
        """컨텍스트 내부 로그"""
        if category is None:
            category = self.log_category
        
        self.logger.log(category, f"[{self.context_name}] {message}")

def create_agent_logger(agent_name: str) -> AgentLogger:
    """
    에이전트별 로거 생성 헬퍼 함수
    
    Args:
        agent_name: 에이전트 이름 (예: 'question_analyzer')
        
    Returns:
        AgentLogger: 에이전트 로거 인스턴스 (로거명: '{agent_name}_logger')
    """
    return AgentLogger(agent_name)

def log_system_info():
    """
    시스템 정보를 로그에 기록합니다.
    
    애플리케이션 시작 시 환경 정보를 기록하여 디버깅에 도움이 됩니다.
    """
    logger = get_logger("system_logger")
    
    try:
        import platform
        import psutil
        
        logger.info("=== 시스템 정보 ===")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU 코어: {psutil.cpu_count()}")
        logger.info(f"메모리: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        logger.info(f"현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 환경 설정 정보
        logger.info("=== 애플리케이션 설정 ===")
        logger.info(f"LLM 제공자: {settings.LLM_PROVIDER}")
        logger.info(f"채팅 모델: {settings.chat_model}")
        logger.info(f"임베딩 모델: {settings.embedding_model}")
        logger.info(f"최대 검색 결과: {settings.MAX_SEARCH_RESULTS}")
        logger.info(f"LangSmith 활성화: {settings.LANGCHAIN_TRACING_V2}")
        
    except ImportError:
        logger.warning("psutil 패키지가 설치되지 않아 상세 시스템 정보를 표시할 수 없습니다.")
    except Exception as e:
        logger.error(f"시스템 정보 수집 실패: {e}")

def get_agent_logger(agent_name: str) -> logging.Logger:
    """
    에이전트별 로거를 간편하게 가져오는 함수
    
    Args:
        agent_name: 에이전트 이름 (예: 'question_analyzer')
        
    Returns:
        logging.Logger: '{agent_name}_logger' 형식의 로거
    """
    return get_logger(f"{agent_name}_logger")

def log_agent_start(agent_name: str, input_data: Any = None):
    """
    에이전트 시작 로그를 기록합니다.
    
    Args:
        agent_name: 에이전트 이름
        input_data: 입력 데이터 (선택적)
    """
    logger = get_agent_logger(agent_name)
    
    if input_data:
        # 민감한 정보는 로그에 포함하지 않도록 간단히 처리
        if isinstance(input_data, str) and len(input_data) > 100:
            data_preview = input_data[:100] + "..."
        else:
            data_preview = str(input_data)[:200]
        
        logger.info(f"[{agent_name}] 에이전트 시작 - 입력: {data_preview}")
    else:
        logger.info(f"[{agent_name}] 에이전트 시작")

def log_agent_end(agent_name: str, success: bool = True, output_data: Any = None, error: str = None):
    """
    에이전트 종료 로그를 기록합니다.
    
    Args:
        agent_name: 에이전트 이름
        success: 성공 여부
        output_data: 출력 데이터 (선택적)
        error: 오류 메시지 (선택적)
    """
    logger = get_agent_logger(agent_name)
    
    if success:
        if output_data:
            # 출력 데이터 요약
            if isinstance(output_data, (list, tuple)):
                data_summary = f"리스트 {len(output_data)}개 항목"
            elif isinstance(output_data, dict):
                data_summary = f"딕셔너리 {len(output_data)}개 키"
            elif isinstance(output_data, str) and len(output_data) > 100:
                data_summary = f"텍스트 {len(output_data)}자"
            else:
                data_summary = str(type(output_data).__name__)
            
            logger.info(f"[{agent_name}] 에이전트 완료 - 출력: {data_summary}")
        else:
            logger.info(f"[{agent_name}] 에이전트 완료")
    else:
        error_msg = error or "알 수 없는 오류"
        logger.error(f"[{agent_name}] 에이전트 실패 - 오류: {error_msg}")

# 모듈 로드 시 기본 로깅 설정
if not logging.getLogger().handlers:
    setup_logging()
    log_system_info()