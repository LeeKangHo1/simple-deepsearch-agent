# models/state.py
"""
LangGraph 워크플로우에서 사용되는 State 클래스 정의
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
import time

class ResearchState(TypedDict):
    """
    딥 리서치 챗봇의 전체 워크플로우에서 공유되는 상태 클래스
    
    모든 에이전트가 이 State를 입력받아 수정하고 반환합니다.
    """
    
    # ================================
    # 기본 입력 및 출력
    # ================================
    user_input: str
    """사용자가 입력한 원본 질문"""
    
    markdown_answer: str
    """최종 마크다운 형식 응답 (출처 포함)"""
    
    # ================================
    # 질문 분석 단계
    # ================================
    sub_queries: List[str]
    """질문 분석 에이전트가 생성한 하위 쿼리 목록"""
    
    # ================================
    # 검색 및 문서 수집 단계
    # ================================
    search_results: List[Dict[str, Any]]
    """원본 검색 결과 (중복 제거 전)"""
    
    documents: List[Dict[str, Any]]
    """최종 선별된 문서 리스트
    각 문서 구조:
    {
        'title': str,           # 문서 제목
        'url': str,             # 문서 URL
        'content': str,         # 문서 내용
        'source': str,          # 검색 엔진 ('duckduckgo' or 'tavily')
        'relevance_score': float, # 질문과의 관련도 점수
        'embedding': List[float]  # 문서 임베딩 벡터 (선택적)
    }
    """
    
    # ================================
    # 요약 및 분석 단계
    # ================================
    summaries: List[str]
    """문서 요약 에이전트가 생성한 각 문서별 요약"""
    
    insights: List[str]
    """인사이트 생성 에이전트가 도출한 핵심 시사점"""
    
    # ================================
    # 검증 및 반복 처리
    # ================================
    is_valid: bool
    """검증 에이전트의 검증 결과 (True: 통과, False: 재처리 필요)"""
    
    validation_feedback: str
    """검증 에이전트의 피드백 메시지 (검증 실패 시 수정 방향)"""
    
    retry_count: int
    """검증 실패로 인한 재시도 횟수"""
    
    # ================================
    # 진행 상태 추적
    # ================================
    current_step: str
    """현재 처리 중인 단계 ('analyzing', 'searching', 'summarizing', 'generating', 'validating', 'completed')"""
    
    logs: List[str]
    """각 단계별 진행 로그 (UI 표시용)"""
    
    step_timestamps: Dict[str, float]
    """각 단계별 시작 시간 (성능 측정용)"""
    
    # ================================
    # 오류 처리
    # ================================
    error_message: Optional[str]
    """오류 발생 시 에러 메시지"""
    
    has_error: bool
    """오류 발생 여부 플래그"""


class StateManager:
    """State 조작을 위한 헬퍼 클래스"""
    
    @staticmethod
    def create_initial_state(user_input: str) -> ResearchState:
        """
        초기 상태 생성
        
        Args:
            user_input: 사용자 입력 질문
            
        Returns:
            ResearchState: 초기화된 상태 객체
        """
        return ResearchState(
            # 기본 입력 및 출력
            user_input=user_input,
            markdown_answer="",
            
            # 질문 분석 단계
            sub_queries=[],
            
            # 검색 및 문서 수집 단계
            search_results=[],
            documents=[],
            
            # 요약 및 분석 단계
            summaries=[],
            insights=[],
            
            # 검증 및 반복 처리
            is_valid=False,
            validation_feedback="",
            retry_count=0,
            
            # 진행 상태 추적
            current_step="initialized",
            logs=[],
            step_timestamps={},
            
            # 오류 처리
            error_message=None,
            has_error=False
        )
    
    @staticmethod
    def add_log(state: ResearchState, message: str) -> ResearchState:
        """
        진행 로그 추가
        
        Args:
            state: 현재 상태
            message: 로그 메시지
            
        Returns:
            ResearchState: 로그가 추가된 상태
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 새로운 상태 반환 (불변성 유지)
        new_state = state.copy()
        existing_logs = state.get("logs", [])
        new_state["logs"] = existing_logs + [log_entry]
        return new_state
    
    @staticmethod
    def set_step(state: ResearchState, step: str, message: str = "") -> ResearchState:
        """
        현재 단계 설정 및 타임스탬프 기록
        
        Args:
            state: 현재 상태
            step: 새로운 단계명
            message: 단계 변경 메시지
            
        Returns:
            ResearchState: 단계가 업데이트된 상태
        """
        new_state = state.copy()
        new_state["current_step"] = step
        existing_timestamps = state.get("step_timestamps", {})
        new_state["step_timestamps"] = existing_timestamps.copy()
        new_state["step_timestamps"][step] = time.time()
        
        if message:
            new_state = StateManager.add_log(new_state, message)
        
        return new_state
    
    @staticmethod
    def set_error(state: ResearchState, error_message: str) -> ResearchState:
        """
        오류 상태 설정
        
        Args:
            state: 현재 상태
            error_message: 오류 메시지
            
        Returns:
            ResearchState: 오류가 설정된 상태
        """
        new_state = state.copy()
        new_state["has_error"] = True
        new_state["error_message"] = error_message
        new_state["current_step"] = "error"
        
        return StateManager.add_log(new_state, f"❌ 오류 발생: {error_message}")
    
    @staticmethod
    def increment_retry(state: ResearchState) -> ResearchState:
        """
        재시도 횟수 증가
        
        Args:
            state: 현재 상태
            
        Returns:
            ResearchState: 재시도 횟수가 증가된 상태
        """
        new_state = state.copy()
        current_retry_count = state.get("retry_count", 0)
        new_state["retry_count"] = current_retry_count + 1
        
        return StateManager.add_log(
            new_state, 
            f"🔄 검증 실패로 재시도 중... ({new_state['retry_count']}/{2})"
        )
    
    @staticmethod
    def get_execution_time(state: ResearchState, step: str) -> Optional[float]:
        """
        특정 단계의 실행 시간 계산
        
        Args:
            state: 현재 상태
            step: 단계명
            
        Returns:
            float: 실행 시간 (초), 없으면 None
        """
        timestamps = state.get("step_timestamps", {})
        step_keys = list(timestamps.keys())
        
        if step not in step_keys:
            return None
        
        step_index = step_keys.index(step)
        start_time = timestamps[step]
        
        # 다음 단계가 있으면 다음 단계 시작 시간까지
        if step_index + 1 < len(step_keys):
            next_step = step_keys[step_index + 1]
            end_time = timestamps[next_step]
        else:
            # 마지막 단계면 현재 시간까지
            end_time = time.time()
        
        return end_time - start_time
    
    @staticmethod
    def get_total_execution_time(state: ResearchState) -> float:
        """
        전체 실행 시간 계산
        
        Args:
            state: 현재 상태
            
        Returns:
            float: 총 실행 시간 (초)
        """
        timestamps = state.get("step_timestamps", {})
        if not timestamps:
            return 0.0
        
        start_time = min(timestamps.values())
        end_time = max(timestamps.values())
        
        return end_time - start_time
    
    @staticmethod
    def get_state_summary(state: ResearchState) -> Dict[str, Any]:
        """
        상태 요약 정보 반환 (디버깅 및 모니터링용)
        
        Args:
            state: 현재 상태
            
        Returns:
            dict: 상태 요약 정보
        """
        return {
            "current_step": state["current_step"],
            "user_input": state["user_input"][:100] + "..." if len(state["user_input"]) > 100 else state["user_input"],
            "sub_queries_count": len(state["sub_queries"]),
            "documents_count": len(state["documents"]),
            "summaries_count": len(state["summaries"]),
            "insights_count": len(state["insights"]),
            "retry_count": state["retry_count"],
            "is_valid": state["is_valid"],
            "has_error": state["has_error"],
            "total_execution_time": StateManager.get_total_execution_time(state),
            "logs_count": len(state["logs"])
        }


# 상태 검증을 위한 유틸리티 함수들
def validate_state_transition(from_step: str, to_step: str) -> bool:
    """
    상태 전환의 유효성 검증
    
    Args:
        from_step: 현재 단계
        to_step: 이동할 단계
        
    Returns:
        bool: 유효한 전환이면 True
    """
    valid_transitions = {
        "initialized": ["analyzing"],
        "analyzing": ["searching"],
        "searching": ["summarizing"],
        "summarizing": ["generating_insights"],
        "generating_insights": ["generating_response"],
        "generating_response": ["validating"],
        "validating": ["completed", "generating_insights"],  # 재시도 가능
        "completed": [],
        "error": []
    }
    
    return to_step in valid_transitions.get(from_step, [])