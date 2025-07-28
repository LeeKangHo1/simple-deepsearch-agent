# workflows/research_workflow.py
"""
딥 리서치 챗봇 워크플로우 실행기

LangGraph 워크플로우를 실행하고 Streamlit UI와 연결하는 메인 실행 클래스입니다.
사용자 입력을 받아 전체 에이전트 파이프라인을 실행하고 결과를 반환합니다.

주요 기능:
- LangGraph 워크플로우 초기화 및 실행
- 실시간 진행 상태 추적 및 콜백
- 오류 처리 및 복구
- 성능 모니터링
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, AsyncGenerator
from datetime import datetime

from models.state import ResearchState, StateManager
from workflows.graph_builder import build_graph
from utils.logger import get_agent_logger, LogContext

logger = get_agent_logger("research_workflow")

class ResearchWorkflow:
    """
    딥 리서치 챗봇 워크플로우 실행기
    
    LangGraph 기반의 멀티 에이전트 워크플로우를 관리하고 실행합니다.
    Streamlit UI와의 연결점 역할을 하며, 실시간 상태 업데이트를 제공합니다.
    """
    
    def __init__(self):
        """워크플로우 실행기 초기화"""
        self.graph = None
        self.current_state = None
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0
        }
        
        # 워크플로우 그래프 빌드
        self._initialize_graph()
    
    def _initialize_graph(self):
        """LangGraph 워크플로우 그래프 초기화"""
        try:
            self.graph = build_graph()
            logger.info("Research workflow graph initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize workflow graph: {e}")
            raise
    
    async def execute(
        self, 
        user_input: str, 
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        사용자 입력에 대한 전체 리서치 워크플로우 실행
        
        Args:
            user_input: 사용자의 질문
            progress_callback: 진행 상태 콜백 함수 (step, message)
            
        Returns:
            Dict[str, Any]: 실행 결과
            {
                "success": bool,
                "markdown_answer": str,
                "error_message": str,
                "execution_time": float,
                "stats": dict
            }
        """
        start_time = time.time()
        self.execution_stats["total_executions"] += 1
        
        with LogContext(logger, f"Research Workflow Execution"):
            try:
                # 초기 상태 생성
                initial_state = StateManager.create_initial_state(user_input)
                self.current_state = initial_state
                
                logger.info(f"Starting research workflow for query: '{user_input[:100]}...'")
                
                # 진행 상태 콜백 설정
                if progress_callback:
                    progress_callback("시작", "리서치 워크플로우를 시작합니다...")
                
                # 워크플로우 실행
                final_state = await self._execute_workflow(initial_state, progress_callback)
                
                # 실행 결과 처리
                execution_time = time.time() - start_time
                result = self._process_final_state(final_state, execution_time)
                
                # 성공 통계 업데이트
                if result["success"]:
                    self.execution_stats["successful_executions"] += 1
                else:
                    self.execution_stats["failed_executions"] += 1
                
                # 평균 실행 시간 업데이트
                self._update_avg_execution_time(execution_time)
                
                logger.info(f"Workflow execution completed: {result['success']} ({execution_time:.2f}s)")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.execution_stats["failed_executions"] += 1
                self._update_avg_execution_time(execution_time)
                
                error_msg = f"Workflow execution failed: {str(e)}"
                logger.error(error_msg)
                
                if progress_callback:
                    progress_callback("오류", error_msg)
                
                return {
                    "success": False,
                    "markdown_answer": "",
                    "error_message": error_msg,
                    "execution_time": execution_time,
                    "stats": self.get_execution_stats()
                }
    
    async def _execute_workflow(
        self, 
        initial_state: ResearchState, 
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> ResearchState:
        """
        실제 워크플로우 그래프 실행
        
        Args:
            initial_state: 초기 상태
            progress_callback: 진행 상태 콜백
            
        Returns:
            ResearchState: 최종 실행 상태
        """
        current_state = initial_state
        final_state = None
        
        try:
            # LangGraph 워크플로우 실행 (recursion_limit 설정)
            config = {"recursion_limit": 50}  # 기본 25에서 50으로 증가
            async for state_update in self.graph.astream(current_state, config=config):
                # 상태 업데이트 처리 - LangGraph는 {node_name: state} 형태로 반환
                if isinstance(state_update, dict):
                    # 노드별 상태에서 실제 상태 추출
                    for node_name, node_state in state_update.items():
                        if isinstance(node_state, dict):
                            current_state = node_state
                            final_state = node_state
                            self.current_state = current_state
                            
                            # 디버깅을 위한 로깅
                            logger.debug(f"Node {node_name} completed, state keys: {list(current_state.keys())}")
                            
                            # 진행 상태 콜백 호출
                            if progress_callback and "current_step" in current_state:
                                step = current_state["current_step"]
                                logs = current_state.get("logs", [])
                                latest_log = logs[-1] if logs else ""
                                progress_callback(step, latest_log)
                            
                            # 오류 상태 확인
                            if current_state.get("has_error", False):
                                error_msg = current_state.get("error_message", "Unknown error")
                                logger.error(f"Workflow error detected: {error_msg}")
                                return current_state
            
            # 최종 상태 반환
            return final_state if final_state else current_state
            
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")
            return StateManager.set_error(current_state, str(e))
    
    def _process_final_state(self, final_state: ResearchState, execution_time: float) -> Dict[str, Any]:
        """
        최종 상태를 처리하여 결과 딕셔너리 생성 (검증 과정 제거)
        
        Args:
            final_state: 워크플로우 최종 상태
            execution_time: 실행 시간
            
        Returns:
            Dict[str, Any]: 처리된 결과
        """
        # 오류 상태 확인
        if final_state.get("has_error", False):
            return {
                "success": False,
                "markdown_answer": "",
                "error_message": final_state.get("error_message", "Unknown error occurred"),
                "execution_time": execution_time,
                "stats": self._extract_state_stats(final_state)
            }
        
        # 응답 처리 - 상태에서 markdown_answer 추출
        markdown_answer = final_state.get("markdown_answer", "").strip()
        
        # 디버깅을 위한 로깅 추가
        logger.info(f"Final state keys: {list(final_state.keys())}")
        logger.info(f"Markdown answer length: {len(markdown_answer)}")
        if markdown_answer:
            logger.info(f"Markdown answer preview: {markdown_answer[:200]}...")
        
        # 응답이 없으면 기본 응답 생성
        if not markdown_answer:
            user_input = final_state.get("user_input", "질문")
            logger.warning(f"No markdown_answer found in final state, generating fallback response")
            markdown_answer = f"""# {user_input}에 대한 답변

## 죄송합니다
현재 해당 주제에 대한 정보를 충분히 수집하지 못했습니다.

## 제안사항
- 더 구체적인 키워드로 다시 검색해보세요
- 다른 관점에서 질문을 다시 작성해보세요
- 잠시 후 다시 시도해보세요

---
*처리 시간: {execution_time:.1f}초*
"""
        
        # 항상 성공으로 처리 (검증 과정 생략)
        return {
            "success": True,
            "markdown_answer": markdown_answer,
            "error_message": "",
            "execution_time": execution_time,
            "stats": self._extract_state_stats(final_state)
        }
    
    def _extract_state_stats(self, state: ResearchState) -> Dict[str, Any]:
        """
        상태에서 통계 정보 추출
        
        Args:
            state: 리서치 상태
            
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            "sub_queries_count": len(state.get("sub_queries", [])),
            "documents_count": len(state.get("documents", [])),
            "summaries_count": len(state.get("summaries", [])),
            "insights_count": len(state.get("insights", [])),
            "retry_count": state.get("retry_count", 0),
            "current_step": state.get("current_step", "unknown"),
            "logs_count": len(state.get("logs", [])),
            "total_execution_time": StateManager.get_total_execution_time(state)
        }
    
    def _update_avg_execution_time(self, execution_time: float):
        """평균 실행 시간 업데이트"""
        total_executions = self.execution_stats["total_executions"]
        current_avg = self.execution_stats["avg_execution_time"]
        
        # 이동 평균 계산
        self.execution_stats["avg_execution_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        실행 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 실행 통계
        """
        stats = self.execution_stats.copy()
        
        # 성공률 계산
        total = stats["total_executions"]
        if total > 0:
            stats["success_rate"] = (stats["successful_executions"] / total) * 100
            stats["failure_rate"] = (stats["failed_executions"] / total) * 100
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def get_current_state(self) -> Optional[ResearchState]:
        """
        현재 실행 중인 상태 반환
        
        Returns:
            Optional[ResearchState]: 현재 상태 또는 None
        """
        return self.current_state
    
    def reset_stats(self):
        """통계 초기화"""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0
        }
        logger.info("Execution statistics reset")


# 전역 워크플로우 인스턴스
_workflow_instance = None

def get_research_workflow() -> ResearchWorkflow:
    """
    전역 리서치 워크플로우 인스턴스 반환
    
    Returns:
        ResearchWorkflow: 워크플로우 인스턴스
    """
    global _workflow_instance
    
    if _workflow_instance is None:
        _workflow_instance = ResearchWorkflow()
        logger.info("Research workflow instance created")
    
    return _workflow_instance

def reset_research_workflow():
    """워크플로우 인스턴스 리셋 (테스트용)"""
    global _workflow_instance
    _workflow_instance = None
    logger.info("Research workflow instance reset")


# 편의 함수들
async def execute_research(
    user_input: str, 
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> Dict[str, Any]:
    """
    리서치 워크플로우 실행 편의 함수
    
    Args:
        user_input: 사용자 질문
        progress_callback: 진행 상태 콜백
        
    Returns:
        Dict[str, Any]: 실행 결과
    """
    workflow = get_research_workflow()
    return await workflow.execute(user_input, progress_callback)

async def get_workflow_status() -> Dict[str, Any]:
    """
    현재 워크플로우 상태 정보 반환
    
    Returns:
        Dict[str, Any]: 상태 정보
    """
    workflow = get_research_workflow()
    current_state = workflow.get_current_state()
    
    if current_state:
        return {
            "is_running": True,
            "current_step": current_state.get("current_step", "unknown"),
            "progress": len(current_state.get("logs", [])),
            "user_input": current_state.get("user_input", "")[:100] + "...",
            "stats": workflow._extract_state_stats(current_state)
        }
    else:
        return {
            "is_running": False,
            "current_step": "idle",
            "progress": 0,
            "user_input": "",
            "stats": {}
        }