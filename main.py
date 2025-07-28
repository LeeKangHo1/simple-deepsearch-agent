# main.py
import streamlit as st
import time
import asyncio
from typing import List, Dict, Optional
import logging

# 프로젝트 모듈들
from workflows.research_workflow import execute_research, get_workflow_status
from config.settings import settings
from utils.logger import setup_logging, get_logger

# 로깅 설정
setup_logging()
logger = get_logger("streamlit_app")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Deep Research Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .status-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .status-message {
        font-size: 14px;
        color: #666;
        margin: 0.5rem 0;
    }
    
    .result-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-question {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #2196f3;
    }
    
    .processing-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .spinner {
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #1f77b4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """세션 상태 초기화"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = ""
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    if 'result' not in st.session_state:
        st.session_state.result = ""
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

def add_status_message(message: str):
    """상태 메시지 추가"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.status_messages.append(f"[{timestamp}] {message}")

async def execute_research_workflow(query: str):
    """실제 LangGraph 워크플로우 실행"""
    
    def progress_callback(step: str, message: str):
        """진행 상태 콜백 함수"""
        add_status_message(message)
        st.session_state.current_step = step
    
    try:
        # 워크플로우 실행
        result = await execute_research(query, progress_callback)
        
        # 디버깅을 위한 로깅 추가
        logger.info(f"Workflow result: success={result.get('success')}, answer_length={len(result.get('markdown_answer', ''))}")
        
        if result["success"]:
            markdown_answer = result["markdown_answer"]
            if markdown_answer and markdown_answer.strip():
                st.session_state.result = markdown_answer
                st.session_state.execution_stats = result["stats"]
                add_status_message(f"✅ 리서치 완료! (실행 시간: {result['execution_time']:.1f}초)")
            else:
                # 답변이 비어있는 경우
                st.session_state.result = f"# 검색 결과 없음\n\n'{query}'에 대한 정보를 찾을 수 없습니다.\n\n다른 키워드로 다시 시도해보세요."
                add_status_message("⚠️ 검색 결과가 없습니다.")
        else:
            error_msg = result.get("error_message", "알 수 없는 오류가 발생했습니다.")
            st.session_state.result = f"# 오류 발생\n\n{error_msg}"
            add_status_message(f"❌ 리서치 실패: {error_msg}")
            logger.error(f"Research workflow failed: {error_msg}")
    
    except Exception as e:
        error_msg = f"워크플로우 실행 중 오류 발생: {str(e)}"
        st.session_state.result = f"# 시스템 오류\n\n{error_msg}"
        add_status_message(f"❌ 시스템 오류: {error_msg}")
        logger.error(f"Workflow execution error: {e}", exc_info=True)

def main():
    """메인 애플리케이션"""
    init_session_state()
    
    # 헤더
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔍 Deep Research Chatbot")
    st.markdown("**다각도 심층 조사를 통한 신뢰할 수 있는 정보 제공**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 질문 입력 섹션
    st.markdown("### 💭 질문을 입력해주세요")
    
    with st.form("research_form"):
        user_input = st.text_area(
            label="연구하고 싶은 주제나 질문을 입력하세요",
            placeholder="예: 오픈소스 LLM 트렌드 알려줘",
            height=100,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 심층 조사 시작",
                use_container_width=True,
                disabled=st.session_state.processing
            )
    
    # 프로세싱 시작
    if submitted and user_input.strip():
        st.session_state.processing = True
        st.session_state.user_question = user_input.strip()
        st.session_state.status_messages = []
        st.session_state.result = ""
        st.rerun()
    
    # 진행 상태 표시
    if st.session_state.processing:
        st.markdown("### 📊 진행 상태")
        
        # 상태 컨테이너
        status_container = st.empty()
        
        with status_container.container():
            st.markdown('<div class="status-container">', unsafe_allow_html=True)
            
            # 현재 진행 단계 표시
            if st.session_state.current_step:
                st.markdown(f"""
                <div class="processing-indicator">
                    <div class="spinner"></div>
                    <strong>{st.session_state.current_step}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # 진행 상태 메시지들
            if st.session_state.status_messages:
                st.markdown("**진행 기록:**")
                for msg in st.session_state.status_messages[-5:]:  # 최근 5개만 표시
                    st.markdown(f'<div class="status-message">✓ {msg}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 실제 워크플로우 실행
        if not st.session_state.result:
            # 비동기 워크플로우 실행
            try:
                asyncio.run(execute_research_workflow(st.session_state.user_question))
            except Exception as e:
                logger.error(f"Failed to run research workflow: {e}")
                st.session_state.result = f"# 실행 오류\n\n워크플로우 실행에 실패했습니다: {str(e)}"
                add_status_message(f"❌ 실행 실패: {str(e)}")
            
            # 완료 처리
            st.session_state.processing = False
            st.session_state.current_step = "완료!"
            st.rerun()
    
    # 결과 표시
    if st.session_state.result and st.session_state.user_question:
        st.markdown("### 📋 조사 결과")
        
        # 사용자 질문 표시
        st.markdown(f"""
        <div class="user-question">
            <strong>📝 질문:</strong> {st.session_state.user_question}
        </div>
        """, unsafe_allow_html=True)
        
        # 결과 표시
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.result)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 새 검색 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 새로운 질문하기", use_container_width=True):
                # 상태 초기화
                st.session_state.processing = False
                st.session_state.current_step = ""
                st.session_state.status_messages = []
                st.session_state.result = ""
                st.session_state.user_question = ""
                st.rerun()
    
    # 사이드바 정보 (선택사항)
    with st.sidebar:
        st.markdown("### ℹ️ 시스템 정보")
        st.markdown("""
        **Deep Research Chatbot**
        - 멀티 에이전트 기반 심층 조사
        - 자동 검증 및 재검색 시스템
        - 신뢰할 수 있는 출처 기반 응답
        
        **기능:**
        - 질문 자동 확장
        - 병렬 웹 검색 (DuckDuckGo + Tavily)
        - 스마트 문서 요약
        - 품질 검증 및 재검색
        """)
        
        st.markdown("---")
        st.markdown("**설정 정보:**")
        st.markdown(f"- LLM 제공자: {settings.LLM_PROVIDER}")
        st.markdown(f"- 채팅 모델: {settings.chat_model}")
        st.markdown(f"- 최대 검색 결과: {settings.MAX_SEARCH_RESULTS}")
        st.markdown(f"- LangSmith: {'활성화' if settings.LANGCHAIN_TRACING_V2 else '비활성화'}")
        
        # 실행 통계 표시 (있는 경우)
        if hasattr(st.session_state, 'execution_stats') and st.session_state.execution_stats:
            st.markdown("---")
            st.markdown("**실행 통계:**")
            stats = st.session_state.execution_stats
            st.markdown(f"- 하위 쿼리: {stats.get('sub_queries_count', 0)}개")
            st.markdown(f"- 수집 문서: {stats.get('documents_count', 0)}개")
            st.markdown(f"- 인사이트: {stats.get('insights_count', 0)}개")
            st.markdown(f"- 재시도: {stats.get('retry_count', 0)}회")
        
        st.markdown("---")
        st.markdown("**개발 상태:** MVP")
        st.markdown("**기술 스택:** LangChain + LangGraph + Streamlit")

if __name__ == "__main__":
    main()