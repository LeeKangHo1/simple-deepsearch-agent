# graph/graph_builder.py
from langgraph.graph import StateGraph
from models.state import ResearchState

# 에이전트 노드 임포트
from agents.question_analyzer import analyze_question_node
from agents.web_search import web_search_node
from agents.doc_summarizer import summarize_documents_node
from agents.insight_generator import generate_insights_node
from agents.response_generator import generate_response_node
from agents.validator import validate_response_node

def route_on_validation(state: ResearchState):
    if state.is_valid:
        return "final_output"
    elif state.retry_count >= 1:  # 1회 재시도 제한
        return "final_output"
    else:
        return "web_search"

def build_graph():
    """LangGraph 상태 기반 워크플로우 생성"""
    workflow = StateGraph(ResearchState)

    # 노드 정의
    workflow.add_node("input", lambda state: state)
    workflow.add_node("question_analyzer", analyze_question_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("doc_summarizer", summarize_documents_node)
    workflow.add_node("insight_generator", generate_insights_node)
    workflow.add_node("response_generator", generate_response_node)
    workflow.add_node("response_validator", validate_response_node)
    workflow.add_node("final_output", lambda state: state)

    # 순차 흐름 정의
    workflow.set_entry_point("input")
    workflow.add_edge("input", "question_analyzer")
    workflow.add_edge("question_analyzer", "web_search")
    workflow.add_edge("web_search", "doc_summarizer")
    workflow.add_edge("doc_summarizer", "insight_generator")
    workflow.add_edge("insight_generator", "response_generator")
    workflow.add_edge("response_generator", "response_validator")

    # 조건부 라우팅 - 검증 결과에 따라 분기
    workflow.add_conditional_edges(
        "response_validator", 
        route_on_validation,
        {
            "final_output": "final_output",
            "web_search": "web_search"  # 재검색 루프
        }
    )

    return workflow.compile()