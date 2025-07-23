# tests/test_web_search.py

import pytest
from unittest.mock import AsyncMock
from agents.web_search import WebSearchAgent, reset_web_search_agent
from models.data_models import Document, SearchResult
from models.state import ResearchState

# 🧪 비동기 테스트 설정
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="function")
def agent(monkeypatch):
    reset_web_search_agent()
    agent = WebSearchAgent(
        max_results_per_query=3,
        max_total_results=5,
        similarity_threshold=0.9
    )

    # ✅ SearchResult mock 반환
    monkeypatch.setattr(agent.search_service, "search_all_engines", AsyncMock(return_value=[
        SearchResult(
            title="테스트 문서",
            url="https://example.com",
            content="이것은 테스트 문서입니다.",
            engine="duckduckgo",
            score=0.95,
            published_date="2024-01-01"
        )
    ]))
    return agent

async def test_search_multiple_queries_valid(agent):
    queries = ["GPT-4o 성능", "Meta LLaMA"]
    original_question = "오픈소스 LLM 트렌드"

    results = await agent.search_multiple_queries(queries, original_question)

    assert isinstance(results, list)
    assert all(isinstance(doc, Document) for doc in results)
    assert len(results) <= 5
    for doc in results:
        assert doc.title and doc.url and doc.content

async def test_search_multiple_queries_invalid(agent):
    with pytest.raises(ValueError):
        await agent.search_multiple_queries(["", None], original_question="LLM")

async def test_process_state_success(agent):
    state = ResearchState({
        "user_input": "오픈소스 LLM의 최신 동향은?",
        "sub_queries": ["Mistral 모델 성능", "LLaMA2 사용 사례"],
        "step_timestamps": {},
        "logs": [],
    })

    new_state = await agent.process_state(state)

    assert "documents" in new_state
    assert isinstance(new_state["documents"], list)
    assert any("📊 검색 엔진별 결과" in log for log in new_state["logs"])

async def test_process_state_missing_queries(agent):
    state = ResearchState({
        "user_input": "질문 없음",
        "sub_queries": [],  # ✅ 실제로 빈 쿼리로 에러 유도
        "step_timestamps": {},
        "logs": [],
    })

    new_state = await agent.process_state(state)

    assert new_state.get("has_error") is True
    assert new_state.get("error_message", "").startswith("웹 검색 실패")


def test_statistics_tracking(agent):
    agent._update_statistics(query_count=2, raw_docs=10, final_docs=6, processing_time=3.2)
    stats = agent.get_statistics()

    assert stats["total_searches"] == 1
    assert stats["total_documents_found"] == 10
    assert stats["total_documents_after_dedup"] == 6
    assert stats["avg_search_time"] > 0

def test_reset_statistics(agent):
    agent._update_statistics(1, 5, 3, 1.5)
    agent.reset_statistics()
    stats = agent.get_statistics()

    assert stats["total_searches"] == 0
    assert stats["total_documents_found"] == 0
    assert stats["avg_search_time"] == 0.0
