# tests/test_insight_generator.py

import pytest
from unittest.mock import AsyncMock, patch
from agents.insight_generator import InsightGenerator, generate_insights_node
from models.data_models import DocumentSummary, Insight
from models.state import ResearchState

@pytest.fixture
def sample_summaries():
    return [
        DocumentSummary(
            document_hash="doc1",
            summary="오픈소스 LLM의 사용이 증가하고 있다. 많은 기업들이 비용과 유연성 때문에 오픈소스를 선택하고 있다.",
            confidence_score=0.8,
            word_count=40
        ),
        DocumentSummary(
            document_hash="doc2",
            summary="상업용 LLM과 오픈소스의 성능 격차가 줄어들고 있으며, 커뮤니티 기반 개발이 활발하다.",
            confidence_score=0.7,
            word_count=35
        )
    ]

@pytest.mark.asyncio
@patch("agents.insight_generator.InsightGenerator._generate_insights_llm", new_callable=AsyncMock)
async def test_generate_insights_success(mock_llm_call, sample_summaries):
    # 가짜 응답 세팅
    mock_llm_call.return_value.success = True
    mock_llm_call.return_value.content = (
        "===INSIGHT_1===\n오픈소스 LLM은 비용과 커스터마이징 유연성으로 인해 채택이 증가하고 있다.\n"
        "===INSIGHT_2===\n상업용 모델과의 성능 격차가 줄어들면서 오픈소스가 시장 주류가 될 가능성이 높다."
    )

    generator = InsightGenerator()
    insights = await generator.generate_insights(sample_summaries, user_question="오픈소스 LLM 트렌드")

    assert isinstance(insights, list)
    assert all(isinstance(insight, Insight) for insight in insights)
    assert len(insights) == 2

@pytest.mark.asyncio
@patch("agents.insight_generator.InsightGenerator.generate_insights", new_callable=AsyncMock)
async def test_generate_insights_node_success(mock_generate, sample_summaries):
    mock_generate.return_value = [
        Insight(
            content="오픈소스가 상업용을 대체할 수 있는 가능성이 증가하고 있다.",
            category="trend",
            confidence_score=0.85,
            supporting_documents=["doc1", "doc2"]
        )
    ]

    state = ResearchState({
        "user_input": "오픈소스 LLM 트렌드는?",
        "summaries": [s.summary for s in sample_summaries],
        "step_timestamps": {},
        "logs": []
    })

    new_state = await generate_insights_node(state)

    assert "insights" in new_state
    assert isinstance(new_state["insights"], list)
    assert len(new_state["insights"]) == 1
    assert "오픈소스" in new_state["insights"][0]
