import pytest
from agents.response_generator import ResponseGenerator, generate_response_node
from models.state import ResearchState
from models.data_models import ResearchResponse
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
@patch("agents.response_generator.ResponseGenerator._generate_markdown_response", new_callable=AsyncMock)
async def test_generate_response_success(mock_llm_call):
    # given
    test_insights = [
        "AI 기술의 발전으로 기업들의 디지털 전환이 가속화되고 있다.",
        "오픈소스 LLM의 성능 향상으로 기업 도입이 증가하고 있다.",
        "AI 윤리와 규제에 대한 관심이 높아지고 있다."
    ]

    test_summaries = [
        "AI 기술이 빠르게 발전하면서 다양한 산업 분야에 적용되고 있다.",
        "기업들이 비용 절감과 효율성 향상을 위해 AI 도입을 검토하고 있다."
    ]

    test_documents = [
        {
            "title": "AI 기술 동향 2024",
            "url": "https://example.com/ai-trends",
            "content": "..."
        },
        {
            "title": "기업 AI 도입 현황",
            "url": "https://techreport.com/enterprise-ai",
            "content": "..."
        }
    ]

    mock_llm_call.return_value.success = True
    mock_llm_call.return_value.content = "# AI 기술 동향 분석\n## 주요 현황...\n## 주요 인사이트...\n## 향후 전망..."

    generator = ResponseGenerator()
    response = await generator.generate_response(
        insights=test_insights,
        summaries=test_summaries,
        documents=test_documents,
        user_question="AI 기술 동향은 어떻게 되나요?"
    )

    # then
    assert isinstance(response, ResearchResponse)
    assert response.markdown_content.startswith("# ")
    assert response.insights_count == len(test_insights)
    assert response.documents_used == len(test_documents)
    assert response.word_count > 0


@pytest.mark.asyncio
@patch("agents.response_generator.ResponseGenerator.generate_response", new_callable=AsyncMock)
async def test_generate_response_node_success(mock_generate):
    mock_generate.return_value = ResearchResponse(
        markdown_content="# AI 기술 보고서",
        sources=[{"title": "Test", "url": "https://a.com", "domain": "a.com"}],
        insights_count=2,
        documents_used=1,
        word_count=1200
    )

    state = ResearchState({
        "insights": ["인사이트 1", "인사이트 2"],
        "summaries": ["요약 1"],
        "documents": [{"title": "Doc", "url": "https://a.com", "content": "내용"}],
        "user_input": "AI에 대해 알려줘",
        "step_timestamps": {},
        "logs": [],
    })

    new_state = await generate_response_node(state)

    assert new_state.get("markdown_answer", "").startswith("# AI 기술 보고서")
    assert "최종 응답 생성 완료" in new_state.get("logs")[-1]
