import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from dataclasses import asdict

from agents.doc_summarizer import DocumentSummarizer, summarize_documents_node
from models.data_models import Document, DocumentSummary, SearchEngine
from models.state import ResearchState

pytestmark = pytest.mark.asyncio

@pytest.fixture
def summarizer():
    return DocumentSummarizer(batch_size=2)

@pytest.fixture
def sample_documents():
    return [
        Document(
            title="Valid",
            url="https://test.com",
            content="요약할 수 있는 충분한 내용입니다. " * 3,  # 63자 이상으로 증가
            source=SearchEngine.TAVILY
        ),

        Document(
            title="Doc2",
            url="https://example.com/2",
            content="LLaMA2 모델의 성능과 사용 사례에 대해 설명합니다."  * 3,
            source=SearchEngine.TAVILY
        ),
    ]

# ✅ 전처리 필터링 테스트 (유효한 문서 1개 통과)
async def test_preprocess_documents_filters_invalid(summarizer):
    docs = [
        Document(title="", url="", content="", source=SearchEngine.DUCKDUCKGO),  # invalid
        Document(title="Valid", url="https://test.com", content="요약할 수 있는 충분한 내용입니다." * 3, source=SearchEngine.TAVILY)
    ]
    processed = summarizer._preprocess_documents(docs)
    assert len(processed) == 1
    assert processed[0].title == "Valid"

# ✅ 요약 성공 테스트 (LLM 응답 mock)
@patch("agents.doc_summarizer.DocumentSummarizer._generate_batch_summary", new_callable=AsyncMock)
async def test_summarize_documents_success(mock_llm_call, summarizer, sample_documents):
    mock_llm_call.return_value.success = True
    mock_llm_call.return_value.content = (
        "===SUMMARY_1===\n오픈소스 LLM 동향 요약입니다.\n"
        "===SUMMARY_2===\nLLaMA2의 주요 특징 요약입니다.\n"
    )

    summaries = await summarizer.summarize_documents(sample_documents, user_question="최근 LLM 트렌드")
    assert len(summaries) == 2
    assert all(isinstance(summary, DocumentSummary) for summary in summaries)

# ✅ LangGraph 요약 노드 성공 테스트
@patch("agents.doc_summarizer.DocumentSummarizer.summarize_documents", new_callable=AsyncMock)
async def test_summarize_documents_node_success(mock_summarize, sample_documents):
    mock_summarize.return_value = [
        DocumentSummary(
            document_hash="abc",
            summary="요약 A",
            key_points=["요약 A"],
            confidence_score=0.9,
            word_count=12
        )
    ]

    state = ResearchState({
        "user_input": "최근 LLM 동향은?",
        "documents": [asdict(doc) for doc in sample_documents],
        "step_timestamps": {},
        "logs": [],
    })

    new_state = await summarize_documents_node(state)
    assert "summaries" in new_state
    assert isinstance(new_state["summaries"], list)
    assert len(new_state["summaries"]) == 1
