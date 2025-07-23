import pytest
from agents.validator import ResponseValidator, validate_response_node
from models.data_models import ValidationResult
from models.state import ResearchState
import textwrap

@pytest.mark.asyncio
async def test_validate_response_node_success():
    state = ResearchState({
        "markdown_answer": textwrap.dedent("""
        # AI 기술 트렌드

        ## 주요 현황 *(출처: aiweekly.com)*
        - 다양한 기업이 AI 기술을 도입 중입니다.

        ## 주요 인사이트 *(출처: techdaily.com)*
        - AI 윤리와 규제 중요성이 부각되고 있습니다.
        - 오픈소스 LLM이 상용 모델을 위협하고 있습니다.

        ## 향후 전망 *(출처: aiweekly.com)*
        - AI 거버넌스 체계 강화가 예상됩니다.

        ---
        ### 출처
        - [AI 주간 리포트](https://aiweekly.com/report)
        - [기술 뉴스](https://techdaily.com/article)
        """),
        "user_input": "AI 기술 트렌드는 어떤가요?",
        "insights": [
            "AI 윤리와 규제 중요성이 부각되고 있습니다.",
            "오픈소스 LLM이 상용 모델을 위협하고 있습니다."
        ],
        "documents": [
            {"title": "AI 주간 리포트", "url": "https://aiweekly.com/report"},
            {"title": "기술 뉴스", "url": "https://techdaily.com/article"}
        ],
        "step_timestamps": {},
        "logs": []
    })

    new_state = await validate_response_node(state)

    assert new_state.get("is_valid") is True
    assert "✅ 응답 검증 통과" in "".join(new_state.get("logs", []))
    assert new_state.get("validation_feedback") is not None

@pytest.mark.asyncio
async def test_response_validator_detects_missing_sections():
    validator = ResponseValidator()
    response_content = """
    # AI 기술 트렌드

    오픈소스 LLM이 급부상하고 있습니다.
    """
    original_insights = ["오픈소스 LLM이 급부상하고 있습니다."]
    original_sources = [
        {"title": "뉴스", "url": "https://example.com", "domain": "example.com"}
    ]

    result: ValidationResult = await validator.validate_response(
        response_content,
        user_question="최근 AI 기술 동향은?",
        original_insights=original_insights,
        original_sources=original_sources
    )

    assert result.is_valid is False
    assert any("섹션" in issue for issue in result.issues)
    assert result.confidence_score < 1.0
    assert result.feedback is not None
