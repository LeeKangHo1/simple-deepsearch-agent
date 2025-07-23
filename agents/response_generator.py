# agents/response_generator.py
"""
응답 생성 에이전트

인사이트와 문서 정보를 바탕으로 사용자에게 제공할 최종 마크다운 응답을 생성합니다.
구조화된 형식과 적절한 출처 표시를 통해 신뢰도 높은 정보를 제공합니다.

주요 기능:
- 마크다운 형식의 구조화된 응답 생성
- 섹션별 출처 표시 (현황 30%, 인사이트 50%, 전망 20%)
- 사용자 질문 기반 제목 생성
- 모든 인사이트 포함 보장
- 문서 출처 자동 추출 및 정리
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 모듈들
from models.data_models import Document, Insight, ResearchResponse
from models.state import ResearchState, StateManager
from services.llm_service import get_llm_service, LLMResponse
from utils.text_processing import clean_text, extract_keywords, truncate_text
from utils.validators import validate_response_structure
from utils.logger import create_agent_logger, log_execution_time

logger = logging.getLogger(__name__)
agent_logger = create_agent_logger("response_generator")

class ResponseGenerator:
    """
    응답 생성 에이전트 클래스
    
    인사이트, 문서 요약, 출처 정보를 종합하여 구조화된 마크다운 응답을 생성합니다.
    사용자 질문에 최적화된 형태로 정보를 구성하고 신뢰할 수 있는 출처를 제공합니다.
    
    응답 구조:
    - 제목: 사용자 질문 기반 자동 생성
    - 주요 현황 (30%): 핵심 내용과 출처
    - 주요 인사이트 (50%): 모든 인사이트 포함
    - 향후 전망 (20%): 예상 변화나 트렌드
    - 출처 목록: 참고한 모든 문서
    """
    
    def __init__(self, max_response_length: int = 8000):
        """
        응답 생성기 초기화
        
        Args:
            max_response_length: 최대 응답 길이 (기본: 8000자)
        """
        self.max_response_length = max_response_length
        self.llm_service = get_llm_service()
        
        # 섹션별 비중 설정
        self.section_ratios = {
            "현황": 0.30,    # 30%
            "인사이트": 0.50,  # 50%
            "전망": 0.20     # 20%
        }
        
        # 통계 추적
        self.total_responses_generated = 0
        self.total_insights_included = 0
        self.total_sources_processed = 0
    
    @log_execution_time
    async def generate_response(
        self,
        insights: List[str],
        summaries: List[str],
        documents: List[Dict[str, Any]],
        user_question: str = ""
    ) -> ResearchResponse:
        """
        최종 마크다운 응답 생성
        
        Args:
            insights: 인사이트 리스트
            summaries: 문서 요약 리스트
            documents: 원본 문서 정보 리스트
            user_question: 사용자 질문
            
        Returns:
            ResearchResponse: 생성된 응답 객체
        """
        if not insights:
            logger.warning("응답 생성할 인사이트가 없습니다.")
            return self._create_empty_response()
        
        agent_logger.start_step(f"응답 생성 시작 ({len(insights)}개 인사이트)")
        
        try:
            # 1단계: 출처 정보 처리
            sources_info = self._process_sources(documents)
            
            # 2단계: LLM을 통한 응답 생성
            llm_response = await self._generate_markdown_response(
                insights, summaries, sources_info, user_question
            )
            
            if not llm_response.success:
                agent_logger.end_step("응답 생성", False, f"LLM 호출 실패: {llm_response.error_message}")
                return self._create_empty_response()
            
            # 3단계: 응답 후처리 및 검증
            final_response = self._post_process_response(
                llm_response.content, sources_info, insights
            )
            
            # 4단계: ResearchResponse 객체 생성
            research_response = ResearchResponse(
                markdown_content=final_response,
                sources=sources_info,
                insights_count=len(insights),
                documents_used=len(documents),
                word_count=len(final_response)
            )
            
            # 5단계: 통계 업데이트
            self.total_responses_generated += 1
            self.total_insights_included += len(insights)
            self.total_sources_processed += len(sources_info)
            
            agent_logger.end_step(
                "응답 생성 완료",
                True,
                f"응답 길이: {len(final_response)}자, 출처: {len(sources_info)}개"
            )
            
            return research_response
            
        except Exception as e:
            agent_logger.end_step("응답 생성", False, f"오류 발생: {str(e)}")
            logger.error(f"응답 생성 중 오류 발생: {e}")
            return self._create_empty_response()
    
    def _process_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        문서들에서 출처 정보 추출 및 정리
        
        Args:
            documents: 원본 문서 정보 리스트
            
        Returns:
            List[Dict[str, str]]: 정리된 출처 정보 리스트
        """
        sources = []
        seen_domains = set()
        
        for doc in documents:
            try:
                title = doc.get("title", "").strip()
                url = doc.get("url", "").strip()
                
                if not title or not url:
                    continue
                
                # 도메인 추출
                domain = self._extract_domain(url)
                
                # 중복 도메인 체크 (같은 사이트에서 여러 문서가 온 경우 대표 1개만)
                if domain in seen_domains:
                    continue
                
                seen_domains.add(domain)
                
                # 제목 길이 조정
                display_title = truncate_text(title, 60, "...")
                
                source_info = {
                    "title": display_title,
                    "url": url,
                    "domain": domain
                }
                
                sources.append(source_info)
                
            except Exception as e:
                logger.warning(f"출처 처리 실패: {e}")
                continue
        
        logger.info(f"출처 처리 완료: {len(sources)}개 고유 출처")
        return sources
    
    def _extract_domain(self, url: str) -> str:
        """
        URL에서 도메인 추출
        
        Args:
            url: 추출할 URL
            
        Returns:
            str: 도메인명
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # www. 제거
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # 포트 번호 제거
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain if domain else 'unknown'
        except Exception:
            return 'unknown'
    
    async def _generate_markdown_response(
        self,
        insights: List[str],
        summaries: List[str],
        sources_info: List[Dict[str, str]],
        user_question: str
    ) -> LLMResponse:
        """
        LLM을 통한 마크다운 응답 생성
        
        Args:
            insights: 인사이트 리스트
            summaries: 요약 리스트
            sources_info: 출처 정보
            user_question: 사용자 질문
            
        Returns:
            LLMResponse: LLM 응답
        """
        # 인사이트를 모두 포함한 텍스트 생성
        all_insights_text = "\n".join([f"- {insight}" for insight in insights])
        
        # 요약 텍스트 생성
        summaries_text = "\n".join([f"• {summary}" for summary in summaries])
        
        # 출처 도메인 리스트 생성
        source_domains = [source["domain"] for source in sources_info]
        sources_text = ", ".join(source_domains[:10])  # 최대 10개까지
        
        # 사용자 질문 기반 제목 힌트
        title_hint = user_question if user_question.strip() else "연구 결과"
        
        # 응답 생성 프롬프트
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 리서치 보고서 작성 전문가입니다. 주어진 정보를 바탕으로 구조화된 마크다운 응답을 작성해주세요.

작성 규칙:
1. 마크다운 형식으로 구조화 (제목, 소제목, 불릿 포인트 등)
2. 사용자 질문에 직접적으로 답변하는 제목 생성
3. 반드시 다음 구조를 따를 것:
   - # [제목] (사용자 질문 기반)
   - ## 주요 현황 *(출처: 도메인들)*
   - ## 주요 인사이트 *(출처: 도메인들)*  
   - ## 향후 전망 *(출처: 도메인들)*
   - --- 출처 목록
4. 섹션별 비중: 현황 30%, 인사이트 50%, 전망 20%
5. 모든 인사이트를 빠짐없이 포함해야 함
6. 각 섹션 제목 옆에 출처 표시: *(출처: example.com, news.com)*
7. 최대 {self.max_response_length}자 이내
8. 객관적이고 정확한 정보만 포함

**중요**: 제공된 모든 인사이트를 반드시 '주요 인사이트' 섹션에 포함해야 합니다."""),
            ("user", """사용자 질문: {question}

모든 인사이트 (반드시 모두 포함):
{insights}

문서 요약들:
{summaries}

출처 도메인들: {sources}

위 정보를 바탕으로 구조화된 마크다운 응답을 생성해주세요.""")
        ])
        
        return await self.llm_service._call_llm(
            prompt_template=prompt_template,
            input_variables={
                "question": title_hint,
                "insights": all_insights_text,
                "summaries": summaries_text,
                "sources": sources_text
            },
            agent_type="response_generator",
            output_parser=self.llm_service.str_parser,
            expected_type=str
        )
    
    def _post_process_response(
        self,
        raw_response: str,
        sources_info: List[Dict[str, str]],
        original_insights: List[str]
    ) -> str:
        """
        응답 후처리 및 품질 개선
        
        Args:
            raw_response: LLM이 생성한 원본 응답
            sources_info: 출처 정보 리스트
            original_insights: 원본 인사이트 리스트
            
        Returns:
            str: 후처리된 최종 응답
        """
        try:
            # 1단계: 기본 정제
            cleaned_response = clean_text(raw_response, remove_extra_whitespace=True)
            
            # 2단계: 마크다운 구조 검증 및 보정
            cleaned_response = self._ensure_markdown_structure(cleaned_response)
            
            # 3단계: 인사이트 누락 검증 및 보완
            cleaned_response = self._ensure_all_insights_included(
                cleaned_response, original_insights
            )
            
            # 4단계: 출처 정보 추가/보완
            cleaned_response = self._ensure_sources_section(cleaned_response, sources_info)
            
            # 5단계: 길이 조정
            if len(cleaned_response) > self.max_response_length:
                cleaned_response = self._truncate_response_smartly(cleaned_response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"응답 후처리 실패: {e}")
            return raw_response  # 실패 시 원본 반환
    
    def _ensure_markdown_structure(self, response: str) -> str:
        """
        마크다운 구조 검증 및 보정
        
        Args:
            response: 검증할 응답
            
        Returns:
            str: 구조가 보정된 응답
        """
        lines = response.split('\n')
        corrected_lines = []
        
        has_main_title = False
        required_sections = ['주요 현황', '주요 인사이트', '향후 전망']
        found_sections = set()
        
        for line in lines:
            line = line.strip()
            
            # 메인 제목 확인
            if line.startswith('# ') and not has_main_title:
                has_main_title = True
                corrected_lines.append(line)
                continue
            
            # 섹션 제목 확인
            if line.startswith('## '):
                for section in required_sections:
                    if section in line:
                        found_sections.add(section)
                        break
                corrected_lines.append(line)
                continue
            
            corrected_lines.append(line)
        
        # 누락된 섹션 추가
        missing_sections = set(required_sections) - found_sections
        if missing_sections:
            logger.warning(f"누락된 섹션 감지: {missing_sections}")
            # 기본 섹션 구조 추가 (간단한 형태로)
            for section in missing_sections:
                corrected_lines.append(f"\n## {section}")
                corrected_lines.append("정보를 분석 중입니다.")
        
        return '\n'.join(corrected_lines)
    
    def _ensure_all_insights_included(self, response: str, original_insights: List[str]) -> str:
        """
        모든 인사이트가 포함되었는지 검증하고 누락된 것 추가
        
        Args:
            response: 검증할 응답
            original_insights: 원본 인사이트 리스트
            
        Returns:
            str: 인사이트가 보완된 응답
        """
        # 현재 응답에서 인사이트 섹션 찾기
        insight_pattern = r'## 주요 인사이트.*?\n(.*?)(?=\n## |$)'
        match = re.search(insight_pattern, response, re.DOTALL)
        
        if not match:
            logger.warning("인사이트 섹션을 찾을 수 없습니다.")
            return response
        
        current_insight_section = match.group(1)
        
        # 누락된 인사이트 찾기
        missing_insights = []
        for insight in original_insights:
            # 인사이트의 핵심 키워드로 포함 여부 확인
            insight_keywords = extract_keywords(insight, max_keywords=3)
            
            # 키워드 중 일부라도 현재 섹션에 있는지 확인
            if insight_keywords:
                keyword_found = any(
                    keyword.lower() in current_insight_section.lower() 
                    for keyword in insight_keywords
                )
                
                if not keyword_found:
                    missing_insights.append(insight)
        
        # 누락된 인사이트 추가
        if missing_insights:
            logger.info(f"누락된 인사이트 {len(missing_insights)}개 추가")
            
            additional_insights = '\n'.join([f"- {insight}" for insight in missing_insights])
            
            # 인사이트 섹션 교체
            new_insight_section = current_insight_section.rstrip() + '\n' + additional_insights
            response = re.sub(
                insight_pattern,
                f'## 주요 인사이트 *(출처: 참고 문서들)*\n{new_insight_section}',
                response,
                flags=re.DOTALL
            )
        
        return response
    
    def _ensure_sources_section(self, response: str, sources_info: List[Dict[str, str]]) -> str:
        """
        출처 섹션 확인 및 추가
        
        Args:
            response: 확인할 응답
            sources_info: 출처 정보 리스트
            
        Returns:
            str: 출처 섹션이 보완된 응답
        """
        # 이미 출처 섹션이 있는지 확인
        if '---' in response and '출처' in response:
            return response
        
        # 출처 섹션 생성
        sources_section = "\n\n---\n### 출처\n"
        for source in sources_info[:10]:  # 최대 10개까지
            sources_section += f"- [{source['title']}]({source['url']})\n"
        
        return response + sources_section
    
    def _truncate_response_smartly(self, response: str) -> str:
        """
        응답을 지능적으로 자르기 (섹션 구조 유지)
        
        Args:
            response: 자를 응답
            
        Returns:
            str: 길이가 조정된 응답
        """
        if len(response) <= self.max_response_length:
            return response
        
        # 섹션별로 분리
        sections = response.split('\n## ')
        
        if len(sections) <= 1:
            # 섹션이 없으면 단순 자르기
            return truncate_text(response, self.max_response_length, "\n\n...")
        
        # 각 섹션의 목표 길이 계산
        available_length = self.max_response_length - 200  # 여유분
        target_lengths = {}
        
        for i, section in enumerate(sections):
            if i == 0:  # 제목 부분
                target_lengths[i] = min(len(section), 200)
            elif '주요 현황' in section:
                target_lengths[i] = int(available_length * self.section_ratios["현황"])
            elif '주요 인사이트' in section:
                target_lengths[i] = int(available_length * self.section_ratios["인사이트"])
            elif '향후 전망' in section:
                target_lengths[i] = int(available_length * self.section_ratios["전망"])
            else:
                target_lengths[i] = min(len(section), 300)
        
        # 각 섹션을 목표 길이로 자르기
        truncated_sections = []
        for i, section in enumerate(sections):
            if i in target_lengths:
                max_len = target_lengths[i]
                if len(section) > max_len:
                    section = truncate_text(section, max_len, "...")
            
            truncated_sections.append(section)
        
        # 다시 조합
        result = truncated_sections[0]  # 제목 부분
        for section in truncated_sections[1:]:
            result += '\n## ' + section
        
        return result
    
    def _create_empty_response(self) -> ResearchResponse:
        """
        빈 응답 객체 생성 (오류 시 사용)
        
        Returns:
            ResearchResponse: 기본 응답 객체
        """
        return ResearchResponse(
            markdown_content="# 응답 생성 실패\n\n죄송합니다. 응답을 생성할 수 없습니다.",
            sources=[],
            insights_count=0,
            documents_used=0,
            word_count=0
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            "total_responses_generated": self.total_responses_generated,
            "total_insights_included": self.total_insights_included,
            "total_sources_processed": self.total_sources_processed,
            "max_response_length": self.max_response_length,
            "section_ratios": self.section_ratios,
            "avg_insights_per_response": round(
                self.total_insights_included / max(1, self.total_responses_generated), 2
            )
        }


# LangGraph 노드 함수
async def generate_response_node(state: ResearchState) -> ResearchState:
    """
    응답 생성 LangGraph 노드
    
    State의 insights, summaries, documents를 받아서 최종 마크다운 응답을 생성합니다.
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 최종 응답이 추가된 업데이트된 상태
    """
    logger.info("=== 응답 생성 노드 시작 ===")
    
    try:
        # 현재 단계 설정
        state = StateManager.set_step(state, "generating_response", "📝 최종 응답 생성 중...")
        
        # 필수 데이터 확인
        insights = state.get("insights", [])
        summaries = state.get("summaries", [])
        documents = state.get("documents", [])
        user_question = state.get("user_input", "")
        
        if not insights:
            error_msg = "응답 생성할 인사이트가 없습니다."
            logger.warning(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 응답 생성기 생성 및 실행
        generator = ResponseGenerator()
        research_response = await generator.generate_response(
            insights=insights,
            summaries=summaries,
            documents=documents,
            user_question=user_question
        )
        
        if not research_response.markdown_content:
            error_msg = "마크다운 응답 생성에 실패했습니다."
            logger.error(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 응답 구조 검증
        is_valid, validation_errors = validate_response_structure(research_response.markdown_content)
        if not is_valid:
            logger.warning(f"응답 구조 검증 실패: {validation_errors}")
            # 검증 실패해도 계속 진행 (검증 에이전트에서 다시 처리)
        
        # 상태 업데이트
        new_state = state.copy()
        new_state["markdown_answer"] = research_response.markdown_content
        
        # 처리 통계 로깅
        stats = generator.get_processing_stats()
        logger.info(f"응답 생성 완료: {stats}")
        
        new_state = StateManager.add_log(
            new_state,
            f"📝 최종 응답 생성 완료: {research_response.word_count}자, "
            f"{research_response.insights_count}개 인사이트, {research_response.documents_used}개 문서 활용"
        )
        
        logger.info("=== 응답 생성 노드 완료 ===")
        return new_state
        
    except Exception as e:
        error_msg = f"응답 생성 노드에서 오류 발생: {str(e)}"
        logger.error(error_msg)
        return StateManager.set_error(state, error_msg)


# 유틸리티 함수들
def create_response_generator(max_length: int = 8000) -> ResponseGenerator:
    """
    응답 생성기 인스턴스 생성 헬퍼 함수
    
    Args:
        max_length: 최대 응답 길이
        
    Returns:
        ResponseGenerator: 설정된 응답 생성기 인스턴스
    """
    return ResponseGenerator(max_response_length=max_length)

