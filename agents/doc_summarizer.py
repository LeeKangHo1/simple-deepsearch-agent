# agents/doc_summarizer.py
"""
문서 요약 에이전트

검색으로 수집된 문서들을 배치 단위로 처리하여 핵심 내용만 추출한 요약문을 생성합니다.
LLM을 활용한 지능적 요약과 함께 오류 처리 및 품질 검증 기능을 포함합니다.

주요 기능:
- 배치 처리를 통한 효율적인 다중 문서 요약
- JSON 파싱 실패 시 구분자 기반 폴백 처리
- 문서 길이에 따른 적응적 요약
- 품질 검증 및 오류 복구
"""

import asyncio
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 모듈들
from models.data_models import Document, DocumentSummary
from models.state import ResearchState, StateManager
from services.llm_service import get_llm_service, LLMResponse
from utils.text_processing import clean_text, is_valid_text, truncate_text
from utils.validators import validate_document
from utils.logger import create_agent_logger, log_execution_time

logger = logging.getLogger(__name__)
agent_logger = create_agent_logger("doc_summarizer")

class DocumentSummarizer:
    """
    문서 요약 에이전트 클래스
    
    여러 문서를 배치 단위로 처리하여 효율적으로 요약을 생성합니다.
    LLM의 응답 품질을 보장하기 위한 다양한 오류 처리 전략을 포함합니다.
    
    처리 흐름:
    1. 문서 전처리 및 유효성 검증
    2. 배치 단위로 그룹핑 (기본 5개씩)
    3. LLM을 통한 배치 요약 생성
    4. JSON 파싱 및 폴백 처리
    5. 품질 검증 및 후처리
    """
    
    def __init__(self, batch_size: int = 5, max_content_length: int = 1500, max_summary_length: int = 200):
        """
        문서 요약기 초기화
        
        Args:
            batch_size: 배치 크기 (기본: 5개)
            max_content_length: 문서별 최대 내용 길이 (기본: 1500자)
            max_summary_length: 요약문 최대 길이 (기본: 200자)
        """
        self.batch_size = batch_size
        self.max_content_length = max_content_length
        self.max_summary_length = max_summary_length
        self.llm_service = get_llm_service()
        
        # 통계 추적
        self.total_documents_processed = 0
        self.total_summaries_generated = 0
        self.failed_documents = 0
        self.batch_retry_count = 0
    
    @log_execution_time
    async def summarize_documents(self, documents: List[Document], user_question: str = "") -> List[DocumentSummary]:
        """
        문서 리스트를 배치 처리로 요약
        
        Args:
            documents: 요약할 문서 리스트
            user_question: 사용자 원본 질문 (요약 방향성 제공용)
            
        Returns:
            List[DocumentSummary]: 생성된 문서 요약 리스트
        """
        if not documents:
            logger.warning("요약할 문서가 없습니다.")
            return []
        
        agent_logger.start_step(f"문서 요약 시작 ({len(documents)}개 문서)")
        
        try:
            # 1단계: 문서 전처리 및 필터링
            valid_documents = self._preprocess_documents(documents)
            
            if not valid_documents:
                agent_logger.end_step("문서 전처리", False, "유효한 문서가 없음")
                return []
            
            # 2단계: 배치 처리로 요약 생성
            summaries = await self._process_documents_in_batches(valid_documents, user_question)
            
            # 3단계: 통계 업데이트
            self.total_documents_processed += len(valid_documents)
            self.total_summaries_generated += len(summaries)
            
            agent_logger.end_step(
                "문서 요약 완료", 
                True, 
                f"{len(summaries)}/{len(valid_documents)} 요약 생성 완료"
            )
            
            return summaries
            
        except Exception as e:
            agent_logger.end_step("문서 요약", False, f"오류 발생: {str(e)}")
            logger.error(f"문서 요약 중 오류 발생: {e}")
            return []
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 전처리 및 유효성 검증
        
        Args:
            documents: 전처리할 문서 리스트
            
        Returns:
            List[Document]: 유효성 검증을 통과한 문서 리스트
        """
        valid_documents = []
        filtered_count = 0
        
        for doc in documents:
            try:
                # 문서 유효성 검증
                is_valid, errors = validate_document(doc)
                if not is_valid:
                    logger.debug(f"문서 검증 실패 '{doc.title}': {errors}")
                    filtered_count += 1
                    continue
                
                # 내용 길이 검증 (너무 짧거나 빈 문서 제외)
                cleaned_content = clean_text(doc.content)
                if len(cleaned_content.strip()) < 20:  # 20자 미만은 제외
                    logger.debug(f"문서 내용이 너무 짧음: '{doc.title}' ({len(cleaned_content)}자)")
                    filtered_count += 1
                    continue
                
                # 문서 내용 길이 조정 (1500자로 제한)
                if len(doc.content) > self.max_content_length:
                    doc.content = doc.content[:self.max_content_length]
                
                valid_documents.append(doc)
                
            except Exception as e:
                logger.warning(f"문서 전처리 중 오류: {doc.title} - {e}")
                filtered_count += 1
        
        if filtered_count > 0:
            logger.info(f"문서 전처리 완료: {len(valid_documents)}개 유효, {filtered_count}개 필터링됨")
        
        return valid_documents
    
    async def _process_documents_in_batches(self, documents: List[Document], user_question: str) -> List[DocumentSummary]:
        """
        문서들을 배치 단위로 처리하여 요약 생성
        
        Args:
            documents: 처리할 문서 리스트
            user_question: 사용자 질문
            
        Returns:
            List[DocumentSummary]: 생성된 요약 리스트
        """
        all_summaries = []
        
        # 문서들을 배치 크기만큼 그룹핑
        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        
        logger.info(f"배치 처리 시작: {len(batches)}개 배치, 배치당 최대 {self.batch_size}개 문서")
        
        for batch_idx, batch in enumerate(batches):
            try:
                logger.debug(f"배치 {batch_idx + 1}/{len(batches)} 처리 중... ({len(batch)}개 문서)")
                
                # 배치 요약 시도
                batch_summaries = await self._summarize_batch(batch, user_question, batch_idx)
                all_summaries.extend(batch_summaries)
                
            except Exception as e:
                logger.error(f"배치 {batch_idx + 1} 처리 실패: {e}")
                # 실패한 배치는 원본 내용 일부를 사용한 기본 요약으로 대체
                fallback_summaries = self._create_fallback_summaries(batch)
                all_summaries.extend(fallback_summaries)
        
        return all_summaries
    
    async def _summarize_batch(self, batch: List[Document], user_question: str, batch_idx: int) -> List[DocumentSummary]:
        """
        단일 배치의 문서들을 요약
        
        Args:
            batch: 요약할 문서 배치
            user_question: 사용자 질문
            batch_idx: 배치 인덱스
            
        Returns:
            List[DocumentSummary]: 배치 요약 결과
        """
        max_retries = 1  # 재시도 1번만
        
        for attempt in range(max_retries + 1):
            try:
                # LLM을 통한 배치 요약 생성
                llm_response = await self._generate_batch_summary(batch, user_question)
                
                if not llm_response.success:
                    if attempt < max_retries:
                        logger.warning(f"배치 {batch_idx + 1} LLM 호출 실패, 재시도 중...")
                        self.batch_retry_count += 1
                        continue
                    else:
                        raise Exception(f"LLM 호출 실패: {llm_response.error_message}")
                
                # 응답 파싱 및 요약 객체 생성
                summaries = self._parse_batch_response(llm_response.content, batch)
                
                if len(summaries) > 0:
                    logger.debug(f"배치 {batch_idx + 1} 처리 완료: {len(summaries)}/{len(batch)} 요약 생성")
                    return summaries
                else:
                    if attempt < max_retries:
                        logger.warning(f"배치 {batch_idx + 1} 파싱 실패, 재시도 중...")
                        continue
                    else:
                        raise Exception("요약 파싱에 실패했습니다.")
                        
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"배치 {batch_idx + 1} 처리 실패 (시도 {attempt + 1}), 재시도: {e}")
                    await asyncio.sleep(1)  # 재시도 전 대기
                    continue
                else:
                    logger.error(f"배치 {batch_idx + 1} 최종 실패: {e}")
                    # 폴백: 성공한 부분만 사용하고 나머지는 원본 내용 일부 사용
                    return self._create_fallback_summaries(batch)
        
        return []
    
    async def _generate_batch_summary(self, batch: List[Document], user_question: str) -> LLMResponse:
        """
        LLM을 통해 배치 요약 생성
        
        Args:
            batch: 요약할 문서 배치
            user_question: 사용자 질문
            
        Returns:
            LLMResponse: LLM 응답
        """
        # 배치용 프롬프트 내용 구성
        batch_content = ""
        for i, doc in enumerate(batch, 1):
            # 문서 내용 정제 및 길이 제한
            cleaned_content = clean_text(doc.content)
            truncated_content = cleaned_content[:self.max_content_length]
            
            batch_content += f"=== 문서 {i} ===\n"
            batch_content += f"제목: {doc.title}\n"
            batch_content += f"출처: {doc.url}\n"
            batch_content += f"내용: {truncated_content}\n\n"
        
        # 사용자 질문 컨텍스트 추가
        question_context = f"\n\n사용자 질문: {user_question}" if user_question.strip() else ""
        
        # 배치 요약 프롬프트 템플릿
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 문서 요약 전문가입니다. 주어진 {len(batch)}개 문서를 각각 {self.max_summary_length}자 이내로 요약해주세요.

요약 규칙:
1. 각 문서의 핵심 내용과 주요 포인트만 포함
2. 객관적이고 정확한 정보만 추출
3. 개인적인 의견이나 추측은 제외  
4. 명확하고 간결한 문장으로 작성
5. 각 요약은 {self.max_summary_length}자를 초과하지 말 것
6. 사용자 질문과 관련된 내용에 우선순위를 둘 것{question_context}

출력 형식 - 각 요약을 구분자로 분리:
===SUMMARY_1===
첫 번째 문서의 요약 내용
===SUMMARY_2===  
두 번째 문서의 요약 내용
===SUMMARY_3===
세 번째 문서의 요약 내용

(문서 개수만큼 반복)"""),
            ("user", "{batch_content}")
        ])
        
        return await self.llm_service._call_llm(
            prompt_template=prompt_template,
            input_variables={"batch_content": batch_content},
            agent_type="doc_summarizer",
            output_parser=self.llm_service.str_parser,
            expected_type=str
        )
    
    def _parse_batch_response(self, response_content: str, batch: List[Document]) -> List[DocumentSummary]:
        """
        배치 응답을 파싱하여 DocumentSummary 객체 생성
        
        Args:
            response_content: LLM 응답 내용
            batch: 원본 문서 배치
            
        Returns:
            List[DocumentSummary]: 파싱된 요약 객체 리스트
        """
        summaries = []
        
        try:
            # 1단계: JSON 파싱 시도 (혹시 JSON 형태로 반환된 경우)
            try:
                json_data = json.loads(response_content)
                if isinstance(json_data, list):
                    summary_texts = json_data
                else:
                    raise ValueError("JSON이 배열 형태가 아님")
            except (json.JSONDecodeError, ValueError):
                # 2단계: 구분자 기반 파싱 (폴백 전략)
                summary_texts = self._parse_with_separators(response_content)
            
            # 3단계: DocumentSummary 객체 생성
            for i, doc in enumerate(batch):
                if i < len(summary_texts):
                    # 성공적으로 파싱된 요약 사용
                    summary_text = summary_texts[i].strip()
                    
                    # 요약문 유효성 검증
                    if len(summary_text) >= 10 and is_valid_text(summary_text):
                        summary = DocumentSummary(
                            document_hash=doc.content_hash,
                            summary=summary_text,
                            key_points=self._extract_key_points(summary_text),
                            confidence_score=0.8,  # 배치 처리 기본 신뢰도
                            word_count=len(summary_text)
                        )
                        summaries.append(summary)
                    else:
                        # 유효하지 않은 요약인 경우 폴백
                        fallback_summary = self._create_single_fallback_summary(doc)
                        summaries.append(fallback_summary)
                        self.failed_documents += 1
                else:
                    # 요약이 부족한 경우 폴백 (원본 내용 일부 사용)
                    fallback_summary = self._create_single_fallback_summary(doc)
                    summaries.append(fallback_summary)
                    self.failed_documents += 1
            
            logger.debug(f"배치 파싱 완료: {len(summaries)} 요약 생성")
            return summaries
            
        except Exception as e:
            logger.error(f"배치 응답 파싱 실패: {e}")
            # 전체 실패 시 모든 문서에 대해 폴백 요약 생성
            return self._create_fallback_summaries(batch)
    
    def _parse_with_separators(self, content: str) -> List[str]:
        """
        구분자를 사용하여 응답 내용 파싱
        
        Args:
            content: 파싱할 응답 내용
            
        Returns:
            List[str]: 분리된 요약 텍스트 리스트
        """
        summaries = []
        
        try:
            # ===SUMMARY_N=== 패턴으로 분리
            pattern = r'===SUMMARY_\d+===(.*?)(?====SUMMARY_\d+===|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            if matches:
                summaries = [match.strip() for match in matches]
                logger.debug(f"구분자 파싱 성공: {len(summaries)}개 요약 추출")
            else:
                # 대안 구분자 시도: === 단독
                parts = re.split(r'={3,}', content)
                summaries = [part.strip() for part in parts if part.strip() and len(part.strip()) > 10]
                logger.debug(f"대안 구분자 파싱: {len(summaries)}개 요약 추출")
            
            return summaries
            
        except Exception as e:
            logger.error(f"구분자 파싱 실패: {e}")
            return []
    
    def _extract_key_points(self, summary_text: str) -> List[str]:
        """
        요약문에서 핵심 포인트 추출
        
        Args:
            summary_text: 요약 텍스트
            
        Returns:
            List[str]: 핵심 포인트 리스트
        """
        # 간단한 문장 분리로 핵심 포인트 추출
        sentences = re.split(r'[.!?]\s+', summary_text)
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # 의미 있는 길이의 문장만
                key_points.append(sentence)
        
        return key_points[:3]  # 최대 3개까지
    
    def _create_fallback_summaries(self, batch: List[Document]) -> List[DocumentSummary]:
        """
        폴백 요약 생성 (배치 전체 실패 시)
        
        Args:
            batch: 실패한 배치 문서들
            
        Returns:
            List[DocumentSummary]: 폴백 요약 리스트
        """
        fallback_summaries = []
        
        for doc in batch:
            fallback_summary = self._create_single_fallback_summary(doc)
            fallback_summaries.append(fallback_summary)
        
        logger.info(f"폴백 요약 생성 완료: {len(fallback_summaries)}개")
        return fallback_summaries
    
    def _create_single_fallback_summary(self, doc: Document) -> DocumentSummary:
        """
        단일 문서에 대한 폴백 요약 생성
        
        Args:
            doc: 폴백 요약을 생성할 문서
            
        Returns:
            DocumentSummary: 폴백 요약 객체
        """
        # 원본 내용의 처음 부분을 요약으로 사용
        cleaned_content = clean_text(doc.content)
        fallback_text = truncate_text(cleaned_content, self.max_summary_length, "...")
        
        return DocumentSummary(
            document_hash=doc.content_hash,
            summary=fallback_text,
            key_points=[fallback_text],
            confidence_score=0.3,  # 낮은 신뢰도 표시
            word_count=len(fallback_text)
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        success_rate = 0.0
        if self.total_documents_processed > 0:
            success_rate = (self.total_summaries_generated - self.failed_documents) / self.total_documents_processed * 100
        
        return {
            "total_documents_processed": self.total_documents_processed,
            "total_summaries_generated": self.total_summaries_generated,
            "failed_documents": self.failed_documents,
            "batch_retry_count": self.batch_retry_count,
            "success_rate_percent": round(success_rate, 2),
            "batch_size": self.batch_size,
            "max_content_length": self.max_content_length,
            "max_summary_length": self.max_summary_length
        }


# LangGraph 노드 함수
async def summarize_documents_node(state: ResearchState) -> ResearchState:
    """
    문서 요약 LangGraph 노드
    
    State의 documents를 받아서 요약을 생성하고 summaries에 저장합니다.
    
    Args:
        state: 현재 연구 상태
        
    Returns:
        ResearchState: 요약이 추가된 업데이트된 상태
    """
    logger.info("=== 문서 요약 노드 시작 ===")
    
    try:
        # 현재 단계 설정
        state = StateManager.set_step(state, "summarizing", "📝 문서 요약 중...")
        
        # 문서 확인
        if not state.get("documents"):
            error_msg = "요약할 문서가 없습니다."
            logger.warning(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # Document 객체 생성
        documents = []
        for doc_dict in state["documents"]:
            try:
                doc = Document(
                    title=doc_dict.get("title", ""),
                    url=doc_dict.get("url", ""),
                    content=doc_dict.get("content", ""),
                    source=doc_dict.get("source", "unknown")
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"문서 객체 생성 실패: {e}")
                continue
        
        if not documents:
            error_msg = "유효한 문서 객체를 생성할 수 없습니다."
            logger.error(error_msg)
            return StateManager.set_error(state, error_msg)
        
        # 문서 요약기 생성 및 실행
        summarizer = DocumentSummarizer()
        summaries = await summarizer.summarize_documents(
            documents=documents,
            user_question=state.get("user_input", "")
        )
        
        if not summaries:
            # 문서가 없을 때 기본 요약 생성
            logger.warning("문서 요약이 없어 기본 요약을 생성합니다.")
            user_question = state.get("user_input", "")
            fallback_summary = f"{user_question}에 대한 정보를 찾는 중입니다. 현재 관련 문서를 수집하지 못했습니다."
            summary_texts = [fallback_summary]
        else:
            # 요약 결과를 문자열 리스트로 변환하여 상태에 저장
            summary_texts = [summary.summary for summary in summaries]
        
        # 요약 결과를 문자열 리스트로 변환하여 상태에 저장
        summary_texts = [summary.summary for summary in summaries]
        
        # 상태 업데이트
        new_state = state.copy()
        new_state["summaries"] = summary_texts
        
        # 처리 통계 로깅
        stats = summarizer.get_processing_stats()
        logger.info(f"문서 요약 완료: {stats}")
        
        new_state = StateManager.add_log(
            new_state, 
            f"📝 문서 요약 완료: {len(summary_texts)}개 요약 생성 (성공률: {stats['success_rate_percent']}%)"
        )
        
        logger.info("=== 문서 요약 노드 완료 ===")
        return new_state
        
    except Exception as e:
        error_msg = f"문서 요약 노드에서 오류 발생: {str(e)}"
        logger.error(error_msg)
        return StateManager.set_error(state, error_msg)


# 유틸리티 함수들
def create_document_summarizer(batch_size: int = 5) -> DocumentSummarizer:
    """
    문서 요약기 인스턴스 생성 헬퍼 함수
    
    Args:
        batch_size: 배치 크기
        
    Returns:
        DocumentSummarizer: 설정된 문서 요약기 인스턴스
    """
    return DocumentSummarizer(batch_size=batch_size)

