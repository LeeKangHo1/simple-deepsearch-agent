# services/vector_db.py
"""
Chroma 벡터 데이터베이스 서비스 (업데이트 버전)

문서의 임베딩 벡터를 저장하고 유사도 검색을 수행하는 서비스.
중복 문서 제거, 관련성 기반 정렬, 의미적 검색 등의 기능을 제공합니다.

주요 기능:
- 문서 임베딩 생성 및 저장
- 유사도 기반 중복 문서 제거
- 의미적 유사도 검색
- 벡터 DB 관리 (생성, 삭제, 초기화)
- 메타데이터 기반 필터링
- 배치 처리를 통한 성능 최적화

업데이트 내용:
- NumPy 기반 최적화된 코사인 유사도 계산
- 표준 RAG 호환성 (-1~1 범위 유지)
- 조정된 임계값 (0.85 → 0.7)
"""

import logging
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import asdict
import numpy as np

# Chroma 및 임베딩 관련
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 프로젝트 모듈들
from models.data_models import Document, SearchEngine
from config.llm_config import get_cached_embedding_model
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Chroma 벡터 데이터베이스 서비스 클래스 (업데이트 버전)
    
    문서들의 임베딩 벡터를 관리하고 유사도 기반 검색을 제공.
    중복 제거와 관련성 정렬을 통해 고품질 문서 세트를 생성합니다.
    
    사용 예시:
        vector_db = VectorDBService()
        await vector_db.add_documents(documents)
        similar_docs = await vector_db.find_similar_documents(query, top_k=5)
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        벡터 DB 서비스 초기화
        
        Args:
            collection_name: 컬렉션 이름 (기본값: 설정에서 가져옴)
        """
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # 통계 추적
        self.total_documents = 0
        self.total_embeddings_generated = 0
        self.duplicates_removed = 0
        
        # 초기화
        self._initialize_client()
    
    def _initialize_client(self):
        """Chroma 클라이언트 및 컬렉션 초기화"""
        try:
            # Chroma 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,  # 텔레메트리 비활성화
                    allow_reset=True
                )
            )
            
            # 임베딩 모델 초기화
            self.embedding_model = get_cached_embedding_model()
            
            # 컬렉션 가져오기 또는 생성
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Existing collection '{self.collection_name}' loaded")
            except ValueError:
                # 컬렉션이 없으면 새로 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Deep research chatbot document storage"}
                )
                logger.info(f"New collection '{self.collection_name}' created")
            
            # 기존 문서 수 확인
            self.total_documents = self.collection.count()
            logger.info(f"Vector DB initialized with {self.total_documents} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector DB: {e}")
            raise
    
    async def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 50,
        similarity_threshold: float = 0.7  # 업데이트: 0.85 → 0.7
    ) -> Tuple[List[Document], int]:
        """
        문서들을 벡터 DB에 추가 (중복 제거 포함)
        
        Args:
            documents: 추가할 문서 리스트
            batch_size: 배치 크기 (임베딩 생성 최적화용)
            similarity_threshold: 중복 판정 임계값 (-1.0~1.0, 표준 RAG 호환)
            
        Returns:
            Tuple[List[Document], int]: (고유 문서 리스트, 제거된 중복 개수)
        """
        if not documents:
            return [], 0
        
        start_time = time.time()
        logger.info(f"Adding {len(documents)} documents to vector DB...")
        
        # 1단계: 해시 기반 중복 제거
        unique_docs = self._remove_hash_duplicates(documents)
        hash_duplicates = len(documents) - len(unique_docs)
        
        # 2단계: 임베딩 기반 중복 제거
        final_docs, embedding_duplicates = await self._remove_embedding_duplicates(
            unique_docs, similarity_threshold, batch_size
        )
        
        # 3단계: 벡터 DB에 저장
        await self._store_documents(final_docs, batch_size)
        
        total_duplicates = hash_duplicates + embedding_duplicates
        self.duplicates_removed += total_duplicates
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"Document processing completed: "
            f"{len(final_docs)}/{len(documents)} documents added, "
            f"{total_duplicates} duplicates removed in {elapsed_time:.2f}s"
        )
        
        return final_docs, total_duplicates
    
    def _remove_hash_duplicates(self, documents: List[Document]) -> List[Document]:
        """
        해시 기반 중복 문서 제거
        
        빠른 1차 중복 제거를 위해 content_hash를 사용.
        완전히 동일한 문서들을 효율적으로 제거합니다.
        
        Args:
            documents: 중복 제거할 문서 리스트
            
        Returns:
            List[Document]: 해시 기반 중복이 제거된 문서 리스트
        """
        seen_hashes = set()
        unique_docs = []
        
        for doc in documents:
            doc_hash = doc.content_hash
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
        
        removed_count = len(documents) - len(unique_docs)
        if removed_count > 0:
            logger.debug(f"Hash-based deduplication: {removed_count} exact duplicates removed")
        
        return unique_docs
    
    async def _remove_embedding_duplicates(
        self, 
        documents: List[Document], 
        threshold: float,
        batch_size: int
    ) -> Tuple[List[Document], int]:
        """
        임베딩 기반 유사 문서 제거 (업데이트 버전)
        
        의미적으로 유사한 문서들을 임베딩 벡터 비교를 통해 제거.
        표준 RAG 호환 코사인 유사도를 사용합니다.
        
        Args:
            documents: 중복 제거할 문서 리스트
            threshold: 유사도 임계값 (-1.0~1.0, 표준 RAG 스케일)
            batch_size: 임베딩 생성 배치 크기
            
        Returns:
            Tuple[List[Document], int]: (중복 제거된 문서, 제거된 개수)
        """
        if not documents:
            return [], 0
        
        logger.debug(f"Starting embedding-based deduplication for {len(documents)} documents (threshold: {threshold})")
        
        # 임베딩 생성
        await self._generate_embeddings(documents, batch_size)
        
        # 유사도 기반 중복 제거
        unique_docs = []
        removed_count = 0
        
        for i, doc in enumerate(documents):
            is_duplicate = False
            
            # 이미 선택된 문서들과 비교
            for j, unique_doc in enumerate(unique_docs):
                similarity = self._calculate_cosine_similarity(
                    doc.embedding, unique_doc.embedding
                )
                
                if similarity > threshold:
                    # 중복으로 판정 - 더 긴 내용을 가진 문서 선택
                    if len(doc.content) > len(unique_doc.content):
                        # 기존 문서를 새 문서로 교체
                        unique_docs[j] = doc
                        logger.debug(f"Replaced document (similarity: {similarity:.3f}): '{unique_doc.title}' → '{doc.title}'")
                    else:
                        logger.debug(f"Removed duplicate (similarity: {similarity:.3f}): '{doc.title}'")
                    
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        if removed_count > 0:
            logger.debug(f"Embedding-based deduplication: {removed_count} similar documents removed")
        
        return unique_docs, removed_count
    
    async def _generate_embeddings(self, documents: List[Document], batch_size: int):
        """
        문서들의 임베딩 벡터 생성
        
        배치 처리를 통해 효율적으로 임베딩을 생성하고
        각 문서 객체에 임베딩을 저장합니다.
        
        Args:
            documents: 임베딩을 생성할 문서 리스트
            batch_size: 배치 크기
        """
        logger.debug(f"Generating embeddings for {len(documents)} documents")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 임베딩할 텍스트 준비 (제목 + 내용 조합)
            texts = []
            for doc in batch:
                # 제목과 내용을 조합하여 더 풍부한 임베딩 생성
                combined_text = f"{doc.title}\n\n{doc.content[:1000]}"  # 첫 1000자만 사용
                texts.append(combined_text)
            
            try:
                # 임베딩 생성
                embeddings = await self.embedding_model.aembed_documents(texts)
                
                # 각 문서에 임베딩 할당 (NumPy 배열로 변환)
                for doc, embedding in zip(batch, embeddings):
                    doc.embedding = np.array(embedding, dtype=np.float32)  # NumPy 배열로 저장
                
                self.total_embeddings_generated += len(batch)
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                # 실패한 경우 빈 임베딩 할당 (오류 방지)
                for doc in batch:
                    # OpenAI embedding 차원에 맞는 zero 벡터
                    embedding_dim = 1536  # text-embedding-3-large 차원
                    doc.embedding = np.zeros(embedding_dim, dtype=np.float32)
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        표준 RAG 호환 코사인 유사도 계산 (업데이트 버전)
        
        NumPy를 사용한 최적화된 구현으로 실제 프로덕션 RAG에서 사용되는 방식.
        -1~1 범위의 표준 코사인 유사도를 반환합니다.
        
        Args:
            vec1, vec2: 비교할 벡터들 (NumPy 배열)
            
        Returns:
            float: 코사인 유사도 (-1.0~1.0)
                   1.0: 완전히 같은 방향 (매우 유사)
                   0.0: 직교 (무관함)
                  -1.0: 완전히 반대 방향 (매우 다름)
        """
        try:
            # NumPy 배열로 변환 (이미 배열이면 변환하지 않음)
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1, dtype=np.float32)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2, dtype=np.float32)
            
            # 벡터 유효성 검사
            if vec1.size == 0 or vec2.size == 0 or vec1.shape != vec2.shape:
                return 0.0
            
            # NumPy 벡터화 연산으로 코사인 유사도 계산
            # cos(θ) = (A·B) / (|A|×|B|)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Zero vector 처리
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 표준 코사인 유사도 (정규화 없음, -1~1 범위)
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # NumPy scalar을 Python float로 변환
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _store_documents(self, documents: List[Document], batch_size: int):
        """
        문서들을 Chroma DB에 저장
        
        Args:
            documents: 저장할 문서 리스트
            batch_size: 배치 크기
        """
        if not documents:
            return
        
        logger.debug(f"Storing {len(documents)} documents to Chroma DB")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Chroma DB 형식으로 데이터 준비
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in batch:
                # 고유 ID 생성 (content_hash 기반)
                doc_id = f"doc_{doc.content_hash}"
                ids.append(doc_id)
                
                # 임베딩 벡터 (NumPy 배열을 리스트로 변환)
                if isinstance(doc.embedding, np.ndarray):
                    embeddings.append(doc.embedding.tolist())
                else:
                    embeddings.append(doc.embedding)
                
                # 메타데이터 (검색 및 필터링용)
                metadata = {
                    "title": doc.title,
                    "url": doc.url,
                    "source": doc.source.value,
                    "doc_type": doc.doc_type.value,
                    "url_domain": doc.url_domain,
                    "relevance_score": doc.relevance_score,
                    "content_length": len(doc.content),
                    "created_at": doc.created_at.isoformat()
                }
                metadatas.append(metadata)
                
                # 전문 검색용 텍스트 (제목 + 스니펫)
                document_text = f"{doc.title}\n{doc.snippet}"
                documents_text.append(document_text)
            
            try:
                # Chroma DB에 추가
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_text
                )
                
                logger.debug(f"Stored batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to Chroma DB")
                
            except Exception as e:
                logger.error(f"Failed to store batch to Chroma DB: {e}")
        
        # 전체 문서 수 업데이트
        self.total_documents = self.collection.count()
    
    async def find_similar_documents(
        self, 
        query: str, 
        top_k: int = 10,
        min_similarity: float = 0.3,  # 업데이트: 표준 RAG 스케일
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서들을 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            min_similarity: 최소 유사도 임계값 (-1.0~1.0, 표준 RAG 스케일)
            filters: 메타데이터 필터 (예: {"source": "tavily"})
            
        Returns:
            List[Dict[str, Any]]: 유사한 문서들의 메타데이터 리스트
        """
        if not query.strip():
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_model.aembed_query(query)
            
            # 벡터 유사도 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 50),  # 필터링을 고려해 더 많이 가져옴
                where=filters,  # 메타데이터 필터
                include=["metadatas", "distances", "documents"]
            )
            
            # 결과 정리 및 필터링
            similar_docs = []
            for i, (metadata, distance, document) in enumerate(zip(
                results["metadatas"][0],
                results["distances"][0], 
                results["documents"][0]
            )):
                # Chroma의 거리를 표준 코사인 유사도로 변환
                # Chroma는 cosine distance = 1 - cosine_similarity를 사용
                similarity = 1 - distance
                
                if similarity >= min_similarity:
                    doc_info = {
                        "title": metadata["title"],
                        "url": metadata["url"],
                        "source": metadata["source"],
                        "similarity": round(similarity, 3),
                        "snippet": document,
                        "relevance_score": metadata.get("relevance_score", 0.0)
                    }
                    similar_docs.append(doc_info)
            
            # 유사도 순으로 정렬하고 상위 k개만 반환
            similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.debug(f"Found {len(similar_docs)} similar documents for query: '{query[:50]}...'")
            return similar_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def get_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        해시값으로 특정 문서 조회
        
        Args:
            content_hash: 문서의 content_hash
            
        Returns:
            Optional[Dict[str, Any]]: 문서 메타데이터 또는 None
        """
        try:
            doc_id = f"doc_{content_hash}"
            results = self.collection.get(
                ids=[doc_id],
                include=["metadatas", "documents"]
            )
            
            if results["ids"]:
                metadata = results["metadatas"][0]
                document = results["documents"][0]
                
                return {
                    "title": metadata["title"],
                    "url": metadata["url"],
                    "source": metadata["source"],
                    "content": document,
                    "created_at": metadata["created_at"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by hash: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            return {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "total_embeddings_generated": self.total_embeddings_generated,
                "duplicates_removed": self.duplicates_removed,
                "persist_directory": settings.CHROMA_PERSIST_DIRECTORY,
                "embedding_model": settings.embedding_model,
                "similarity_threshold": 0.7,  # 업데이트된 기본값
                "cosine_similarity_range": "[-1.0, 1.0] (표준 RAG 호환)"
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """컬렉션의 모든 문서 삭제"""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Deep research chatbot document storage"}
            )
            
            # 통계 초기화
            self.total_documents = 0
            self.total_embeddings_generated = 0
            self.duplicates_removed = 0
            
            logger.info(f"Collection '{self.collection_name}' cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
    
    def delete_documents_by_source(self, source: SearchEngine):
        """특정 검색 엔진에서 온 문서들만 삭제"""
        try:
            # 해당 소스의 문서들 조회
            results = self.collection.get(
                where={"source": source.value},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # 문서들 삭제
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} documents from source: {source.value}")
            
            # 문서 수 업데이트
            self.total_documents = self.collection.count()
            
        except Exception as e:
            logger.error(f"Failed to delete documents by source: {e}")
    
    def get_similarity_distribution(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        현재 컬렉션의 유사도 분포 분석 (디버깅용)
        
        Args:
            sample_size: 분석할 샘플 크기
            
        Returns:
            Dict[str, Any]: 유사도 분포 통계
        """
        try:
            # 랜덤 샘플 가져오기
            all_results = self.collection.get(
                limit=min(sample_size, self.total_documents),
                include=["embeddings", "metadatas"]
            )
            
            if len(all_results["embeddings"]) < 2:
                return {"error": "Not enough documents for analysis"}
            
            similarities = []
            embeddings = [np.array(emb) for emb in all_results["embeddings"]]
            
            # 모든 쌍의 유사도 계산
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._calculate_cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            if not similarities:
                return {"error": "No similarities calculated"}
            
            similarities = np.array(similarities)
            
            return {
                "sample_size": len(embeddings),
                "total_pairs": len(similarities),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "percentiles": {
                    "25th": float(np.percentile(similarities, 25)),
                    "50th": float(np.percentile(similarities, 50)),
                    "75th": float(np.percentile(similarities, 75)),
                    "90th": float(np.percentile(similarities, 90)),
                    "95th": float(np.percentile(similarities, 95))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze similarity distribution: {e}")
            return {"error": str(e)}


# 전역 벡터 DB 서비스 인스턴스
_vector_db_instance = None

def get_vector_db_service() -> VectorDBService:
    """
    전역 벡터 DB 서비스 인스턴스 반환
    
    Returns:
        VectorDBService: 벡터 DB 서비스 인스턴스
    """
    global _vector_db_instance
    
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBService()
        logger.info("Vector DB service instance created")
    
    return _vector_db_instance

def reset_vector_db_service():
    """벡터 DB 서비스 인스턴스 리셋 (테스트용)"""
    global _vector_db_instance
    _vector_db_instance = None
    logger.info("Vector DB service instance reset")