# test_url_debug.py
"""
URL 수집 디버깅 테스트
"""

import asyncio
import logging
from services.search_service import SearchService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_url_collection():
    """URL 수집 테스트"""
    
    search_service = SearchService()
    
    # 간단한 검색 테스트
    query = "오픈소스 LLM 트렌드"
    print(f"검색 쿼리: {query}")
    
    try:
        # 검색 실행
        documents = await search_service.search_all_engines(query, max_results=2)
        
        print(f"\n수집된 문서 수: {len(documents)}")
        
        for i, doc in enumerate(documents, 1):
            print(f"\n=== 문서 {i} ===")
            print(f"제목: {doc.title}")
            print(f"URL: {doc.url}")
            print(f"내용 길이: {len(doc.content)}자")
            print(f"도메인: {doc.url_domain}")
            
            # to_dict 테스트
            doc_dict = doc.to_dict()
            print(f"딕셔너리 URL: {doc_dict.get('url')}")
    
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        logger.error(f"URL collection test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_url_collection())