# test_integration.py
"""
통합 테스트 스크립트

전체 워크플로우를 테스트하여 각 단계별 출력을 확인합니다.
"""

import asyncio
import logging
from workflows.research_workflow import execute_research

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_workflow():
    """워크플로우 통합 테스트"""
    
    def progress_callback(step: str, message: str):
        """진행 상태 출력"""
        print(f"[{step}] {message}")
    
    # 테스트 질문
    test_query = "오픈소스 LLM 최신 트렌드 2023"
    
    print(f"=== 통합 테스트 시작 ===")
    print(f"질문: {test_query}")
    print("=" * 50)
    
    try:
        # 워크플로우 실행
        result = await execute_research(test_query, progress_callback)
        
        print("\n" + "=" * 50)
        print("=== 실행 결과 ===")
        print(f"성공 여부: {result['success']}")
        print(f"실행 시간: {result['execution_time']:.2f}초")
        
        if result['success']:
            markdown_answer = result['markdown_answer']
            print(f"응답 길이: {len(markdown_answer)}자")
            print("\n=== 마크다운 응답 ===")
            print(markdown_answer)
            
            # 통계 정보
            stats = result.get('stats', {})
            print(f"\n=== 통계 정보 ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print(f"오류 메시지: {result['error_message']}")
    
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        logger.error(f"Integration test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_workflow())