# run.py
"""
Deep Research Chatbot 실행 스크립트

Streamlit 애플리케이션을 실행하기 전에 필요한 초기화 작업을 수행합니다.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")
    
    # .env 파일 확인
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env 파일이 없습니다.")
        return False
    
    # 필수 환경 변수 확인
    from config.settings import settings
    try:
        # 설정 로드 테스트
        print(f"✅ LLM 제공자: {settings.LLM_PROVIDER}")
        print(f"✅ 채팅 모델: {settings.chat_model}")
        print(f"✅ Tavily 사용 가능: {'예' if settings.TAVILY_API_KEY else '아니오'}")
        print(f"✅ LangSmith 추적: {'활성화' if settings.LANGCHAIN_TRACING_V2 else '비활성화'}")
        return True
    except Exception as e:
        print(f"❌ 환경 설정 오류: {e}")
        return False

def check_dependencies():
    """필수 패키지 확인"""
    print("\n📦 필수 패키지 확인 중...")
    
    required_packages = [
        'streamlit',
        'langchain',
        'langgraph', 
        'chromadb',
        'openai',
        'duckduckgo_search',
        'tavily-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tavily-python':
                import tavily
            elif package == 'duckduckgo_search':
                import duckduckgo_search
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n설치 명령어: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def initialize_directories():
    """필요한 디렉토리 생성"""
    print("\n📁 디렉토리 초기화 중...")
    
    directories = [
        "data/chroma_db",
        "data/temp", 
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def run_streamlit():
    """Streamlit 애플리케이션 실행"""
    print("\n🚀 Deep Research Chatbot 시작 중...")
    print("브라우저에서 http://localhost:8501 을 열어주세요.")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Deep Research Chatbot을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("🔍 Deep Research Chatbot")
    print("=" * 50)
    
    # 환경 확인
    if not check_environment():
        print("\n❌ 환경 설정을 확인해주세요.")
        return
    
    # 패키지 확인
    if not check_dependencies():
        print("\n❌ 필수 패키지를 설치해주세요.")
        return
    
    # 디렉토리 초기화
    initialize_directories()
    
    # Streamlit 실행
    run_streamlit()

if __name__ == "__main__":
    main()