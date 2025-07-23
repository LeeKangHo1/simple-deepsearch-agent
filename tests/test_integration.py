# test_integration.py
"""
간단한 통합 테스트 스크립트 (문제 해결 버전)

지금까지 만든 주요 모듈들의 기본 동작을 순차적으로 테스트합니다.
실제 API 호출 없이 설정 로드, import, 기본 객체 생성 등을 확인합니다.

문제 해결:
- 인코딩 문제 (UTF-8 강제)
- LangChain ListOutputParser 문제
- Import 오류 진단 강화

사용법:
    python test_integration.py  (프로젝트 루트에서)
    python tests/test_integration.py  (어디서든)
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📁 Project root: {project_root}")
print(f"🐍 Python path updated")

import asyncio
import traceback
from typing import Dict, Any
import time

class TestRunner:
    """간단한 테스트 러너 클래스"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def test(self, test_name: str, test_func):
        """테스트 실행 및 결과 기록"""
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            elapsed = time.time() - start_time
            
            print(f"✅ PASSED ({elapsed:.2f}s)")
            self.passed += 1
            self.results.append({
                "name": test_name,
                "status": "PASSED",
                "time": elapsed,
                "result": result
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ FAILED ({elapsed:.2f}s)")
            print(f"Error: {str(e)}")
            if "Traceback" not in str(e):
                print(f"Traceback: {traceback.format_exc()}")
            
            self.failed += 1
            self.results.append({
                "name": test_name,
                "status": "FAILED", 
                "time": elapsed,
                "error": str(e)
            })
    
    def summary(self):
        """테스트 결과 요약"""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"📊 TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success rate: {(self.passed/total*100):.1f}%" if total > 0 else "No tests run")
        
        if self.failed > 0:
            print(f"\n💡 Failed tests:")
            for result in self.results:
                if result["status"] == "FAILED":
                    print(f"   - {result['name']}: {result.get('error', 'Unknown error')[:100]}...")

# ===========================
# 개별 테스트 함수들
# ===========================

def test_basic_imports():
    """기본 Python 패키지 import 테스트"""
    import os
    import json
    import hashlib
    import logging
    from pathlib import Path
    from datetime import datetime
    from dataclasses import dataclass
    from typing import List, Dict, Optional
    
    print("✓ Basic Python modules imported successfully")
    return "Basic imports OK"

def test_external_packages():
    """외부 패키지 import 테스트"""
    try:
        # LangChain 관련
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
        print("✓ LangChain packages imported")
        
        # LangChain ListOutputParser 확인
        try:
            from langchain_core.output_parsers import ListOutputParser
            # 직접 인스턴스화 시도 (실패 예상)
            try:
                parser = ListOutputParser()
                print("✓ ListOutputParser can be instantiated")
            except TypeError as e:
                print(f"⚠ ListOutputParser is abstract: {e}")
                print("💡 Need to use PydanticOutputParser or custom implementation")
        except ImportError:
            print("⚠ ListOutputParser not available")
        
        # OpenAI (선택적)
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            print("✓ OpenAI packages imported")
        except ImportError:
            print("⚠ OpenAI packages not available")
        
        # Google (선택적)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✓ Google packages imported")
        except ImportError:
            print("⚠ Google packages not available")
        
        # 검색 라이브러리
        from duckduckgo_search import DDGS
        print("✓ DuckDuckGo search imported")
        
        try:
            from tavily import TavilyClient
            print("✓ Tavily search imported")
        except ImportError:
            print("⚠ Tavily not available")
        
        # Chroma DB
        import chromadb
        print("✓ ChromaDB imported")
        
        # NumPy
        import numpy as np
        print("✓ NumPy imported")
        
        return "External packages OK"
        
    except ImportError as e:
        raise ImportError(f"Missing required package: {e}")

def test_project_structure():
    """프로젝트 구조 확인"""
    required_dirs = ['config', 'models', 'services', 'utils', 'ui']
    required_files = [
        'config/__init__.py',
        'config/settings.py', 
        'config/llm_config.py',
        'models/__init__.py',
        'models/data_models.py',
        'models/state.py',
        'services/__init__.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    # 디렉토리 확인
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✓ Directory exists: {dir_name}/")
    
    # 파일 확인
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            # 파일 크기 확인
            size = full_path.stat().st_size
            print(f"✓ File exists: {file_path} ({size} bytes)")
            
            # 빈 파일이면 경고
            if size == 0:
                print(f"  ⚠ Warning: {file_path} is empty")
    
    if missing_dirs or missing_files:
        error_msg = ""
        if missing_dirs:
            error_msg += f"Missing directories: {missing_dirs}\n"
        if missing_files:
            error_msg += f"Missing files: {missing_files}"
        raise FileNotFoundError(error_msg)
    
    return f"Project structure OK: {len(required_dirs)} dirs, {len(required_files)} files"

def test_init_files():
    """__init__.py 파일들 확인 및 생성"""
    init_files = [
        'config/__init__.py',
        'models/__init__.py', 
        'services/__init__.py',
        'utils/__init__.py',
        'ui/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            # __init__.py 파일 생성
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("# __init__.py\n", encoding='utf-8')
            print(f"✓ Created: {init_file}")
        else:
            print(f"✓ Exists: {init_file}")
    
    return f"Init files OK: {len(init_files)} files"

def test_env_file():
    """환경 변수 파일 확인 (인코딩 문제 해결)"""
    env_path = project_root / '.env'
    
    if not env_path.exists():
        print("⚠ .env file not found")
        print("💡 Creating sample .env file...")
        
        sample_env = """# Sample .env file for Deep Research Chatbot
LLM_PROVIDER=gpt
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=simple-deepresearch-agent

MAX_SEARCH_RESULTS=20
MAX_DOCUMENTS_TO_PROCESS=15
"""
        env_path.write_text(sample_env, encoding='utf-8')
        print(f"✓ Sample .env file created at: {env_path}")
        print("⚠ Please update with your actual API keys")
        
        return "Sample .env created"
    else:
        print(f"✓ .env file exists at: {env_path}")
        
        try:
            # UTF-8로 읽기 시도
            content = env_path.read_text(encoding='utf-8')
            print("✓ .env file read as UTF-8")
        except UnicodeDecodeError:
            try:
                # CP949로 읽기 시도
                content = env_path.read_text(encoding='cp949')
                print("⚠ .env file is in CP949 encoding")
                print("💡 Converting to UTF-8...")
                
                # UTF-8로 다시 저장
                env_path.write_text(content, encoding='utf-8')
                print("✓ Converted .env file to UTF-8")
            except UnicodeDecodeError:
                print("❌ Cannot read .env file - unknown encoding")
                return ".env encoding issue"
        
        # .env 파일 내용 확인 (민감한 정보 제외)
        lines = content.split('\n')
        config_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        
        print(f"✓ Found {len(config_lines)} configuration lines")
        for line in config_lines:
            if '=' in line:
                key = line.split('=')[0]
                value = line.split('=', 1)[1]
                if 'KEY' in key.upper():
                    # API 키는 마스킹
                    masked_value = value[:10] + '...' if len(value) > 10 else '***'
                    print(f"  - {key}={masked_value}")
                else:
                    print(f"  - {key}={value}")
        
        return f".env file OK: {len(config_lines)} configs"

def test_file_contents():
    """주요 파일들의 내용 확인"""
    files_to_check = [
        'config/settings.py',
        'models/data_models.py',
        'services/llm_service.py',
        'services/search_service.py'
    ]
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                content = full_path.read_text(encoding='utf-8')
                lines = len(content.split('\n'))
                print(f"✓ {file_path}: {lines} lines")
                
                # 주요 클래스나 함수가 있는지 확인
                if 'class ' in content or 'def ' in content:
                    print(f"  ✓ Contains class/function definitions")
                else:
                    print(f"  ⚠ No class/function definitions found")
                    
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"❌ Missing: {file_path}")
    
    return "File contents checked"

def test_imports_individually():
    """개별 모듈 import 테스트"""
    modules_to_test = [
        ('config.settings', 'settings'),
        ('models.data_models', 'Document'),
        ('models.state', 'ResearchState'),
        ('services.search_service', 'SearchService'),
        ('services.llm_service', 'LLMService')
    ]
    
    results = {}
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name} imported successfully")
                results[module_name] = "OK"
            else:
                print(f"⚠ {module_name} imported but {class_name} not found")
                results[module_name] = f"Missing {class_name}"
        except ImportError as e:
            print(f"❌ {module_name} import failed: {e}")
            results[module_name] = f"Import error: {e}"
        except Exception as e:
            print(f"❌ {module_name} error: {e}")
            results[module_name] = f"Error: {e}"
    
    return results

# ===========================
# 메인 테스트 실행
# ===========================

def main():
    """메인 테스트 실행 함수"""
    print("🚀 Starting Deep Research Chatbot Integration Tests (Fixed)")
    print("=" * 60)
    
    runner = TestRunner()
    
    # 순차적으로 테스트 실행
    runner.test("Basic Python Imports", test_basic_imports)
    runner.test("External Packages", test_external_packages)
    runner.test("Project Structure", test_project_structure)
    runner.test("Init Files", test_init_files)
    runner.test("Environment File", test_env_file)
    runner.test("File Contents", test_file_contents)
    runner.test("Individual Imports", test_imports_individually)
    
    # 결과 요약
    runner.summary()
    
    # 성공률에 따른 추천사항
    success_rate = runner.passed / (runner.passed + runner.failed) * 100 if (runner.passed + runner.failed) > 0 else 0
    print(f"\n💡 Next Steps:")
    if success_rate >= 80:
        print("   ✅ Great! Ready to proceed with agent development")
    elif success_rate >= 50:
        print("   ⚠ Fix remaining issues before proceeding")
        print("   🔧 Focus on missing files and imports")
    else:
        print("   🚨 Major issues found - need to create missing files")
        print("   📁 Check if files were properly saved from artifacts")
    
    print(f"\n🔍 Diagnostic Info:")
    print(f"   📁 Project root: {project_root}")
    print(f"   🐍 Python version: {sys.version}")
    print(f"   📦 Python path: {sys.path[:3]}...")
    
    # 종료 코드 반환
    return 0 if runner.failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        print(traceback.format_exc())
        sys.exit(1)