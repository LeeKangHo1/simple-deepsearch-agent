# tests/conftest.py
import pytest
import chromadb

@pytest.fixture(scope="session", autouse=True)
def initialize_chroma_collection():
    """
    최신 방식으로 chromadb.Client() 생성 → research_docs 컬렉션이 없으면 생성
    """
    client = chromadb.PersistentClient(path="data/chroma_db")  # 경로는 프로젝트에 맞게

    try:
        client.get_collection("research_docs")
    except:
        client.create_collection("research_docs")
