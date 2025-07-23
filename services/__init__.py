from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

class LLMService:
    def __init__(self):
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0

        # 출력 파서 초기화
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        self.list_parser = StrOutputParser()  # ✅ 여기 수정함
