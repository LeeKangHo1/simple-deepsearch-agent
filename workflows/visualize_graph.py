# graph/visualize_graph.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflows.graph_builder import build_graph

# 그래프 컴파일
graph = build_graph()

# 현재 파일 기준 디렉토리 설정
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "..", "data", "graph.png")

# data 폴더가 없으면 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Mermaid 기반 시각화 이미지 저장
graph.get_graph().draw_mermaid_png(output_file_path=output_path)

print(f"✅ 그래프 시각화 저장 완료: {output_path}")