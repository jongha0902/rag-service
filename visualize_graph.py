import os
from utils.ollama_rag import rag_graph

def generate_graph_image():
    print("🎨 그래프 시각화 생성 중...")
    
    try:
        # 1. 그래프 객체 가져오기
        graph = rag_graph.get_graph()
        
        # 2. Mermaid PNG 바이너리 생성
        # (draw_mermaid_png()는 LangChain/LangGraph 내부적으로 Mermaid API를 사용해 이미지를 생성합니다)
        png_data = graph.draw_mermaid_png()
        
        # 3. 파일로 저장
        output_file = "rag_flow.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
            
        print(f"✅ 그래프가 '{output_file}'로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 시각화 생성 실패: {e}")
        print("💡 팁: 로컬 환경에 문제가 있다면 아래 Mermaid 코드를 복사해서 https://mermaid.live 에 붙여넣으세요.")
        
        # 이미지 생성이 안 될 경우 텍스트로 출력
        try:
            print("\n--- Mermaid Code ---")
            print(rag_graph.get_graph().draw_mermaid())
            print("--------------------\n")
        except:
            pass

if __name__ == "__main__":
    generate_graph_image()