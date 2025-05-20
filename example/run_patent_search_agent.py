import os
import sys
import argparse
import yaml
from dotenv import load_dotenv
import pprint

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.patent_search_agent import PatentSearchAgent

# .env 파일 로드 (프로젝트 루트에 있는 .env 파일을 로드하기 위함)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Patent Search Agent Individually")
    parser.add_argument("-k", "--keywords", nargs='+', default=["PEFT 및 RAG를 구비한 축산농가 맞춤형 종량제 AI 서비스 시스템 및 그 방법"], help="Keywords for patent search")
    parser.add_argument("-ys", "--year_start", type=int, default=0, help="Start year for search")
    parser.add_argument("-ye", "--year_end", type=int, default=0, help="End year for search")
    parser.add_argument("-mpk", "--max_patents_per_keyword", type=int, default=2, help="Max patents per keyword")
    parser.add_argument("-c", "--config_file", type=str, default="config/config.yaml", help="Path to the config.yaml file relative to project root")
    parser.add_argument("-db", "--db_path", type=str, default=None, help="ChromaDB persist directory. Overrides .env CHROMA_DB_PATH if set.")
    
    args = parser.parse_args()

    # 프로젝트 루트 경로 기준 파일 경로 재설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(project_root, args.config_file)

    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at {config_file_path}")
        sys.exit(1)
        
    config_data = load_config(config_file_path)
    
    embedding_model = config_data.get("embedding_model_ko", "jhgan/ko-sroberta-multitask")
    db_path = args.db_path if args.db_path else os.getenv("CHROMA_DB_PATH", "./data/embeddings/")
    # db_path가 상대 경로일 경우 프로젝트 루트 기준으로 변경
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)

    kipris_service_key = os.getenv("KIPRIS_SERVICE_KEY")
    if not kipris_service_key:
        print("Warning: KIPRIS_SERVICE_KEY not found in .env file. Patent search may fail or use limited data.")

    print("--- Initializing PatentSearchAgent ---")
    print(f"Keywords: {args.keywords}")
    print(f"Year Range: ({args.year_start}, {args.year_end})")
    print(f"Max Patents per Keyword: {args.max_patents_per_keyword}")
    print(f"Embedding Model: {embedding_model}")
    print(f"ChromaDB Path: {db_path}")
    
    agent = PatentSearchAgent(db_path=db_path, embedding_model_name=embedding_model)

    print("\n--- Running Patent Search ---")
    try:
        result = agent.search_and_store(
            keywords=args.keywords,
            year_range=(args.year_start, args.year_end),
            max_patents_per_keyword=args.max_patents_per_keyword
        )
    except Exception as e:
        print(f"Error during patent search: {e}")
        sys.exit(1)

    print("\n--- Patent Search Agent Results ---")
    print("\nPatent List:")
    if result.get("patent_list"):
        for patent in result["patent_list"]:
            pprint.pprint(patent, indent=2, width=120)
            print("-" * 30)
    else:
        print("No patents found or returned.")

    print("\nRetriever Object:")
    if result.get("retriever"):
        print(f"Retriever type: {type(result['retriever'])}")
        # print(f"Retriever search_kwargs: {result['retriever'].search_kwargs}") # 상세 정보
        # retriever.invoke("테스트 검색어") 등으로 테스트 가능
    else:
        print("Retriever not created.")
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main() 