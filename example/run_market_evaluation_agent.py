import os
import sys
import argparse
import json
import yaml
from dotenv import load_dotenv
import logging
import pprint

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.patent_search_agent import PatentSearchAgent # patent_list 생성을 위해 필요
from agents.market_evaluation_agent import MarketEvaluationAgent

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# JSON 파일 로드 함수
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Market Evaluation Agent Individually")
    parser.add_argument("-pl", "--patent_list_file", type=str, default=None, help="Path to a JSON file containing patent_list. If not provided, PatentSearchAgent will be run.")
    parser.add_argument("-k", "--keywords", nargs='+', default=["RAG"], help="Keywords for PatentSearchAgent if patent_list_file is not provided.")
    parser.add_argument("-ys", "--year_start", type=int, default=0, help="Start year for PatentSearchAgent.")
    parser.add_argument("-ye", "--year_end", type=int, default=0, help="End year for PatentSearchAgent.")
    parser.add_argument("-mpk", "--max_patents_per_keyword", type=int, default=1, help="Max patents for PatentSearchAgent.")
    parser.add_argument("-cfg", "--app_config_file", type=str, default="config/config.yaml", help="Path to the main config.yaml file relative to project root")
    parser.add_argument("-p_dir", "--prompt_dir", type=str, default="prompts", help="Directory for prompts relative to project root")
    parser.add_argument("-c_dir", "--config_dir", type=str, default="config", help="Directory for config files relative to project root")
    parser.add_argument("-db", "--db_path", type=str, default=None, help="ChromaDB persist directory for PatentSearchAgent (if run). Overrides .env CHROMA_DB_PATH.")
    parser.add_argument("--no-tavily", action="store_true", help="Disable Tavily API search and use simulated data")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_config_file_path = os.path.join(project_root, args.app_config_file)
    prompt_dir_path = os.path.join(project_root, args.prompt_dir)
    config_dir_path = os.path.join(project_root, args.config_dir)

    if not os.path.exists(app_config_file_path):
        logger.error(f"App config file not found at {app_config_file_path}")
        sys.exit(1)
    if not os.path.exists(prompt_dir_path):
        logger.error(f"Prompt directory not found at {prompt_dir_path}")
        sys.exit(1)
    if not os.path.exists(config_dir_path):
        logger.error(f"Config directory not found at {config_dir_path}")
        sys.exit(1)
    
    app_config_data = load_config(app_config_file_path)
    embedding_model = app_config_data.get("embedding_model_ko", "jhgan/ko-sroberta-multitask")
    db_path = args.db_path if args.db_path else os.getenv("CHROMA_DB_PATH", "./data/embeddings/")
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)

    # API 키 체크
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in .env file.")
        sys.exit(1)
    if not tavily_api_key and not args.no_tavily:
        logger.warning("TAVILY_API_KEY not found in .env file. Web search will use simulated data.")

    # 특허 리스트 준비
    patent_list = None
    if args.patent_list_file:
        patent_list_file_path = os.path.join(project_root, args.patent_list_file)
        if not os.path.exists(patent_list_file_path):
            logger.error(f"Patent list file not found at {patent_list_file_path}")
            sys.exit(1)
        logger.info(f"Loading patent_list from: {patent_list_file_path}")
        patent_list = load_json_file(patent_list_file_path)
    else:
        logger.info("--- Running PatentSearchAgent to get patent_list ---")
        # MarketEvaluationAgent는 retriever가 필요 없으므로, PatentSearchAgent의 retriever 결과는 사용 안 함.
        search_agent = PatentSearchAgent(db_path=db_path, embedding_model_name=embedding_model)
        search_result = search_agent.search_and_store(
            keywords=args.keywords,
            year_range=(args.year_start, args.year_end),
            max_patents_per_keyword=args.max_patents_per_keyword
        )
        patent_list = search_result["patent_list"]

    if not patent_list:
        logger.error("Failed to obtain patent_list. Exiting.")
        sys.exit(1)

    # Market Evaluation 설정 준비
    # Tavily API 사용 여부 설정
    if args.no_tavily:
        # 설정 파일에 비활성화 플래그 추가
        market_config_path = os.path.join(config_dir_path, "market_eval_config.yaml")
        try:
            if os.path.exists(market_config_path):
                with open(market_config_path, 'r', encoding='utf-8') as f:
                    market_config = yaml.safe_load(f)
                
                market_config["api"]["use_tavily"] = False
                
                with open(market_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(market_config, f, default_flow_style=False)
                logger.info("Updated market_eval_config.yaml to disable Tavily API")
        except Exception as e:
            logger.error(f"Error updating market_eval_config.yaml: {e}")

    logger.info(f"--- Initializing MarketEvaluationAgent with {len(patent_list)} patents ---")
    market_agent = MarketEvaluationAgent(
        llm_api_key=openai_api_key, 
        prompt_dir=prompt_dir_path,
        config_dir=config_dir_path
    )

    logger.info("--- Running Market Evaluation ---")
    try:
        market_eval_result = market_agent.evaluate(patent_list=patent_list)
    except Exception as e:
        logger.error(f"Error during market evaluation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Market Evaluation Agent Results ---")
    if market_eval_result.get("market_eval_results"):
        results = market_eval_result["market_eval_results"]
        
        # 평균 점수 정보 출력
        logger.info(f"Average Market Size Score: {results.get('avg_market_size_score', 0)}/5.0")
        logger.info(f"Average Growth Potential Score: {results.get('avg_growth_potential_score', 0)}/5.0")
        
        # 주요 특허 정보 출력
        logger.info("Top Patents:")
        for i, patent in enumerate(results.get("patents", [])[:3], 1):  # 상위 3개만 출력
            logger.info(f"  {i}. Patent ID: {patent.get('patent_id')}")
            logger.info(f"     Title: {patent.get('title', '')[:80]}...")
            logger.info(f"     Market Size Score: {patent.get('market_size_score')}/5.0")
            logger.info(f"     Growth Potential Score: {patent.get('growth_potential_score')}/5.0")
            logger.info(f"     Expanded Keywords: {', '.join(patent.get('expanded_keywords', [])[:3])}...")
        
        # 결과를 JSON 파일로 저장
        output_file_name = f"{os.path.splitext(os.path.basename(args.patent_list_file))[0]}_market_eval.json" if args.patent_list_file else "market_evaluation_output.json"
        output_file = os.path.join(project_root, "example", output_file_name)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(market_eval_result["market_eval_results"], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    else:
        logger.warning("No market evaluation results produced.")
        
    logger.info("--- Script Finished ---")

if __name__ == "__main__":
    main() 