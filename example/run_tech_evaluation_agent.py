import os
import sys
import argparse
import json
import yaml
from dotenv import load_dotenv
import pprint
import logging # 로깅 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.patent_search_agent import PatentSearchAgent # Retriever 생성을 위해 필요
from agents.tech_evaluation_agent import TechEvaluationAgent

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# JSON 파일 로드 함수 (patent_list용)
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Tech Evaluation Agent Individually")
    parser.add_argument("-pl", "--patent_list_file", type=str, default=None, help="Path to a JSON file containing patent_list. If not provided, PatentSearchAgent will be run.")
    parser.add_argument("-k", "--keywords", nargs='+', default=["RAG"], help="Keywords for PatentSearchAgent if patent_list_file is not provided, and for TechEvaluationAgent.")
    parser.add_argument("-ys", "--year_start", type=int, default=0, help="Start year for PatentSearchAgent.")
    parser.add_argument("-ye", "--year_end", type=int, default=0, help="End year for PatentSearchAgent.")
    parser.add_argument("-cfg", "--app_config_file", type=str, default="config/config.yaml", help="Path to the main config.yaml file relative to project root")
    parser.add_argument("-p_dir", "--prompt_dir", type=str, default="prompts", help="Directory for prompts relative to project root")
    parser.add_argument("-db", "--db_path", type=str, default=None, help="ChromaDB persist directory. Overrides .env CHROMA_DB_PATH if set.")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_config_file_path = os.path.join(project_root, args.app_config_file)
    prompt_dir_path = os.path.join(project_root, args.prompt_dir)

    if not os.path.exists(app_config_file_path):
        logger.error(f"App config file not found at {app_config_file_path}")
        sys.exit(1)
    if not os.path.exists(prompt_dir_path):
        logger.error(f"Prompt directory not found at {prompt_dir_path}")
        sys.exit(1)

    app_config_data = load_config(app_config_file_path)
    embedding_model_name = app_config_data.get("embedding_model_ko", "jhgan/ko-sroberta-multitask")
    db_path = args.db_path if args.db_path else os.getenv("CHROMA_DB_PATH", "./data/embeddings/")
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in .env file.")
        sys.exit(1)

    patent_list = None
    retriever = None
    search_keywords = args.keywords

    if args.patent_list_file:
        patent_list_file_path = os.path.join(project_root, args.patent_list_file)
        if not os.path.exists(patent_list_file_path):
            logger.error(f"Patent list file not found at {patent_list_file_path}")
            sys.exit(1)
        logger.info(f"Loading patent_list from: {patent_list_file_path}")
        patent_list = load_json_file(patent_list_file_path)
        
        logger.info("Initializing PatentSearchAgent to create retriever (assuming patents from file are in DB)... ")
        search_agent = PatentSearchAgent(db_path=db_path, embedding_model_name=embedding_model_name)
        retriever = search_agent.chroma.as_retriever(search_kwargs={"k": 10}) 
        logger.info(f"Retriever created from existing DB at {db_path}")

        if not patent_list:
            logger.warning("Patent list file was specified but failed to load or was empty.")
    else:
        logger.info("--- Running PatentSearchAgent to get patent_list and retriever ---")
        search_agent = PatentSearchAgent(db_path=db_path, embedding_model_name=embedding_model_name)
        search_result = search_agent.search_and_store(
            keywords=search_keywords,
            year_range=(args.year_start, args.year_end)
        )
        patent_list = search_result["patent_list"]
        retriever = search_result["retriever"]

    if not patent_list:
        logger.error("Patent list is empty. Cannot proceed with technical evaluation.")
        sys.exit(1)
    if not retriever:
        logger.error("Failed to obtain retriever. Exiting.")
        sys.exit(1)

    logger.info(f"--- Initializing TechEvaluationAgent with {len(patent_list)} patents ---")
    logger.info(f"Keywords for tech trend analysis: {search_keywords}")

    tech_agent = TechEvaluationAgent(
        retriever=retriever, 
        llm_api_key=openai_api_key, 
        prompt_dir=prompt_dir_path
        # embedding_model_name은 TechEvaluationAgent에서 직접 사용하지 않으므로 제거
    )

    logger.info("--- Running Technical Evaluation ---")
    try:
        tech_eval_result = tech_agent.evaluate(patent_list=patent_list, keywords=search_keywords)
        logger.info(f"Successfully evaluated {len(tech_eval_result.get('tech_eval_results', {}).get('patents', []))} patents")
        
        # 새로운 결과 형식에 맞게 처리
        if tech_eval_result.get('tech_eval_results'):
            tech_eval_data = tech_eval_result['tech_eval_results']
            patents = tech_eval_data.get('patents', [])
            trend_keywords = tech_eval_data.get('trend_keywords', [])
            avg_trend_fit_score = tech_eval_data.get('avg_trend_fit_score', 0)
            avg_originality_score = tech_eval_data.get('avg_originality_score', 0)
            
            logger.info(f"Trend keywords: {trend_keywords}")
            logger.info(f"Average scores - Originality: {avg_originality_score:.2f}, Trend fit: {avg_trend_fit_score:.2f}")
            
            # 점수 범위 계산
            if patents:
                orig_scores = [p.get('originality_score', 0) for p in patents]
                trend_scores = [p.get('trend_fit_score', 0) for p in patents]
                
                if orig_scores:
                    logger.info(f"Originality scores: min={min(orig_scores):.2f}, max={max(orig_scores):.2f}, avg={sum(orig_scores)/len(orig_scores):.2f}")
                if trend_scores:
                    logger.info(f"Trend fit scores: min={min(trend_scores):.2f}, max={max(trend_scores):.2f}, avg={sum(trend_scores)/len(trend_scores):.2f}")
    except Exception as e:
        logger.error(f"Error during technical evaluation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Tech Evaluation Agent Results ---")
    if tech_eval_result and tech_eval_result.get("tech_eval_results"):
        output_file_name = "tech_evaluation_output.json"
        if args.patent_list_file:
            base_name = os.path.basename(args.patent_list_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file_name = f"{name_without_ext}_tech_eval.json"
        
        output_file = os.path.join(project_root, "example", output_file_name)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tech_eval_result["tech_eval_results"], f, ensure_ascii=False, indent=2)
        logger.info(f"Results also saved to: {output_file}")
        
        # 특허 정보 출력 수정
        patents = tech_eval_result["tech_eval_results"].get("patents", [])
        for patent in patents[:3]:  # 처음 3개 특허만 표시
            logger.info(f"  Patent ID: {patent.get('patent_id')}, Originality: {patent.get('originality_score')}, Trend Fit: {patent.get('trend_fit_score')}")
            logger.info(f"    Title: {patent.get('title', '')[:100]}...")
            logger.info(f"    Summary: {patent.get('tech_summary', '')[:100]}...")
    else:
        logger.warning("No technical evaluation results produced.")
    
    logger.info("--- Script Finished ---")

if __name__ == "__main__":
    main() 