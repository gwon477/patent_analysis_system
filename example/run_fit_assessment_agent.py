import os
import sys
import argparse
import json
import yaml
from dotenv import load_dotenv
import pprint

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.fit_assessment_agent import FitAssessmentAgent

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# JSON 파일 로드 함수
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 설정 파일 로드 함수 (FitAssessmentAgent 내부에서도 사용되지만, persona 기본값 등을 위해 여기서도 필요시 사용)
def load_app_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Fit Assessment Agent Individually")
    parser.add_argument("-t", "--tech_eval_file", type=str, default="example/tech_evaluation_output.json", 
                        help="Path to JSON file with tech_eval_results (default: example/tech_evaluation_output.json)")
    parser.add_argument("-m", "--market_eval_file", type=str, default="example/market_evaluation_output.json", 
                        help="Path to JSON file with market_eval_results (default: example/market_evaluation_output.json)")
    parser.add_argument("-p", "--persona", type=str, default=None, 
                        help="Persona for assessment (e.g., investor, strategy). Uses default from config if not provided.")
    parser.add_argument("-cfg_dir", "--config_dir", type=str, default="config", 
                        help="Directory for config files (weights.yaml, persona_profiles.yaml) relative to project root")
    parser.add_argument("-app_cfg", "--app_config_file", type=str, default="config/config.yaml", 
                        help="Path to the main app config.yaml for default_persona, relative to project root")
    parser.add_argument("-p_dir", "--prompt_dir", type=str, default="prompts", 
                        help="Directory for prompts relative to project root")
    
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tech_eval_file_path = os.path.join(project_root, args.tech_eval_file)
    market_eval_file_path = os.path.join(project_root, args.market_eval_file)
    config_dir_path = os.path.join(project_root, args.config_dir)
    prompt_dir_path = os.path.join(project_root, args.prompt_dir)
    app_config_file_path = os.path.join(project_root, args.app_config_file)

    # 필수 파일/디렉토리 존재 확인
    required_paths = [config_dir_path, prompt_dir_path, app_config_file_path]
    
    for f_path in required_paths:
        if not os.path.exists(f_path):
            print(f"Error: Required directory or file not found at {f_path}")
            sys.exit(1)
    
    # 평가 파일 존재 확인
    for eval_path in [tech_eval_file_path, market_eval_file_path]:
        if not os.path.exists(eval_path):
            print(f"Error: Evaluation file not found at {eval_path}")
            print(f"Please check the path or run the corresponding evaluation agent first.")
            sys.exit(1)

    # 평가 결과 로드
    try:
        tech_eval_results = load_json_file(tech_eval_file_path)
        market_eval_results = load_json_file(market_eval_file_path)
        print(f"Successfully loaded tech evaluation results from: {tech_eval_file_path}")
        print(f"Successfully loaded market evaluation results from: {market_eval_file_path}")
    except Exception as e:
        print(f"Error loading evaluation files: {e}")
        sys.exit(1)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        sys.exit(1)
        
    # 페르소나 결정 (CLI 인자 > app_config의 default_persona > FitAssessmentAgent 기본값)
    persona_to_use = args.persona
    if not persona_to_use:
        app_cfg_data = load_app_config(app_config_file_path)
        persona_to_use = app_cfg_data.get("default_persona", "investor") # FitAssessmentAgent의 기본값과 일치시킴

    print(f"--- Initializing FitAssessmentAgent with Persona: {persona_to_use} ---")
    fit_agent = FitAssessmentAgent(llm_api_key=openai_api_key, prompt_dir=prompt_dir_path, config_dir=config_dir_path)

    print("--- Running Fit Assessment ---")
    try:
        fit_assessment_result = fit_agent.assess(
            tech_eval_results=tech_eval_results,
            market_eval_results=market_eval_results,
            persona=persona_to_use
        )
    except Exception as e:
        print(f"Error during fit assessment: {e}")
        sys.exit(1)

    print("\n--- Fit Assessment Agent Results ---")
    if fit_assessment_result.get("fit_eval_results"):
        pprint.pprint(fit_assessment_result["fit_eval_results"], indent=2, width=120)
        # 결과를 JSON 파일로 저장 (선택적)
        output_file = os.path.join(project_root, "example", "fit_assessment_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(fit_assessment_result["fit_eval_results"], f, ensure_ascii=False, indent=2)
        print(f"\nResults also saved to: {output_file}")
    else:
        print("No fit assessment results produced.")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main() 