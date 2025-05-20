import os
import sys
import argparse
import json
import yaml
from dotenv import load_dotenv
import pprint

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.report_generator_agent import ReportGeneratorAgent

# .env 파일 로드 (API 키는 ReportGeneratorAgent에서 직접 사용하지 않지만, 일관성을 위해 로드)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# JSON 파일 로드 함수
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 설정 파일 로드 함수
def load_app_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Report Generator Agent Individually")
    parser.add_argument("-f", "--fit_eval_file", required=True, help="Path to JSON file with fit_eval_results")
    parser.add_argument("-t", "--tech_eval_file", required=True, help="Path to JSON file with tech_eval_results")
    parser.add_argument("-m", "--market_eval_file", required=True, help="Path to JSON file with market_eval_results")
    parser.add_argument("-p", "--persona", type=str, default=None, help="Persona for report generation (e.g., investor, strategy). Uses default from config if not provided.")
    parser.add_argument("-cfg_dir", "--config_dir", type=str, default="config", help="Directory for config files (persona_profiles.yaml) relative to project root")
    parser.add_argument("-app_cfg", "--app_config_file", type=str, default="config/config.yaml", help="Path to the main app config.yaml for default_persona, relative to project root")
    parser.add_argument("-p_dir", "--prompt_dir", type=str, default="prompts", help="Directory for prompt templates relative to project root")
    parser.add_argument("-o_dir", "--output_dir", type=str, default="example/generated_reports", help="Directory to save generated reports, relative to project root")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fit_eval_file_path = os.path.join(project_root, args.fit_eval_file)
    tech_eval_file_path = os.path.join(project_root, args.tech_eval_file)
    market_eval_file_path = os.path.join(project_root, args.market_eval_file)
    config_dir_path = os.path.join(project_root, args.config_dir)
    prompt_dir_path = os.path.join(project_root, args.prompt_dir)
    output_dir_path = os.path.join(project_root, args.output_dir)
    app_config_file_path = os.path.join(project_root, args.app_config_file)

    for f_path in [fit_eval_file_path, tech_eval_file_path, market_eval_file_path, config_dir_path, prompt_dir_path, app_config_file_path]:
        if not os.path.exists(f_path):
            print(f"Error: Required file or directory not found at {f_path}")
            sys.exit(1)
            
    os.makedirs(output_dir_path, exist_ok=True)

    fit_eval_results = load_json_file(fit_eval_file_path)
    tech_eval_results = load_json_file(tech_eval_file_path)
    market_eval_results = load_json_file(market_eval_file_path)
    
    persona_to_use = args.persona
    if not persona_to_use:
        app_cfg_data = load_app_config(app_config_file_path)
        persona_to_use = app_cfg_data.get("default_persona", "investor")

    print(f"--- Initializing ReportGeneratorAgent with Persona: {persona_to_use} ---")
    report_agent = ReportGeneratorAgent(prompt_dir=prompt_dir_path, config_dir=config_dir_path, output_dir=output_dir_path)

    print("--- Generating Report ---")
    try:
        report_generation_result = report_agent.generate(
            fit_eval_results=fit_eval_results,
            tech_eval_results=tech_eval_results,
            market_eval_results=market_eval_results,
            persona=persona_to_use
        )
    except Exception as e:
        print(f"Error during report generation: {e}")
        sys.exit(1)

    print("\n--- Report Generator Agent Results ---")
    if report_generation_result and report_generation_result.get("report_path"):
        print(f"Report generated successfully at: {report_generation_result['report_path']}")
        # 터미널에 보고서 내용 일부 출력 (선택적)
        try:
            with open(report_generation_result['report_path'], 'r', encoding='utf-8') as f_report:
                print("\nReport Content (first 500 chars):")
                print(f_report.read(500) + "...")
        except Exception as e:
            print(f"Could not read report content: {e}")
    else:
        print("Report generation failed or path not returned.")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main() 