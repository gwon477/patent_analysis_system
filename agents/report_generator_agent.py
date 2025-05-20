from utils.jinja2_loader import load_jinja2_template
import yaml
import os
from datetime import datetime

# FitAssessmentAgent와 동일한 함수 사용 또는 별도 유틸리티로 분리 가능
def load_persona_report_detail(persona_name="default", config_dir="./config"):
    persona_profiles_path = f"{config_dir}/persona_profiles.yaml"
    report_detail = "summary" # 기본값
    language = "ko" # 기본값
    try:
        with open(persona_profiles_path, 'r', encoding='utf-8') as f:
            personas = yaml.safe_load(f)
        if persona_name in personas:
            report_detail = personas[persona_name].get('report_detail', report_detail)
            language = personas[persona_name].get('language', language)
            print(f"[ReportGeneratorAgent] Using report settings for persona: {persona_name} (Detail: {report_detail}, Lang: {language})")
        elif persona_name != "default":
            print(f"Warning: Persona '{persona_name}' not found in profiles. Using default report settings.")
    except FileNotFoundError:
        print(f"Warning: Persona profiles file not found at {persona_profiles_path}. Using default report settings.")
    return report_detail, language

class ReportGeneratorAgent:
    def __init__(self, prompt_dir="./prompts", config_dir="./config", output_dir="./reports"):
        self.prompt_dir = prompt_dir
        self.config_dir = config_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, fit_eval_results, tech_eval_results, market_eval_results, persona="investor"):
        if not fit_eval_results:
            print("[ReportGeneratorAgent] Fit evaluation results are missing. Cannot generate report.")
            return {"report_path": None}

        report_detail, lang = load_persona_report_detail(persona, self.config_dir)
        template_name = f"report_template_{lang}.j2"
        
        try:
            report_template = load_jinja2_template(self.prompt_dir, template_name)
        except Exception as e:
            print(f"[ReportGeneratorAgent] Error loading template {template_name}: {e}. Falling back to ko template.")
            # Fallback to Korean template if specified language template is not found
            report_template = load_jinja2_template(self.prompt_dir, "report_template_ko.j2")
            lang = "ko"

        # Get all patents from fit_eval_results
        all_patents = fit_eval_results.get("patents", [])
        
        # Sort patents by total_score in descending order
        sorted_patents = sorted(all_patents, key=lambda x: float(x.get('total_score', 0)), reverse=True)
        
        # Select top 5 patents
        top_patents = sorted_patents[:min(5, len(sorted_patents))]
        
        # Create mapping of tech and market evaluations for faster lookup
        # tech_eval_results와 market_eval_results 처리 시 안전 장치 추가
        tech_map = {}
        market_map = {}
        
        # tech_eval_results가 리스트이고 비어있지 않은 경우에만 처리
        if isinstance(tech_eval_results, list) and tech_eval_results:
            tech_map = {t.get("patent_id", ""): t for t in tech_eval_results if isinstance(t, dict) and t.get("patent_id")}
        
        # market_eval_results가 리스트이고 비어있지 않은 경우에만 처리
        if isinstance(market_eval_results, list) and market_eval_results:
            market_map = {m.get("patent_id", ""): m for m in market_eval_results if isinstance(m, dict) and m.get("patent_id")}
        
        # Enhanced data for detailed report
        enhanced_top_patents = []
        for patent in top_patents:
            patent_id = patent.get("patent_id", "")
            tech_eval = tech_map.get(patent_id, {})
            market_eval = market_map.get(patent_id, {})
            
            # Calculate tech_score and market_score (average of component scores)
            try:
                tech_scores = patent.get("tech_scores", {})
                if isinstance(tech_scores, dict):
                    originality_score = float(tech_scores.get("originality_score", 0))
                    trend_fit_score = float(tech_scores.get("trend_fit_score", 0))
                    tech_score = round((originality_score + trend_fit_score) / 2, 2)
                else:
                    tech_score = 0.0
            except (ValueError, TypeError):
                tech_score = 0.0
            
            try:
                market_scores = patent.get("market_scores", {})
                if isinstance(market_scores, dict):
                    market_size_score = float(market_scores.get("market_size_score", 0))
                    growth_potential_score = float(market_scores.get("growth_potential_score", 0))
                    market_score = round((market_size_score + growth_potential_score) / 2, 2)
                else:
                    market_score = 0.0
            except (ValueError, TypeError):
                market_score = 0.0
            
            # Create enhanced patent data with summarized action
            action_summary = str(patent.get("recommended_action", "N/A"))
            if len(action_summary) > 50:  # Truncate if too long for table display
                action_summary = action_summary[:50] + "..."
            
            enhanced_patent = {
                "patent_id": patent_id,
                "title": patent.get("title", "N/A"),
                "tech_score": tech_score,
                "market_score": market_score,
                "alignment_score": patent.get("alignment_score", "N/A"),
                "total_score": patent.get("total_score", "N/A"),
                "grade": patent.get("grade", "N/A"),
                "tech_scores": patent.get("tech_scores", {}),
                "market_scores": patent.get("market_scores", {}),
                "tech_summary": patent.get("tech_summary", "N/A"),
                "market_summary": patent.get("market_summary", "N/A"),
                "alignment_reason": patent.get("alignment_reason", "N/A"),
                "recommended_action": patent.get("recommended_action", "N/A"),
                "action_summary": action_summary
            }
            
            enhanced_top_patents.append(enhanced_patent)
        
        # Add current date for the report
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # Prepare data for the template
        template_data = {
            "top_patents": enhanced_top_patents,
            "persona": persona,
            "lang": lang,
            "today_date": today_date,
            "overall_assessment": fit_eval_results.get("overall_assessment", "")
        }
        
        # Generate report content using the template
        report_content = report_template.render(**template_data)
        
        # Generate file name with persona and language
        report_filename = f"patent_evaluation_report_{persona}_{lang}.md"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Write report to file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"[ReportGeneratorAgent] Report generated: {report_path}")
        return {"report_path": report_path} 