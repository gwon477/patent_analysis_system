# from langchain.llms import OpenAI # OpenAI는 langchain_openai.OpenAI로 변경됨
# from langchain_openai import OpenAI # 수정된 임포트
# from langchain.prompts import PromptTemplate # PromptTemplate 직접 사용 안 함
from utils.jinja2_loader import load_jinja2_template
import yaml # For loading weights from config
import logging
import json
import os
import openai
from typing import List, Dict, Any, Optional, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# 가중치와 페르소나 설정을 로드하는 함수 (필요시 main 또는 config 모듈로 이동 가능)
def load_weights_and_persona_config(persona_name="default", config_dir="./config"):
    default_weights_path = f"{config_dir}/weights.yaml"
    persona_profiles_path = f"{config_dir}/persona_profiles.yaml"

    try:
        with open(default_weights_path, 'r', encoding='utf-8') as f:
            default_weights = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Default weights file not found at {default_weights_path}. Using fallback.")
        default_weights = {'originality': 0.25, 'trend_fit': 0.25, 'market_size': 0.25, 'growth_potential': 0.25}

    selected_weights = default_weights
    report_detail = "summary" # 기본값

    try:
        with open(persona_profiles_path, 'r', encoding='utf-8') as f:
            personas = yaml.safe_load(f)
        if persona_name in personas and 'weights' in personas[persona_name]:
            selected_weights = personas[persona_name]['weights']
            report_detail = personas[persona_name].get('report_detail', report_detail)
            print(f"[FitAssessmentAgent] Using weights for persona: {persona_name}")
        elif persona_name != "default":
            print(f"Warning: Persona '{persona_name}' not found or has no weights. Using default weights.")
    except FileNotFoundError:
        print(f"Warning: Persona profiles file not found at {persona_profiles_path}. Using default weights.")
        
    return selected_weights, report_detail

class FitAssessmentAgent:
    """
    기술 평가와 시장 평가 결과를 결합하여 종합적인 평가를 수행하는 에이전트
    """
    
    def __init__(self, llm_api_key, prompt_dir="./prompts", config_dir="./config"):
        """
        Args:
            llm_api_key: OpenAI API 키
            prompt_dir: 프롬프트 템플릿 디렉토리
            config_dir: 설정 파일 디렉토리
        """
        self.api_key = llm_api_key
        # 최신 OpenAI 라이브러리 호환성을 위한 수정
        try:
            # proxies 관련 설정을 제거하고 필수 매개변수만 전달
            self.client = openai.OpenAI(api_key=llm_api_key, default_headers=None, max_retries=2)
        except Exception as e:
            logger.warning(f"[FitAssessmentAgent] OpenAI 클라이언트 초기화 오류: {e}")
            # 기본 매개변수만 사용
            self.client = openai.Client(api_key=llm_api_key)
        
        # 프롬프트 템플릿 로드
        try:
            # 기존 로직 수정: fit_assessment_prompt.j2 대신 fit_alignment_prompt.j2 사용
            self.fit_prompt_template = load_jinja2_template(prompt_dir, "fit_alignment_prompt.j2")
            logger.info(f"[FitAssessmentAgent] Loaded template fit_alignment_prompt.j2")
        except Exception as e:
            logger.error(f"[FitAssessmentAgent] 템플릿 로드 오류: {e}")
            # 템플릿 로드 실패 시에도 작동할 수 있도록 기본 프롬프트 설정
            logger.info(f"[FitAssessmentAgent] Using default prompt instead")
            self.fit_prompt_template = None
        
        # 설정 로드
        self.config = self._load_config(config_dir)
        
        # 등급 기준 설정
        self.grade_thresholds = self.config.get("grade_thresholds", {
            "high": 4.2,
            "medium": 3.5
        })
        
        # 가중치 설정 (기본값)
        self.default_weights = self.config.get("default_weights", {
            "originality_score": 0.25,
            "trend_fit_score": 0.25,
            "market_size_score": 0.25,
            "growth_potential_score": 0.25
        })
        
        # 페르소나별 가중치 설정
        self.persona_weights = self.config.get("persona_weights", {
            "investor": {
                "originality_score": 0.2,
                "trend_fit_score": 0.2,
                "market_size_score": 0.3,
                "growth_potential_score": 0.3
            },
            "strategy": {
                "originality_score": 0.3,
                "trend_fit_score": 0.3,
                "market_size_score": 0.2,
                "growth_potential_score": 0.2
            },
            "r&d": {
                "originality_score": 0.4,
                "trend_fit_score": 0.3,
                "market_size_score": 0.1,
                "growth_potential_score": 0.2
            }
        })
        
        # 결과 저장 경로
        self.output_dir = "./data/results"
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_config(self, config_dir: str) -> Dict[str, Any]:
        """설정 파일을 로드합니다."""
        config_path = os.path.join(config_dir, "fit_assessment_config.yaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"[FitAssessmentAgent] Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"[FitAssessmentAgent] Config file not found at {config_path}. Using defaults.")
            # 기본 설정
            return {
                "grade_thresholds": {
                    "high": 4.2,
                    "medium": 3.5
                },
                "default_weights": {
                    "originality_score": 0.25,
                    "trend_fit_score": 0.25,
                    "market_size_score": 0.25,
                    "growth_potential_score": 0.25
                },
                "persona_weights": {
                    "investor": {
                        "originality_score": 0.2,
                        "trend_fit_score": 0.2,
                        "market_size_score": 0.3,
                        "growth_potential_score": 0.3
                    },
                    "strategy": {
                        "originality_score": 0.3,
                        "trend_fit_score": 0.3,
                        "market_size_score": 0.2,
                        "growth_potential_score": 0.2
                    },
                    "r&d": {
                        "originality_score": 0.4,
                        "trend_fit_score": 0.3,
                        "market_size_score": 0.1,
                        "growth_potential_score": 0.2
                    }
                }
            }
    
    def assess(self, tech_eval_results: Dict[str, Any], market_eval_results: Dict[str, Any], persona: str = "investor") -> Dict[str, Any]:
        """
        기술 평가와 시장 평가 결과를 결합하여 종합 평가를 수행합니다.
        
        Args:
            tech_eval_results: 기술 평가 결과
            market_eval_results: 시장 평가 결과
            persona: 평가 페르소나 (investor, strategy, r&d)
            
        Returns:
            Dict: fit_eval_results 키를 포함하는 딕셔너리
        """
        logger.info(f"[FitAssessmentAgent] Starting assessment with persona: {persona}")
        
        # 페르소나 가중치 선택
        weights = self.persona_weights.get(persona.lower(), self.default_weights)
        logger.info(f"[FitAssessmentAgent] Using weights for persona {persona}: {weights}")
        
        # 결과 통합
        combined_results = self._combine_results(tech_eval_results, market_eval_results)
        
        # 종합 점수 계산 및 등급 산출
        assessment_results = []
        total_score_sum = 0
        
        for patent in combined_results:
            # 종합 점수 계산 - 이제 _assess_alignment에서 수행됨
            weighted_score = self._calculate_total_score(patent, weights)
            
            # 정합성 평가 - 향상된 버전으로 종합 점수, 등급, 추천 액션까지 한번에 생성
            alignment_score, alignment_reason, llm_total_score, grade, recommended_action = self._assess_alignment(
                patent.get("tech_summary", ""),
                patent.get("market_summary", ""),
                patent.get("patent_id", ""),
                patent.get("title", ""),
                patent.get("originality_score", 0),
                patent.get("trend_fit_score", 0),
                patent.get("market_size_score", 0),
                patent.get("growth_potential_score", 0),
                persona
            )
            
            # 양쪽에서 계산된 점수의 평균을 최종 점수로 사용 (차이가 클 경우 로그 기록)
            total_score = (weighted_score + llm_total_score) / 2
            if abs(weighted_score - llm_total_score) > 1.0:
                logger.warning(f"[FitAssessmentAgent] Large difference in score calculation for patent {patent.get('patent_id')}: weighted={weighted_score:.2f}, llm={llm_total_score:.2f}")
            
            total_score_sum += total_score
            
            # 결과 저장
            assessment = {
                "patent_id": patent.get("patent_id"),
                "title": patent.get("title"),
                "total_score": round(total_score, 2),
                "weighted_score": round(weighted_score, 2), # 원래 가중치로 계산된 점수도 저장
                "llm_total_score": round(llm_total_score, 2), # LLM이 생성한 점수도 저장
                "alignment_score": alignment_score,
                "alignment_reason": alignment_reason,
                "grade": grade,
                "recommended_action": recommended_action,
                "tech_scores": {
                    "originality_score": patent.get("originality_score", 0),
                    "trend_fit_score": patent.get("trend_fit_score", 0)
                },
                "market_scores": {
                    "market_size_score": patent.get("market_size_score", 0),
                    "growth_potential_score": patent.get("growth_potential_score", 0)
                },
                "tech_summary": patent.get("tech_summary", ""),
                "market_summary": patent.get("market_summary", ""),
                "tech_details": patent.get("tech_details", {}),
                "market_details": patent.get("market_details", {})
            }
            
            assessment_results.append(assessment)
        
        # 전체 평균 점수
        avg_total_score = total_score_sum / len(assessment_results) if assessment_results else 0
        
        # 결과를 JSON 파일로 저장
        self._save_results_to_json(assessment_results)
        
        # 종합 평가 보고서 생성
        overall_assessment = self._generate_overall_assessment(assessment_results, avg_total_score, persona)
        
        logger.info(f"[FitAssessmentAgent] Assessment complete. Average total score: {avg_total_score:.2f}")
        
        return {
            "fit_eval_results": {
                "patents": assessment_results,
                "avg_total_score": round(avg_total_score, 2),
                "overall_assessment": overall_assessment,
                "persona": persona
            }
        }
    
    def _combine_results(self, tech_eval_results: Dict[str, Any], market_eval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """기술 평가와 시장 평가 결과를 특허 ID를 기준으로 결합합니다."""
        combined_results = []
        
        # 기술 평가 결과 딕셔너리 생성
        tech_dict = {patent.get("patent_id"): patent for patent in tech_eval_results.get("patents", [])}
        
        # 시장 평가 결과 딕셔너리 생성
        market_dict = {patent.get("patent_id"): patent for patent in market_eval_results.get("patents", [])}
        
        # 모든 특허 ID 수집
        all_patent_ids = set(tech_dict.keys()) | set(market_dict.keys())
        
        # 결과 결합
        for patent_id in all_patent_ids:
            tech_data = tech_dict.get(patent_id, {})
            market_data = market_dict.get(patent_id, {})
            
            combined_patent = {
                "patent_id": patent_id,
                "title": tech_data.get("title") or market_data.get("title", ""),
                "originality_score": tech_data.get("originality_score", 0),
                "trend_fit_score": tech_data.get("trend_fit_score", 0),
                "market_size_score": market_data.get("market_size_score", 0),
                "growth_potential_score": market_data.get("growth_potential_score", 0),
                "tech_summary": tech_data.get("tech_summary", ""),
                "market_summary": market_data.get("market_summary", ""),
                "tech_details": tech_data,
                "market_details": market_data
            }
            
            combined_results.append(combined_patent)
        
        logger.info(f"[FitAssessmentAgent] Combined {len(combined_results)} patent results")
        return combined_results
    
    def _calculate_total_score(self, patent: Dict[str, Any], weights: Dict[str, float]) -> float:
        """가중치를 적용하여 종합 점수를 계산합니다."""
        # 각 점수에 가중치 적용
        weighted_scores = []
        total_weight = 0
        
        for score_key, weight in weights.items():
            score = patent.get(score_key, 0)
            weighted_scores.append(score * weight)
            total_weight += weight
        
        # 가중 평균 계산
        if total_weight > 0:
            total_score = sum(weighted_scores) / total_weight
        else:
            total_score = 0
            
        return total_score
    
    def _assess_alignment(self, tech_summary: str, market_summary: str, patent_id: str = "", title: str = "", 
                       originality_score: float = 0, trend_fit_score: float = 0, 
                       market_size_score: float = 0, growth_potential_score: float = 0,
                       persona: str = "investor") -> Tuple[float, str, float, str, str]:
        """기술 요약과 시장 요약 간의 정합성을 평가합니다."""
        if not tech_summary or not market_summary:
            return 0.0, "기술 요약 또는 시장 요약이 누락되어 정합성을 평가할 수 없습니다.", 0.0, "하", "평가 불가: 필요한 정보가 누락되었습니다."
            
        try:
            # Jinja 템플릿을 사용하여 프롬프트 생성
            if self.fit_prompt_template:
                # 템플릿 렌더링에 필요한 모든 변수 전달
                prompt = self.fit_prompt_template.render(
                    tech_summary=tech_summary,
                    market_summary=market_summary,
                    patent_id=patent_id,
                    title=title,
                    originality_score=originality_score,
                    trend_fit_score=trend_fit_score,
                    market_size_score=market_size_score,
                    growth_potential_score=growth_potential_score,
                    persona=persona
                )
            else:
                # 템플릿이 로드되지 않은 경우 기본 프롬프트 사용
                prompt = f"""당신은 기술과 시장 간의 정합성을 평가하는 전문가입니다. 
다음 정보를 바탕으로 기술과 시장 간의 정합성을 평가해주세요:

[기술 요약]
{tech_summary}

[시장 요약]
{market_summary}

위 기술이 해당 시장에 얼마나 적합한지, 기술과 시장이 얼마나 잘 맞물리는지를 1점부터 5점까지의 척도로 평가하세요.
1점: 매우 낮은 정합성
2점: 낮은 정합성
3점: 보통 정합성
4점: 높은 정합성
5점: 매우 높은 정합성

응답 형식:
정합성_점수: [1-5 사이의 숫자]
정합성_이유: [간략한 평가 이유]
종합_점수: [종합 점수]
등급: [상/중/하]
추천_액션: [맞춤형 행동 권고사항]

평가 시 고려할 사항:
1. 기술이 시장의 주요 문제를 해결하는가?
2. 기술이 시장의 현재 트렌드와 일치하는가?
3. 기술이 시장의 성장 잠재력을 활용할 수 있는가?
4. 기술과 시장 간에 명확한 연결점이 있는가?
"""
            
            logger.info(f"[FitAssessmentAgent] Sending alignment assessment prompt for patent ID: {patent_id}")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content.strip()
            
            # 응답에서 점수, 이유, 종합 점수, 등급, 추천 액션 추출
            alignment_score_line = next((line for line in content.split('\n') if line.startswith('정합성_점수:')), '')
            alignment_reason_line = next((line for line in content.split('\n') if line.startswith('정합성_이유:')), '')
            total_score_line = next((line for line in content.split('\n') if line.startswith('종합_점수:')), '')
            grade_line = next((line for line in content.split('\n') if line.startswith('등급:')), '')
            action_line = next((line for line in content.split('\n') if line.startswith('추천_액션:')), '')
            
            # 점수 추출
            try:
                alignment_score = float(alignment_score_line.split(':')[1].strip())
            except (ValueError, IndexError):
                logger.warning(f"[FitAssessmentAgent] Error parsing alignment_score from: {alignment_score_line}")
                alignment_score = 3.0  # 기본값
                
            # 이유 추출
            try:
                alignment_reason = alignment_reason_line.split(':')[1].strip()
            except IndexError:
                logger.warning(f"[FitAssessmentAgent] Error parsing alignment_reason from: {alignment_reason_line}")
                alignment_reason = "분석 과정에서 명확한 이유를 추출할 수 없었습니다."
                
            # 종합 점수 추출
            try:
                total_score = float(total_score_line.split(':')[1].strip())
            except (ValueError, IndexError):
                logger.warning(f"[FitAssessmentAgent] Error parsing total_score from: {total_score_line}")
                # 기본 가중치 계산으로 대체
                total_score = (originality_score + trend_fit_score + market_size_score + growth_potential_score) / 4
            
            # 등급 추출
            try:
                grade = grade_line.split(':')[1].strip()
                if grade not in ["상", "중", "하"]:
                    grade = self._determine_grade(total_score)
            except IndexError:
                logger.warning(f"[FitAssessmentAgent] Error parsing grade from: {grade_line}")
                grade = self._determine_grade(total_score)
                
            # 추천 액션 추출
            try:
                recommended_action = action_line.split(':')[1].strip()
            except IndexError:
                logger.warning(f"[FitAssessmentAgent] Error parsing recommended_action from: {action_line}")
                recommended_action = f"점수({total_score:.1f})와 등급({grade})에 기반한 분석이 필요합니다."
                
            logger.info(f"[FitAssessmentAgent] Alignment assessment complete. Score: {alignment_score}, Grade: {grade}")
            return alignment_score, alignment_reason, total_score, grade, recommended_action
            
        except Exception as e:
            logger.error(f"[FitAssessmentAgent] Error assessing alignment: {e}")
            return 3.0, "정합성 평가 중 오류가 발생했습니다.", 3.0, "중", "평가 중 오류가 발생했습니다. 재평가가 필요합니다."
    
    def _determine_grade(self, total_score: float) -> str:
        """종합 점수를 바탕으로 등급을 결정합니다."""
        if total_score >= self.grade_thresholds.get("high", 4.2):
            return "상"
        elif total_score >= self.grade_thresholds.get("medium", 3.5):
            return "중"
        else:
            return "하"
    
    def _generate_overall_assessment(self, assessment_results: List[Dict[str, Any]], avg_total_score: float, persona: str) -> str:
        """전체 평가 결과를 바탕으로 종합 평가 보고서를 생성합니다."""
        # 상위 3개 특허 추출
        sorted_patents = sorted(assessment_results, key=lambda x: x['total_score'], reverse=True)
        top_patents = sorted_patents[:min(3, len(sorted_patents))]
        
        # 상위 특허 정보 문자열 생성
        top_patents_str = '\n'.join([
            f"- {p['patent_id']} ({p['title']}): 점수 {p['total_score']}/5.0, 등급 {p['grade']}"
            for p in top_patents
        ])
        
        # 종합 등급 결정
        overall_grade = self._determine_grade(avg_total_score)
        
        prompt = f"""당신은 특허 포트폴리오 평가 전문가로서, 다음 정보를 바탕으로 {persona}의 관점에서 종합 평가 보고서를 생성해야 합니다:

평가 개요:
- 평가된 특허 수: {len(assessment_results)}
- 평균 종합 점수: {avg_total_score:.2f}/5.0
- 전체 포트폴리오 등급: {overall_grade}

상위 특허:
{top_patents_str}

페르소나: {persona}

위 정보를 바탕으로, 특허 포트폴리오 전체에 대한 종합적인 평가와 {persona}의 관점에서의 전략적 권고사항을 작성해주세요.
종합 평가는 다음 항목을 포함해야 합니다:
1. 전체 포트폴리오의 기술-시장 적합성 분석
2. 강점과 약점
3. 전략적 권고사항
4. 중점적으로 개발/투자해야 할 영역

한글로 작성하되, 간결하면서도 인사이트가 있는 보고서를 작성해주세요.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            
            overall_assessment = response.choices[0].message.content.strip()
            return overall_assessment
            
        except Exception as e:
            logger.error(f"[FitAssessmentAgent] Error generating overall assessment: {e}")
            
            # 기본 종합 평가 생성
            basic_assessment = f"""
# 특허 포트폴리오 종합 평가 보고서

## 평가 개요
- 평가된 특허 수: {len(assessment_results)}
- 평균 종합 점수: {avg_total_score:.2f}/5.0
- 전체 포트폴리오 등급: {overall_grade}

## 상위 특허
{top_patents_str}

## 종합 평가
평가된 특허 포트폴리오는 전반적으로 {overall_grade}급 수준으로 평가됩니다.

## {persona}를 위한 전략적 권고사항
"""
            
            if overall_grade == "상":
                basic_assessment += f"이 포트폴리오는 높은 가치를 가지고 있으므로, {persona}의 관점에서 적극적인 투자/활용이 권장됩니다."
            elif overall_grade == "중":
                basic_assessment += f"이 포트폴리오는 중간 수준의 가치를 가지고 있으므로, {persona}의 관점에서 선별적 투자/활용이 권장됩니다."
            else:
                basic_assessment += f"이 포트폴리오는 개선이 필요하므로, {persona}의 관점에서 추가 개발 후 재평가가 권장됩니다."
                
            return basic_assessment
    
    def _save_results_to_json(self, assessment_results: List[Dict[str, Any]]) -> None:
        """평가 결과를 JSON 파일로 저장합니다."""
        try:
            output_path = os.path.join(self.output_dir, "final_results.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "patents": assessment_results
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"[FitAssessmentAgent] Saved assessment results to {output_path}")
        except Exception as e:
            logger.error(f"[FitAssessmentAgent] Error saving assessment results: {e}") 