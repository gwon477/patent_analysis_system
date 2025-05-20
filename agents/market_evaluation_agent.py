import os
import logging
import yaml
from typing import List, Dict, Any, Optional
import openai
from utils.jinja2_loader import load_jinja2_template
from utils.market_search import expand_keyword, normalize_scores, extract_market_size, extract_growth_rate
from utils.market_react import MarketReactAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

class MarketEvaluationAgent:
    """특허의 시장 가치를 평가하는 에이전트"""
    
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
            logger.warning(f"[MarketEvaluationAgent] OpenAI 클라이언트 초기화 오류: {e}")
            # 기본 매개변수만 사용
            self.client = openai.Client(api_key=llm_api_key)
        
        # 프롬프트 템플릿 로드
        self.market_prompt_template = load_jinja2_template(prompt_dir, "market_eval_prompt.j2")
        self.react_prompt_template = load_jinja2_template(prompt_dir, "market_react_prompt.j2")
        
        # 설정 로드
        self.config = self._load_config(config_dir)
        
        # Tavily API 키 확인
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("[MarketEvaluationAgent] TAVILY_API_KEY not found in .env file.")
            self.use_tavily = False
        else:
            self.use_tavily = self.config.get("api", {}).get("use_tavily", True)
            logger.info(f"[MarketEvaluationAgent] Tavily API 검색 {self.use_tavily}")

    def _load_config(self, config_dir: str) -> Dict[str, Any]:
        """설정 파일을 로드합니다."""
        config_path = os.path.join(config_dir, "market_eval_config.yaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"[MarketEvaluationAgent] Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"[MarketEvaluationAgent] Config file not found at {config_path}. Using defaults.")
            # 기본 설정
            return {
                "score_ranges": {
                    "market_size": {"min_value": 1, "max_value": 500},
                    "growth_rate": {"min_value": 3.0, "max_value": 30.0}
                },
                "react_agent": {"max_iterations": 3, "keyword_expansions": 3},
                "cache": {"enabled": True, "dir": "./data/market_cache"},
                "api": {"use_tavily": True, "search_max_results": 5}
            }

    def evaluate(self, patent_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        특허 리스트에 대한 시장 평가를 수행합니다.
        
        Args:
            patent_list: 평가할 특허 리스트
            
        Returns:
            Dict: market_eval_results 키를 포함하는 딕셔너리
        """
        logger.info(f"[MarketEvaluationAgent] Evaluating {len(patent_list)} patents")
        
        results = []
        
        for patent in patent_list:
            patent_id = patent.get("patent_id", "Unknown")
            patent_title = patent.get("title", "")
            
            logger.info(f"[MarketEvaluationAgent] Processing patent {patent_id}: {patent_title}")
            
            if not patent_title:
                logger.warning(f"[MarketEvaluationAgent] No title found for patent {patent_id}. Skipping market search.")
                market_info_str = "시장 정보를 조회할 수 없습니다 (특허 제목 누락)."
                market_size = None
                growth_rate = None
                market_size_score = 1.0
                growth_potential_score = 1.0
                evaluation = "특허 제목 정보가 없어 시장 분석을 수행할 수 없습니다."
                expanded_keywords = []
            else:
                # 1. 키워드 확장
                expanded_keywords = self._expand_patent_keywords(patent_title)
                logger.info(f"[MarketEvaluationAgent] Expanded keywords for {patent_id}: {expanded_keywords}")
                
                # 2. ReAct 에이전트로 시장 정보 검색 및 분석
                market_data = self._analyze_market_with_react(patent_title, expanded_keywords)
                
                # 3. 정보 추출 및 점수 계산
                market_size = market_data.get("market_size")
                growth_rate = market_data.get("growth_rate")
                market_info_str = market_data.get("summary", "시장 정보를 찾을 수 없습니다.")
                
                # 4. 점수 계산
                market_size_score = self._calculate_market_size_score(market_size)
                growth_potential_score = self._calculate_growth_potential_score(growth_rate)
                
                # 5. 종합 평가 생성
                evaluation = self._generate_market_evaluation(
                    patent_title, 
                    market_size, 
                    growth_rate, 
                    market_size_score, 
                    growth_potential_score
                )
            
            # 6. 결과 템플릿 렌더링
            market_summary_str = self.market_prompt_template.render(
                patent_id=patent_id,
                title=patent_title,
                market_info=market_info_str,
                market_size=market_size,
                growth_rate=growth_rate,
                market_size_score=market_size_score,
                growth_potential_score=growth_potential_score,
                evaluation=evaluation
            )
            
            # 7. 결과 추가
            results.append({
                "patent_id": patent_id,
                "market_size": market_size,
                "growth_rate": growth_rate,
                "market_size_score": market_size_score,
                "growth_potential_score": growth_potential_score,
                "market_summary": market_summary_str,
                "evaluation": evaluation,
                "expanded_keywords": expanded_keywords
            })
        
        # 8. 전체 평균 점수 계산
        avg_market_size_score = sum(r["market_size_score"] for r in results) / len(results) if results else 0
        avg_growth_potential_score = sum(r["growth_potential_score"] for r in results) / len(results) if results else 0
        
        logger.info(f"[MarketEvaluationAgent] Evaluation complete. Average scores - Market Size: {avg_market_size_score:.2f}, Growth Potential: {avg_growth_potential_score:.2f}")
        
        return {
            "market_eval_results": {
                "patents": results,
                "avg_market_size_score": round(avg_market_size_score, 2),
                "avg_growth_potential_score": round(avg_growth_potential_score, 2)
            }
        }
    
    def _expand_patent_keywords(self, patent_title: str) -> List[str]:
        """특허 제목에서 키워드를 확장합니다."""
        num_expansions = self.config.get("react_agent", {}).get("keyword_expansions", 3)
        try:
            logger.info(f"[MarketEvaluationAgent] Expanding keywords for: {patent_title}")
            expanded = expand_keyword(patent_title, self.api_key, num_expansions)
            logger.info(f"[MarketEvaluationAgent] Expanded to: {expanded}")
            return expanded
        except Exception as e:
            logger.error(f"[MarketEvaluationAgent] Error expanding keywords: {e}")
            # 오류 발생 시 원본 키워드만 반환
            return [patent_title]
    
    def _analyze_market_with_react(self, patent_title: str, expanded_keywords: List[str]) -> Dict[str, Any]:
        """ReAct 에이전트를 사용하여 시장 정보를 분석합니다."""
        if not self.use_tavily or not self.tavily_api_key:
            logger.warning("[MarketEvaluationAgent] Tavily API not available. Using simulated market data.")
            return {
                "market_size": 120,
                "growth_rate": 22.5,
                "summary": f"- 관련 시장: {patent_title} 관련 기술 시장\n- 시장 규모: 120 억 달러\n- 성장률(CAGR): 22.5%\n- 분석 요약: 이 기술 분야는 높은 성장률과 상당한 시장 규모를 보이고 있습니다."
            }
        
        # 키워드 컨텍스트 생성 (확장 키워드가 있을 경우)
        if len(expanded_keywords) > 1:
            keyword_context = ", ".join(expanded_keywords[1:])  # 첫 번째 키워드(원본)를 제외한 나머지
        else:
            keyword_context = ""
        
        # ReAct 에이전트 생성 및 실행
        max_iterations = self.config.get("react_agent", {}).get("max_iterations", 3)
        try:
            agent = MarketReactAgent(
                openai_api_key=self.api_key,
                tavily_api_key=self.tavily_api_key,
                max_iterations=max_iterations
            )
            
            # 에이전트 실행
            logger.info(f"[MarketEvaluationAgent] Running ReAct agent for: {patent_title}")
            result = agent.run(patent_title, keyword_context)
            
            # 결과 로깅
            logger.info(f"[MarketEvaluationAgent] ReAct result - Market Size: {result.get('market_size')}, Growth Rate: {result.get('growth_rate')}")
            
            return result
        except Exception as e:
            logger.error(f"[MarketEvaluationAgent] Error with ReAct agent: {e}")
            # 오류 발생 시 기본 데이터 반환
            return {
                "market_size": 100,
                "growth_rate": 15.0,
                "summary": f"- 관련 시장: {patent_title} 관련 기술 시장\n- 시장 규모: 100 억 달러 (추정)\n- 성장률(CAGR): 15.0% (추정)\n- 분석 요약: 시장 데이터 분석 중 오류가 발생했습니다. 추정값이 사용되었습니다."
            }
    
    def _calculate_market_size_score(self, market_size: Optional[float]) -> float:
        """시장 규모를 점수로 변환합니다."""
        if market_size is None or market_size <= 0:
            return 1.0  # 최소 점수
        
        # 설정에서 범위 가져오기
        min_value = self.config.get("score_ranges", {}).get("market_size", {}).get("min_value", 1)
        max_value = self.config.get("score_ranges", {}).get("market_size", {}).get("max_value", 500)
        
        # 정규화
        score = normalize_scores(market_size, min_value, max_value)
        logger.info(f"[MarketEvaluationAgent] Market size {market_size} -> Score {score}")
        
        return round(score, 1)
    
    def _calculate_growth_potential_score(self, growth_rate: Optional[float]) -> float:
        """성장률을 점수로 변환합니다."""
        if growth_rate is None or growth_rate <= 0:
            return 1.0  # 최소 점수
        
        # 설정에서 범위 가져오기
        min_value = self.config.get("score_ranges", {}).get("growth_rate", {}).get("min_value", 3.0)
        max_value = self.config.get("score_ranges", {}).get("growth_rate", {}).get("max_value", 30.0)
        
        # 정규화
        score = normalize_scores(growth_rate, min_value, max_value)
        logger.info(f"[MarketEvaluationAgent] Growth rate {growth_rate}% -> Score {score}")
        
        return round(score, 1)
    
    def _generate_market_evaluation(self, patent_title: str, market_size: Optional[float], growth_rate: Optional[float], 
                                   market_size_score: float, growth_potential_score: float) -> str:
        """시장 크기와 성장 잠재력 점수를 기반으로 종합 평가를 생성합니다."""
        try:
            prompt = f"""다음 특허와 시장 정보를 기반으로 1-2문장의 간략한 시장 평가를 생성해주세요:

특허 제목: {patent_title}
시장 규모: {market_size if market_size is not None else '정보 없음'} 억 달러 (점수: {market_size_score}/5.0)
성장 잠재력: {growth_rate if growth_rate is not None else '정보 없음'}% CAGR (점수: {growth_potential_score}/5.0)

평가는 두 점수를 모두 고려하되, 특별한 인사이트를 제공해야 합니다. 5점 척도에서 각 점수의 의미:
1점: 매우 낮음
2점: 낮음
3점: 보통
4점: 높음
5점: 매우 높음

평가는 한글로 작성하고, 각 점수가 의미하는 시장 가치와 잠재력을 명확히 설명해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            evaluation = response.choices[0].message.content.strip()
            logger.info(f"[MarketEvaluationAgent] Generated evaluation: {evaluation[:100]}...")
            return evaluation
            
        except Exception as e:
            logger.error(f"[MarketEvaluationAgent] Error generating evaluation: {e}")
            
            # 기본 평가 생성
            if market_size_score >= 4.0 and growth_potential_score >= 4.0:
                return f"해당 특허 기술의 시장 규모({market_size_score}/5.0)와 성장 잠재력({growth_potential_score}/5.0)이 모두 높아 상당한 시장 가치가 예상됩니다."
            elif market_size_score >= 3.5 or growth_potential_score >= 3.5:
                return f"해당 특허 기술은 {'큰 시장 규모' if market_size_score >= 3.5 else '높은 성장 잠재력'}을 가지고 있어 긍정적인 시장 전망을 보여줍니다."
            else:
                return f"해당 특허 기술의 시장 규모({market_size_score}/5.0)와 성장 잠재력({growth_potential_score}/5.0)이 제한적이나, 틈새 시장에서 가치를 창출할 가능성이 있습니다." 