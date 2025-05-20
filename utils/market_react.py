import logging
from typing import List, Dict, Any, Tuple, Optional
import openai
from utils.market_search import MarketSearchTool, extract_market_size, extract_growth_rate
import re

# 로깅 설정
logger = logging.getLogger(__name__)

class MarketReactAgent:
    """
    ReAct 패턴을 사용하여 시장 정보를 분석하는 에이전트입니다.
    Thought -> Action -> Observation 패턴으로 여러 단계의 추론을 수행합니다.
    """
    
    def __init__(self, openai_api_key: str, tavily_api_key: str, max_iterations: int = 3):
        """
        Args:
            openai_api_key: OpenAI API 키
            tavily_api_key: Tavily API 키
            max_iterations: 최대 ReAct 반복 횟수
        """
        self.openai_api_key = openai_api_key
        # 최신 OpenAI 라이브러리 호환성을 위한 수정
        try:
            # proxies 관련 설정을 제거하고 필수 매개변수만 전달
            self.client = openai.OpenAI(api_key=openai_api_key, default_headers=None, max_retries=2)
        except Exception as e:
            logger.warning(f"[MarketReactAgent] OpenAI 클라이언트 초기화 오류: {e}")
            # 기본 매개변수만 사용
            self.client = openai.Client(api_key=openai_api_key)
        
        self.search_tool = MarketSearchTool(api_key=tavily_api_key)
        self.max_iterations = max_iterations
    
    def run(self, query: str, keyword_context: str = "") -> Dict[str, Any]:
        """
        주어진 쿼리에 대해 ReAct 에이전트를 실행합니다.
        
        Args:
            query: 검색 쿼리 (특허 제목)
            keyword_context: 추가 키워드 컨텍스트 (선택적)
            
        Returns:
            에이전트 실행 결과 (시장 규모, 성장률 등 포함)
        """
        # ReAct 프롬프트 초기화
        initial_prompt = self._build_initial_prompt(query, keyword_context)
        current_context = initial_prompt
        
        # 현재 실행 상태 추적
        iteration = 0
        results = {
            "market_size": None,
            "growth_rate": None,
            "summary": "",
            "raw_search_results": []
        }
        
        # ReAct 반복 수행
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[MarketReactAgent] Starting iteration {iteration}/{self.max_iterations}")
            
            # LLM에 현재 컨텍스트 전송하여 다음 단계 결정
            response = self._call_openai(current_context)
            
            # 응답에서 Thought, Action, Action Input 추출
            thought, action, action_input = self._parse_llm_response(response)
            
            # 로깅
            logger.info(f"[MarketReactAgent] Thought: {thought}")
            logger.info(f"[MarketReactAgent] Action: {action}")
            logger.info(f"[MarketReactAgent] Action Input: {action_input}")
            
            # 명시적 Final Answer 또는 최대 반복 횟수 도달 시 종료
            if action.lower() == "final answer":
                results["summary"] = action_input
                # 최종 응답에서 시장 규모와 성장률을 다시 한번 추출 시도
                self._extract_market_info_from_final_answer(action_input, results)
                break
            
            # Action 수행
            observation = self._execute_action(action, action_input)
            logger.info(f"[MarketReactAgent] Observation: {observation[:200]}...")  # 처음 200자만 로깅
            
            # 관찰 결과를 raw_search_results에 추가
            if action.lower().startswith("search"):
                results["raw_search_results"].append({
                    "query": action_input,
                    "type": action.lower(),
                    "observation": observation
                })
            
            # 관찰 결과에서 시장 규모와 성장률 추출 시도
            self._extract_market_info(observation, results)
            
            # 컨텍스트 업데이트
            current_context += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
        
        # 최종 요약 생성 (아직 없는 경우)
        if not results["summary"]:
            results["summary"] = self._generate_final_summary(results, current_context)
        
        return results
    
    def _build_initial_prompt(self, query: str, keyword_context: str) -> str:
        """ReAct 에이전트의 초기 프롬프트를 생성합니다."""
        prompt = f"""시장 조사 전문가로서, '{query}'에 관한 시장 규모와 성장률(CAGR)을 조사하세요.

추가 키워드 정보: {keyword_context}

이 분석 프로세스는 다음 단계로 진행하세요:
1. 먼저 해당 기술/제품이 속한 시장을 정확히 식별하세요 (예: "스마트 홈 자동화 시장", "전기차 배터리 시장" 등).
2. 해당 시장의 규모(억 달러 단위)를 찾으세요. 시장 규모는 주로 "X billion dollars" 또는 "X billion USD" 형식으로 표현됩니다.
3. 해당 시장의 성장률(CAGR %)을 찾으세요. 성장률은 주로 "X% CAGR" 또는 "CAGR of X%" 형식으로 표현됩니다.
4. 발견한 정보를 종합하여 요약하세요.

각 단계에서 다음 형식으로 진행하세요:
Thought: 무엇을 해야 하는지 생각합니다. 이전 검색에서 충분한 정보를 얻었는지 판단하고, 다음 단계를 결정합니다.
Action: 다음 중 하나를 선택합니다 [Search Market Size, Search Growth Rate, Final Answer]
Action Input: 액션에 필요한 입력(검색어)을 제공합니다.
Observation: 당신의 Action 결과가 여기에 표시됩니다.

Final Answer를 선택하면, 발견한 모든 정보를 다음 형식으로 요약해주세요:
- 관련 시장: [식별된 시장]
- 시장 규모: [X 억 달러]
- 성장률(CAGR): [Y%]
- 분석 요약: [간략한 분석]

시장 규모(억 달러)와 성장률(CAGR %)은 숫자 값으로 명확하게 표현하세요. 정확한 값을 찾지 못한 경우 가장 신뢰할 수 있는 추정치를 제시하세요.

시작하세요:
"""
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """OpenAI API를 호출하여 다음 단계를 결정합니다."""
        try:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except AttributeError:
                # 이전 버전 호환
                response = self.client.chat.completions.create(
                    model="gpt-4",  # gpt-4o가 지원되지 않으면 gpt-4 사용
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[MarketReactAgent] Error calling OpenAI: {e}")
            return "Thought: API 호출 오류가 발생했습니다.\nAction: Final Answer\nAction Input: 시장 분석을 완료할 수 없습니다."
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str, str]:
        """LLM 응답에서 Thought, Action, Action Input을 추출합니다."""
        thought = ""
        action = ""
        action_input = ""
        
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Thought:"):
                    thought = line[line.find(":")+1:].strip()
                elif line.startswith("Action:"):
                    action = line[line.find(":")+1:].strip()
                elif line.startswith("Action Input:"):
                    action_input = line[line.find(":")+1:].strip()
            
            # 정보가 부족한 경우 기본값 설정
            if not thought:
                thought = "정보가 부족하여 다음 단계를 결정합니다."
            if not action:
                action = "Final Answer"
            if not action_input and action == "Final Answer":
                action_input = "충분한 정보를 찾을 수 없습니다."
                
            return thought, action, action_input
        except Exception as e:
            logger.error(f"[MarketReactAgent] Error parsing LLM response: {e}")
            return "파싱 오류 발생", "Final Answer", "응답을 처리하는 중 오류가 발생했습니다."
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """지정된 액션을 수행합니다."""
        try:
            if action.lower() == "search market size":
                results = self.search_tool.search(action_input, search_type="market_size")
                return self._format_search_results(results)
            elif action.lower() == "search growth rate":
                results = self.search_tool.search(action_input, search_type="growth_rate")
                return self._format_search_results(results)
            elif action.lower() == "final answer":
                return ""
            else:
                return f"알 수 없는 액션입니다: {action}. 지원되는 액션은 'Search Market Size', 'Search Growth Rate', 'Final Answer'입니다."
        except Exception as e:
            logger.error(f"[MarketReactAgent] Error executing action: {e}")
            return f"액션 실행 중 오류가 발생했습니다: {str(e)}"
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """검색 결과를 읽기 쉬운 형식으로 포맷팅합니다."""
        if not results:
            return "검색 결과가 없습니다."
        
        formatted = "검색 결과:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "제목 없음")
            content = result.get("content", "내용 없음")
            url = result.get("url", "URL 없음")
            
            snippet = f"{i}. {title}\n{content}\n출처: {url}\n\n"
            formatted += snippet
        
        return formatted
    
    def _extract_market_info(self, text: str, results: Dict[str, Any]) -> None:
        """텍스트에서 시장 규모와 성장률 정보를 추출하여 결과 딕셔너리를 업데이트합니다."""
        # 아직 시장 규모가 추출되지 않았다면 추출 시도
        if results["market_size"] is None:
            market_size = extract_market_size(text)
            if market_size > 0:
                results["market_size"] = market_size
                logger.info(f"[MarketReactAgent] Extracted market size: {market_size} billion dollars")
        
        # 아직 성장률이 추출되지 않았다면 추출 시도
        if results["growth_rate"] is None:
            growth_rate = extract_growth_rate(text)
            if growth_rate > 0:
                results["growth_rate"] = growth_rate
                logger.info(f"[MarketReactAgent] Extracted growth rate: {growth_rate}%")
    
    def _extract_market_info_from_final_answer(self, text: str, results: Dict[str, Any]) -> None:
        """최종 답변에서 시장 규모와 성장률 정보를 추출합니다."""
        # 최종 요약에서 시장 규모 추출 시도
        if results["market_size"] is None:
            # 시장 규모 패턴
            market_size_patterns = [
                r'시장\s*규모\s*:\s*(\d+\.?\d*)\s*억\s*달러',  # 시장 규모: 10 억 달러
                r'시장\s*규모\s*:\s*(\d+\.?\d*)',             # 시장 규모: 10
                r'Market\s*Size\s*:\s*(\d+\.?\d*)\s*billion', # Market Size: 10 billion
            ]
            
            for pattern in market_size_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        market_size = float(matches[0])
                        results["market_size"] = market_size
                        logger.info(f"[MarketReactAgent] Extracted market size from final answer: {market_size} billion dollars")
                        break
                    except ValueError:
                        continue
        
        # 최종 요약에서 성장률 추출 시도
        if results["growth_rate"] is None:
            # 성장률 패턴
            growth_rate_patterns = [
                r'성장률\s*\(CAGR\)\s*:\s*(\d+\.?\d*)\s*%',   # 성장률(CAGR): 10%
                r'성장률\s*:\s*(\d+\.?\d*)\s*%',              # 성장률: 10%
                r'CAGR\s*:\s*(\d+\.?\d*)\s*%',                # CAGR: 10%
            ]
            
            for pattern in growth_rate_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        growth_rate = float(matches[0])
                        results["growth_rate"] = growth_rate
                        logger.info(f"[MarketReactAgent] Extracted growth rate from final answer: {growth_rate}%")
                        break
                    except ValueError:
                        continue
    
    def _generate_final_summary(self, results: Dict[str, Any], context: str) -> str:
        """현재까지 수집된 정보를 바탕으로 최종 요약을 생성합니다."""
        market_size = results.get("market_size")
        growth_rate = results.get("growth_rate")
        
        summary_prompt = f"""다음 정보를 바탕으로 시장 분석 요약을 생성해주세요:

시장 규모: {market_size if market_size is not None else '정보 없음'} 억 달러
성장률(CAGR): {growth_rate if growth_rate is not None else '정보 없음'}%

전체 컨텍스트:
{context}

요약은 다음 형식으로 제공해주세요:
- 관련 시장: [식별된 시장]
- 시장 규모: [X 억 달러]
- 성장률(CAGR): [Y%]
- 분석 요약: [간략한 분석 (1-2문장)]

정확한 숫자 값(시장 규모와 성장률)을 반드시 포함하세요. 추정치라도 최대한 정확하게 제시해야 합니다.
"""
        
        try:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except AttributeError:
                # 이전 버전 호환
                response = self.client.chat.completions.create(
                    model="gpt-4",  # gpt-4o가 지원되지 않으면 gpt-4 사용
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[MarketReactAgent] Error generating summary: {e}")
            
            # API 오류 발생 시 간단한 요약 직접 생성
            market_size_str = f"{market_size} 억 달러" if market_size is not None else "정보 없음"
            growth_rate_str = f"{growth_rate}%" if growth_rate is not None else "정보 없음"
            
            return f"""- 관련 시장: 검색된 기술 시장
- 시장 규모: {market_size_str}
- 성장률(CAGR): {growth_rate_str}
- 분석 요약: API 오류로 인해 상세 분석을 제공할 수 없습니다. 시장 규모는 {market_size_str}이며, 성장률은 {growth_rate_str}입니다.""" 