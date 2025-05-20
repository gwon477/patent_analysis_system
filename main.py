import os
import argparse
import yaml
from dotenv import load_dotenv
from typing import TypedDict, List, Tuple, Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from agents.patent_search_agent import PatentSearchAgent
from agents.tech_evaluation_agent import TechEvaluationAgent
from agents.market_evaluation_agent import MarketEvaluationAgent
from agents.fit_assessment_agent import FitAssessmentAgent
from agents.report_generator_agent import ReportGeneratorAgent

# .env 파일에서 환경 변수 로드
load_dotenv()

# 시스템 소개 메시지
SYSTEM_INTRO = """
====================================================
    특허 분석 및 평가 시스템 (Patent Analysis System)
====================================================

이 서비스는 Multi-Agents 기반의 특허 분석 및 평가 시스템입니다.
특정 기술에 대한 최신 특허 자료를 검색하고 분석하여 각 상위 특허들에 대한 
기술성, 독창성, 시장성, 트렌드 적합성 등을 평가하는 종합 분석 서비스입니다.

주요 기능:
1. 키워드 기반 특허 검색 (Patent Search Agent)
2. 기술성 평가 (Tech Evaluation Agent - 독창성, 트렌드 적합성)
3. 시장성 평가 (Market Evaluation Agent - 시장 규모, 성장 잠재력)
4. 특허별 적합성 종합 평가 (Fit Assessment Agent)
5. 상위 특허에 대한 상세 분석 보고서 생성 (Report Generator Agent)

입력하신 키워드를 기반으로 관련 특허를 검색하고 분석하여
투자자, 전략가, R&D 담당자 등 다양한 페르소나에 맞는 맞춤형 보고서를 생성합니다.
"""

# LangGraph 상태 정의
class AgentState(TypedDict):
    keywords: List[str]
    year_range: Tuple[int, int]
    persona: str
    llm_api_key: str
    patent_list: Optional[List[Dict[str, Any]]] # PatentSearchAgent 결과
    retriever: Optional[Any] # PatentSearchAgent 결과 (Langchain Retriever)
    tech_eval_results: Optional[Dict[str, Any]] # TechEvaluationAgent 결과
    market_eval_results: Optional[Dict[str, Any]] # MarketEvaluationAgent 결과
    fit_eval_results: Optional[Dict[str, Any]] # FitAssessmentAgent 결과
    report_path: Optional[str] # ReportGeneratorAgent 결과
    config_dir: str # 설정 파일 디렉토리
    prompt_dir: str # 프롬프트 파일 디렉토리
    db_path: str # ChromaDB 경로
    output_dir: str # 보고서 출력 디렉토리
    max_patents_per_keyword: int # 키워드당 최대 특허 수
    embedding_model: str # 임베딩 모델명

# 에이전트 노드 함수 정의
def patent_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n--- Patent Search --- ")
    agent = PatentSearchAgent(db_path=state["db_path"], embedding_model_name=state["embedding_model"])
    result = agent.search_and_store(
        keywords=state["keywords"],
        year_range=state["year_range"],
        max_patents_per_keyword=state["max_patents_per_keyword"]
    )
    return {"patent_list": result["patent_list"], "retriever": result["retriever"]}

def tech_evaluation_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n--- Technical Evaluation --- ")
    if not state["patent_list"] or state["retriever"] is None:
        print("Skipping technical evaluation due to no patents or retriever.")
        return {"tech_eval_results": None}
    agent = TechEvaluationAgent(retriever=state["retriever"], llm_api_key=state["llm_api_key"], prompt_dir=state["prompt_dir"])
    result = agent.evaluate(patent_list=state["patent_list"], keywords=state["keywords"])
    return {"tech_eval_results": result["tech_eval_results"]}

def market_evaluation_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n--- Market Evaluation --- ")
    if not state["patent_list"]:
        print("Skipping market evaluation due to no patents.")
        return {"market_eval_results": None}
    agent = MarketEvaluationAgent(llm_api_key=state["llm_api_key"], prompt_dir=state["prompt_dir"])
    result = agent.evaluate(patent_list=state["patent_list"])
    return {"market_eval_results": result["market_eval_results"]}

def fit_assessment_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n--- Fit Assessment --- ")
    tech_eval = state.get("tech_eval_results")
    market_eval = state.get("market_eval_results")
    if tech_eval is None or market_eval is None:
        print("Skipping fit assessment due to missing evaluation results.")
        return {"fit_eval_results": None}
    agent = FitAssessmentAgent(llm_api_key=state["llm_api_key"], prompt_dir=state["prompt_dir"], config_dir=state["config_dir"])
    result = agent.assess(
        tech_eval_results=tech_eval,
        market_eval_results=market_eval,
        persona=state["persona"]
    )
    return {"fit_eval_results": result["fit_eval_results"]}

def report_generation_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n--- Report Generation --- ")
    if state.get("fit_eval_results") is None:
        print("Skipping report generation due to no fit assessment results.")
        return {"report_path": None}
    agent = ReportGeneratorAgent(prompt_dir=state["prompt_dir"], config_dir=state["config_dir"], output_dir=state["output_dir"])
    result = agent.generate(
        fit_eval_results=state["fit_eval_results"],
        tech_eval_results=state["tech_eval_results"],
        market_eval_results=state["market_eval_results"],
        persona=state["persona"]
    )
    # 보고서 경로 명확하게 반환
    report_path = result.get("report_path")
    print(f"Report generated at: {report_path}")
    return {"report_path": report_path}

# 병렬 평가 분기점 - 현재 langgraph 버전에 맞게 수정
def patent_search_router(state: AgentState) -> List[str]:
    """특허 검색 후 기술 평가와 시장 평가를 병렬로 실행"""
    if not state.get("patent_list"):
        # 특허 검색 결과가 없으면 워크플로우 종료
        return ["END"]
    # 특허 검색 결과가 있으면 두 평가를 병렬로 실행
    return ["tech_evaluation", "market_evaluation"]

# 병렬 평가 결과 확인 및 진행 분기점
def evaluation_join_router(state: AgentState) -> str:
    """기술 평가와 시장 평가 결과를 확인하고 다음 단계 결정"""
    tech_results = state.get("tech_eval_results")
    market_results = state.get("market_eval_results")
    
    # 두 평가 결과가 모두 있으면 적합성 평가로 진행
    if tech_results is not None and market_results is not None:
        return "fit_assessment"
    
    # 결과가 없으면 워크플로우 종료
    if tech_results is None and market_results is None:
        return "END"
        
    # 하나의 결과만 있는 경우는 워크플로우 설계상 발생하지 않지만,
    # 이론적으로 가능성이 있으므로 END로 처리
    return "END"

def create_workflow() -> StateGraph:
    """워크플로우 그래프 생성 함수"""
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("patent_search", patent_search_node)
    workflow.add_node("tech_evaluation", tech_evaluation_node)
    workflow.add_node("market_evaluation", market_evaluation_node)
    workflow.add_node("fit_assessment", fit_assessment_node)
    workflow.add_node("report_generation", report_generation_node)
    
    # 엣지 연결: 병렬 처리 구현
    workflow.set_entry_point("patent_search")
    
    # 특허 검색 후 tech_evaluation과 market_evaluation으로 병렬 분기
    workflow.add_conditional_edges(
        "patent_search",
        patent_search_router,
        {
            "tech_evaluation": "tech_evaluation",
            "market_evaluation": "market_evaluation",
            "END": END
        }
    )
    
    # tech_evaluation과 market_evaluation이 모두 완료되면 fit_assessment로 진행
    # 완료된 노드에서 다음 노드로 이동
    workflow.add_edge("tech_evaluation", "fit_assessment")
    workflow.add_edge("market_evaluation", "fit_assessment")
    
    # fit_assessment 노드에서 report_generation 노드로 이동
    workflow.add_edge("fit_assessment", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow

# 초기 상태 설정 함수
def get_initial_state(keywords=None, year_start=None, year_end=None, persona=None, config_file="config/config.yaml"):
    # 사용자 입력이 없는 경우 직접 입력 받기
    if keywords is None:
        keyword_input = input("검색할 도메인의 키워드를 입력하세요(예: 'RAG', 'AI', 'Edge AI', 다수 키워드 입력도 가능합니다.): ").strip()
        keywords = keyword_input.split()
    
    if year_start is None:
        year_start = int(input("검색 시작 연도를 입력하세요(0은 전체): ") or "0")
    
    if year_end is None:
        year_end = int(input("검색 종료 연도를 입력하세요(0은 현재): ") or "0")
    
    if persona is None:
        persona = input("페르소나를 선택하세요(investor/strategy/r&d, 기본값 investor): ").strip() or "investor"
    
    # 기본 설정 로드
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 환경 변수에서 API 키 로드
    openai_api_key = os.getenv("OPENAI_API_KEY")
    kipris_service_key = os.getenv("KIPRIS_SERVICE_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    if not kipris_service_key:
        print("Warning: KIPRIS_SERVICE_KEY not found in environment variables. Patent search may fail.")

    return AgentState(
        keywords=keywords,
        year_range=(year_start, year_end),
        persona=persona,
        llm_api_key=openai_api_key,
        patent_list=None,
        retriever=None,
        tech_eval_results=None,
        market_eval_results=None,
        fit_eval_results=None,
        report_path=None,
        config_dir=os.path.dirname(config_file) if os.path.dirname(config_file) else ".",
        prompt_dir="./prompts",
        db_path=os.getenv("CHROMA_DB_PATH", "./data/embeddings/"),
        output_dir="./reports",
        max_patents_per_keyword=5,
        embedding_model=config_data.get("embedding_model_ko", "jhgan/ko-sroberta-multitask")
    )

def main():
    # 시스템 소개 출력
    print(SYSTEM_INTRO)
    
    parser = argparse.ArgumentParser(description="AI Patent Evaluation System")
    parser.add_argument("-k", "--keywords", nargs='+', help="Keywords for patent search (e.g., AI medical)")
    parser.add_argument("-ys", "--year_start", type=int, help="Start year for patent search (e.g., 2020)")
    parser.add_argument("-ye", "--year_end", type=int, help="End year for patent search (e.g., 2023)")
    parser.add_argument("-p", "--persona", type=str, help="Persona for evaluation (e.g., investor, strategy)")
    parser.add_argument("-c", "--config", type=str, default="config/config.yaml", help="Path to the main configuration YAML file")

    args = parser.parse_args()

    # 초기 상태 설정
    initial_state = get_initial_state(
        keywords=args.keywords,
        year_start=args.year_start, 
        year_end=args.year_end,
        persona=args.persona,
        config_file=args.config
    )

    print(f"\n분석 시작:")
    print(f"- 키워드: {initial_state['keywords']}")
    print(f"- 검색 연도 범위: {initial_state['year_range']}")
    print(f"- 페르소나: {initial_state['persona']}")
    print("\n워크플로우를 초기화하고 실행합니다...")

    # 워크플로우 생성 및 실행
    workflow = create_workflow()
    app = workflow.compile()

    # LangGraph 실행
    final_state = {}
    report_path = None
    
    try:
        # 워크플로우 실행 및 상태 추적
        for event in app.stream(initial_state, config=RunnableConfig()):
            if isinstance(event, dict):
                # 이벤트 출력 및 상태 업데이트
                for key, value in event.items():
                    if key != "__start__" and key != "__end__":
                        print(f"Node completed: {key}")
                        if key == "report_generation" and value and "report_path" in value:
                            report_path = value["report_path"]
                # 상태 업데이트
                final_state.update(event)
    except Exception as e:
        print(f"Error during workflow execution: {e}")

    # 최종 결과 확인 - 보고서 경로는 report_generation 노드에서 직접 추출
    if report_path:
        print(f"\n✅ 워크플로우 완료! 보고서가 생성되었습니다: {report_path}")
    else:
        print("\n❌ 워크플로우는 완료되었지만 보고서가 생성되지 않았습니다.")

if __name__ == "__main__":
    main() 