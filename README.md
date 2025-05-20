# AI 특허 평가 멀티에이전트 시스템

## 개요
- LangGraph DAG 기반 멀티에이전트 오케스트레이션
- LangChain + ChromaDB + Jinja2 템플릿 기반 자동화
- CLI에서 입력받아 전체 RAG 특허 평가 파이프라인 수행

## 주요 기술
- LangGraph: DAG 정의, 병렬 분기
- LangChain: LLMChain, Retriever, PromptTemplate
- ChromaDB: 임베딩 기반 특허 검색/유사도
- Jinja2: 프롬프트/보고서 구조화

## 실행 방법
1. `.env`에 API 키 (`OPENAI_API_KEY`, `KIPRIS_SERVICE_KEY`, `TAVILY_API_KEY`) 등 환경설정
2. `requirements.txt` 패키지 설치
3. CLI에서 `python main.py` 실행

## 전체 흐름
1. 특허 검색/임베딩(Chroma 저장)
2. 기술성/시장성 평가(병렬, RAG/LLM 기반)
3. 적합성 평가(LLM+템플릿)
4. 맞춤형 보고서 생성/출력

## 폴더 구조
```
.
├── agents/                  # 각 평가 단계별 에이전트
├── config/                  # 설정 파일 (YAML)
├── data/                    # 입력 데이터 (특허 샘플), 임시 캐시, 임베딩 DB
│   ├── embeddings/          # ChromaDB 저장 경로
│   └── market_info_cache.json # 시장 정보 캐시 (자동 생성)
├── prompts/                 # LLM 및 보고서용 Jinja2 템플릿
├── reports/                 # 생성된 보고서 저장 (자동 생성)
├── utils/                   # 유틸리티 함수
├── .env                     # API 키 등 환경 변수 (직접 생성 필요)
├── main.py                  # 메인 실행 스크립트 (LangGraph 워크플로우)
├── requirements.txt         # Python 패키지 의존성
└── README.md                # 프로젝트 설명
```

## 확장/유지보수
- prompts/, config/ 하위 YAML/템플릿으로 각 단계 맞춤
- agents/ 내 각 Agent 독립 개발/테스트 가능
