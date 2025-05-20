# AI 특허 평가 멀티에이전트 시스템

## 개요
본 프로젝트는 인공지능 기술을 활용하여 특허를 자동으로 검색하고 평가하는 멀티에이전트 기반 시스템입니다. 특허를 기술적, 시장적 측면에서 종합적으로 분석하고, 맞춤형 평가 보고서를 생성합니다.

주요 기능:
- 키워드 기반 특허 검색 및 임베딩 데이터베이스 구축
- 특허의 기술성/시장성/적합성 멀티 측면 평가
- 사용자 페르소나별 맞춤형 보고서 생성
- DAG 기반의 병렬 평가 워크플로우 자동화

## 주요 기술 스택
- **LangGraph**: 방향성 비순환 그래프(DAG) 기반 에이전트 오케스트레이션
- **LangChain**: LLM 체인, 검색 및 추론 파이프라인 구축
- **ChromaDB**: 벡터 기반 특허 임베딩 및 시맨틱 검색
- **Jinja2**: 동적 프롬프트 및 보고서 템플릿 생성
- **KIPRIS API**: 한국 특허정보원 특허 데이터 검색 및 조회

## 설치 및 환경 설정

### 요구 사항
- Python 3.9 이상 (3.11 권장)
- 필수 API 키:
  - OPENAI_API_KEY: OpenAI API 키
  - KIPRIS_SERVICE_KEY: 한국 특허정보원 API 키
  - TAVILY_API_KEY (선택): 시장 조사용 웹 검색 API

### 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/gwon477/patent_analysis_system.git
cd patent_analysis_system
```

2. 환경 설정 스크립트 실행
```bash
# Linux/macOS
chmod +x setup_env.sh  # 실행 권한 부여
./setup_env.sh

# Windows (Git Bash 등 사용)
./setup_env.sh
```

이 스크립트는 다음과 같은 작업을 수행합니다:
- Python 가상환경 생성 (`venv` 디렉토리)
- `requirements.txt`에 명시된 모든 패키지 자동 설치
- 가상환경 활성화 방법 안내

3. 가상환경 활성화
```bash
# Linux/macOS
source venv/bin/activate

# Windows
source venv/Scripts/activate  # Git Bash 사용 시
# 또는
venv\Scripts\activate  # CMD 사용 시
```

4. `.env` 파일 생성 및 API 키 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가하세요:
```
OPENAI_API_KEY=your_openai_api_key_here
KIPRIS_SERVICE_KEY=your_kipris_api_key_here
CHROMA_DB_PATH=./data/embeddings/
TAVILY_API_KEY=your_tavily_api_key_here  # 선택사항
```

## 실행 방법

### 기본 실행

1. 가상환경이 활성화된 상태에서 메인 스크립트 실행:
```bash
python main.py
```

2. 커맨드라인 인터페이스(CLI)에서 다음 정보를 입력하세요:
   - 검색할 특허 키워드
   - 검색 연도 범위
   - 평가 페르소나 (investor/strategy 등)
   - 보고서 언어 (ko/en)

3. 실행이 완료되면 `reports/` 디렉토리에 평가 보고서가 생성됩니다.

### 특정 에이전트만 실행 (개발/테스트용)

각 에이전트는 독립적으로 실행할 수 있습니다:
```bash
# 특허 검색 에이전트만 실행
python example/run_patent_search_agent.py

# 기술 평가 에이전트만 실행
python example/run_tech_evaluation_agent.py

# 시장 평가 에이전트만 실행
python example/run_market_evaluation_agent.py

# 적합성 평가 에이전트만 실행
python example/run_fit_assessment_agent.py

# 보고서 생성 에이전트만 실행
python example/run_report_generator_agent.py
```

## 전체 워크플로우 흐름

1. **특허 검색 및 임베딩**
   - 키워드 기반 KIPRIS API 특허 검색
   - 청구항 및 초록 추출 및 임베딩
   - ChromaDB에 벡터 저장 및 인덱싱

2. **기술성 평가** (병렬 처리)
   - 특허 청구항 기반 RAG 검색
   - 기술 혁신성 및 구현 난이도 평가
   - 종합 기술성 점수 산출

3. **시장성 평가** (병렬 처리)
   - 시장 규모 및 성장성 조사
   - 경쟁 상황 및 시장 진입 장벽 분석
   - 종합 시장성 점수 산출

4. **적합성 평가**
   - 기술-시장 연결성 분석
   - 특허 활용 가능성 평가
   - 종합 적합성 등급 산출

5. **보고서 생성**
   - 페르소나별 맞춤형 보고서 템플릿 적용
   - 평가 결과 요약 및 시각화
   - 최종 보고서 파일 생성

## 폴더 구조
```
.
├── agents/                  # 각 평가 단계별 에이전트
│   ├─ patent_search_agent.py
│   ├─ tech_evaluation_agent.py
│   ├─ market_evaluation_agent.py
│   ├─ fit_assessment_agent.py
│   └─ report_generator_agent.py
├── config/                  # 설정 파일 (YAML)
│   ├─ config.yaml           # 기본 설정
│   ├─ weights.yaml          # 평가 가중치
│   └─ persona_profiles.yaml # 페르소나별 설정
├── data/                    # 입력 데이터, 임시 캐시, 임베딩 DB
│   ├── embeddings/          # ChromaDB 저장 경로
│   ├── sample_patents_ko.json # 한국어 샘플 특허
│   ├── sample_patents_en.json # 영어 샘플 특허
│   └── market_info_cache.json # 시장 정보 캐시 (자동 생성)
├── example/                 # 각 에이전트별 독립 실행 예제
├── prompts/                 # LLM 및 보고서용 Jinja2 템플릿
│   ├─ tech_summary_prompt.j2
│   ├─ market_eval_prompt.j2
│   ├─ fit_alignment_prompt.j2
│   ├─ report_template_ko.j2
│   └─ report_template_en.j2
├── reports/                 # 생성된 보고서 저장 (자동 생성)
├── utils/                   # 유틸리티 함수
│   ├─ chunking.py           # 텍스트 청크 분할
│   ├─ websearch.py          # 웹 검색 기능
│   └─ jinja2_loader.py      # 템플릿 로더
├── setup_env.sh             # 환경 설정 스크립트
├── run_patent_workflow.sh   # 전체 워크플로우 실행 스크립트
├── main.py                  # 메인 실행 스크립트 (LangGraph 워크플로우)
├── requirements.txt         # Python 패키지 의존성
└── README.md                # 프로젝트 설명
```

## 커스터마이징 및 확장

### 설정 파일 수정
- `config/config.yaml`: 기본 언어, 임베딩 모델, 기본 페르소나 등 설정
- `config/weights.yaml`: 평가 항목별 가중치 조정
- `config/persona_profiles.yaml`: 새로운 페르소나 및 평가 기준 추가

### 프롬프트 템플릿 수정
- `prompts/` 디렉토리의 Jinja2 템플릿 파일을 수정하여 평가 기준 변경
- 새로운 언어 지원을 위한 템플릿 추가 가능

### 에이전트 확장
- `agents/` 디렉토리에 새로운 평가 에이전트 추가 가능
- `main.py`의 LangGraph DAG에 새 노드 추가

## 주의사항 및 제약

- KIPRIS API는 한국 특허만 검색 가능합니다. 다른 국가 특허는 추가 API 연동이 필요합니다.
- 기본 OpenAI GPT 모델 사용 시 비용이 발생할 수 있습니다.
- 평가 결과는 참고용이며, 실제 특허 가치 평가는 전문가의 검토가 필요합니다.

## 라이센스

이 프로젝트는 MIT 라이센스에 따라 배포됩니다.
