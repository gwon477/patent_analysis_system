# 최종 프로젝트 디렉토리/파일 구조

project_root/
│
├─ agents/
│    ├─ patent_search_agent.py
│    ├─ tech_evaluation_agent.py
│    ├─ market_evaluation_agent.py
│    ├─ fit_assessment_agent.py
│    ├─ report_generator_agent.py
│
├─ data/
│    ├─ sample_patents_ko.json
│    ├─ sample_patents_en.json
│    └─ embeddings/            # Chroma VectorDB 실제 저장 경로
│
├─ prompts/
│    ├─ tech_summary_prompt.j2
│    ├─ market_eval_prompt.j2
│    ├─ fit_alignment_prompt.j2
│    ├─ report_template_ko.j2
│    └─ report_template_en.j2
│
├─ config/
│    ├─ config.yaml
│    ├─ weights.yaml
│    ├─ persona_profiles.yaml
│
├─ utils/
│    ├─ chunking.py
│    ├─ websearch.py
│    ├─ jinja2_loader.py
│
├─ main.py
├─ requirements.txt
├─ .env
├─ README.md

