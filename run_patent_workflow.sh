#!/bin/bash

# 현재 디렉토리 확인
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# 키워드 설정 (기본값: RAG AI)
KEYWORDS=${1:-"RAG AI"}
echo "Using keywords: $KEYWORDS"

# 페르소나 설정 (기본값: investor)
PERSONA=${2:-"investor"}
echo "Using persona: $PERSONA"

# 특허 검색 단계
echo -e "\n\n===== 1. 특허 검색 =====\n"
python example/run_patent_search_agent.py -k $KEYWORDS

# 기술 평가 단계
echo -e "\n\n===== 2. 기술 평가 =====\n"
python example/run_tech_evaluation_agent.py -i example/patent_search_output.json -k $KEYWORDS

# 시장 평가 단계
echo -e "\n\n===== 3. 시장 평가 =====\n"
python example/run_market_evaluation_agent.py -i example/patent_search_output.json

# 적합성 평가 단계
echo -e "\n\n===== 4. 적합성 평가 =====\n"
python example/run_fit_assessment_agent.py -t example/tech_evaluation_output.json -m example/market_evaluation_output.json -p $PERSONA

# 보고서 생성 단계
echo -e "\n\n===== 5. 보고서 생성 =====\n"
python example/run_report_generator_agent.py -f example/fit_assessment_output.json -t example/tech_evaluation_output.json -m example/market_evaluation_output.json -p $PERSONA -o_dir ./reports

echo -e "\n\n===== 워크플로우 완료 =====\n"
echo "보고서가 ./reports 디렉토리에 생성되었습니다." 