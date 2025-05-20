from langchain.prompts import PromptTemplate
from utils.jinja2_loader import load_jinja2_template
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import logging
import os
from langchain_community.tools.tavily_search import TavilySearchResults
import openai  # openai 패키지 직접 사용
import requests  # HTTP 요청용
import json  # JSON 처리용
from sklearn.metrics.pairwise import cosine_similarity

# 로깅 설정
logger = logging.getLogger(__name__)

class TechEvaluationAgent:
    def __init__(self, retriever, llm_api_key, prompt_dir="./prompts"):
        # API 키 설정 - OpenAI 클라이언트를 직접 사용하지 않음
        os.environ["OPENAI_API_KEY"] = llm_api_key
        self.api_key = llm_api_key
        self.tech_summary_prompt_template = load_jinja2_template(prompt_dir, "tech_summary_prompt.j2")
        
        # RetrievalQA가 retriever를 필수로 요구하므로, None이 아닌 실제 retriever를 받아야 함.
        if retriever is None:
            raise ValueError("Retriever cannot be None for TechEvaluationAgent")
            
        # retriever만 저장하고 QA chain은 사용하지 않음
        self.retriever = retriever
        
        # Tavily API 키 설정
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.warning("[TechEvaluationAgent] TAVILY_API_KEY not found in .env file. Using simulated search.")
            self.use_real_search = False
        else:
            logger.info("[TechEvaluationAgent] Using Tavily API for web search")
            self.use_real_search = True
            self.search_tool = TavilySearchResults(api_key=tavily_api_key)
    
    def _call_openai(self, prompt, model="gpt-4o", temperature=0.7):
        """OpenAI API 직접 호출 - requests 라이브러리 사용"""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"[TechEvaluationAgent] API error: {response.status_code}, {response.text}")
                return f"API 호출 오류: {response.status_code}"
        except Exception as e:
            logger.error(f"[TechEvaluationAgent] Error calling OpenAI API: {e}")
            return f"API 호출 오류: {str(e)}"
    
    def _normalize_scores_to_five(self, scores):
        """점수 배열을 정규화하여 최고 점수가 5.0이 되도록 조정하고 소수점 첫째 자리까지 반올림"""
        if not scores:
            return []
            
        # 최고 점수가 이미 5.0인지 확인
        max_score = max(scores)
        
        if max_score <= 0:
            # 모든 점수가 0 이하인 경우 기본값 반환
            return [1.0] * len(scores)
            
        # 이미 모든 점수가 1.0-5.0 범위 내인지 확인
        all_in_range = all(1.0 <= score <= 5.0 for score in scores)
        
        normalized_scores = []
        if all_in_range and max_score == 5.0:
            # 이미 정규화되어 있으면 그대로 사용
            normalized_scores = scores
        else:
            # 최고 점수를 5.0으로 정규화하고 비례적으로 조정
            normalization_factor = 5.0 / max_score
            normalized_scores = [max(1.0, min(5.0, score * normalization_factor)) for score in scores]
        
        # 소수점 첫째 자리까지 반올림
        rounded_scores = [round(score, 1) for score in normalized_scores]
        
        return rounded_scores
    
    def _create_search_trend_chain(self):
        """이 함수는 더이상 사용하지 않습니다."""
        pass
        
    def _perform_web_search(self, query):
        """Tavily API를 사용하여 실제 웹 검색 수행"""
        try:
            logger.info(f"[TechEvaluationAgent] Performing Tavily web search for: {query}")
            search_results = self.search_tool.invoke(query)
            logger.info(f"[TechEvaluationAgent] Search completed, found {len(search_results)} results")
            return str(search_results)
        except Exception as e:
            logger.error(f"[TechEvaluationAgent] Error during web search: {e}")
            return f"검색 오류 발생: {str(e)}"
    
    def _extract_trend_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 리스트를 사용하여 기술 트렌드 키워드 추출"""
        query = ", ".join(keywords) + " 기술 트렌드"
        
        try:
            # 웹 검색 수행
            if self.use_real_search:
                search_results = self._perform_web_search(query)
            else:
                logger.info(f"[TechEvaluationAgent] Simulating web search for: {query}")
                search_results = f"'{query}'에 대한 검색 결과: 최신 기술 트렌드는 인공지능, 확장성, 경량화, 온디바이스 처리, 개인화, 멀티모달 등의 키워드가 관련되어 있습니다."
            
            # 트렌드 키워드 추출 프롬프트
            prompt = f"""다음 검색어와 관련된 최신 기술 트렌드를 분석하고, 핵심 기술 키워드 5개를 추출해주세요:
            
            검색어: {query}
            검색 결과: {search_results}
            
            현재 기술 동향과 미래 발전 방향을 중심으로 분석해주세요.
            결과는 쉼표로 구분된 핵심 기술 키워드 5개만 정확히 반환해주세요."""
            
            # OpenAI API 직접 호출
            result = self._call_openai(prompt)
            
            # 쉼표로 구분된 결과 파싱
            trend_keywords = [kw.strip() for kw in result.split(",")][:5]
            logger.info(f"[TechEvaluationAgent] Extracted trend keywords: {trend_keywords}")
            return trend_keywords
            
        except Exception as e:
            logger.error(f"[TechEvaluationAgent] Error extracting trend keywords: {e}")
            # 오류 발생 시 기본 키워드 반환
            return ["인공지능", "경량화", "확장성", "최적화", "자동화"]
    
    def _calculate_trend_fit_scores(self, patents: List[Dict[str, Any]], trend_keywords: List[str]) -> List[float]:
        """특허 요약문(abstract)과 트렌드 키워드 간의 관련성 점수 계산 - OpenAI API 사용"""
        if not patents or not trend_keywords:
            return [0.0] * len(patents)
        
        trend_keywords_str = ", ".join(trend_keywords)
        scores = []
        
        for patent in patents:
            abstract = patent.get("abstract", "")
            if not abstract:
                scores.append(0.0)
                continue
            
            # OpenAI API를 사용하여 관련성 평가
            prompt = f"""
            다음 특허 요약문과 기술 트렌드 키워드 간의 관련성을 평가해주세요.
            
            특허 요약문: 
            {abstract}
            
            기술 트렌드 키워드: 
            {trend_keywords_str}
            
            위 특허 요약문이 언급된 기술 트렌드 키워드와 얼마나 관련이 있는지 1-5 척도로 점수를 매겨주세요.
            1: 거의 관련 없음
            2: 약간 관련 있음
            3: 어느 정도 관련 있음
            4: 상당히 관련 있음
            5: 매우 관련 있음
            
            최종 점수만 숫자로 반환해주세요.
            """
            
            try:
                response = self._call_openai(prompt)
                # 숫자만 추출
                score = float(''.join(c for c in response if c.isdigit() or c == '.'))
                # 범위 확인
                score = max(1.0, min(5.0, score))
                scores.append(score)
                logger.info(f"[TechEvaluationAgent] Patent {patent.get('patent_id')}: trend fit score={score}")
            except Exception as e:
                logger.error(f"[TechEvaluationAgent] Error calculating trend fit score: {e}")
                scores.append(1.0)  # 오류 시 최소값 할당
        
        return self._normalize_scores_to_five(scores)
    
    def _calculate_originality_scores(self, patents: List[Dict[str, Any]]) -> List[float]:
        """각 특허의 청구항과 다른 특허들의 청구항 간 코사인 유사도를 계산하여 독창성 점수 산출 - OpenAI API 사용"""
        if not patents:
            return []
        
        claims_texts = [patent.get("claims", "") for patent in patents]
        claims_embeddings = []
        
        logger.info(f"[TechEvaluationAgent] Calculating originality scores for {len(patents)} patents using OpenAI embeddings")
        
        # 청구항 텍스트를 OpenAI API로 임베딩 벡터 생성
        for i, claim in enumerate(claims_texts):
            if not claim:
                # 빈 청구항인 경우 임의의 임베딩 벡터 생성 (zeros)
                claims_embeddings.append(np.zeros(1536))  # OpenAI 임베딩 차원
                logger.warning(f"[TechEvaluationAgent] Empty claims for patent {patents[i].get('patent_id')}")
                continue
                
            try:
                # OpenAI API 직접 호출로 임베딩 계산
                url = "https://api.openai.com/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "text-embedding-ada-002",
                    "input": claim[:8000]  # 최대 8000자 제한
                }
                
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    embedding = response.json()["data"][0]["embedding"]
                    claims_embeddings.append(embedding)
                    logger.info(f"[TechEvaluationAgent] Generated embedding for patent {patents[i].get('patent_id')}")
                else:
                    logger.error(f"[TechEvaluationAgent] API error: {response.status_code}, {response.text}")
                    claims_embeddings.append(np.zeros(1536))  # 오류 시 기본값
            except Exception as e:
                logger.error(f"[TechEvaluationAgent] Error generating embeddings: {e}")
                claims_embeddings.append(np.zeros(1536))  # 오류 시 기본값
        
        # 유사도 행렬 계산
        similarity_matrix = np.zeros((len(claims_embeddings), len(claims_embeddings)))
        
        for i in range(len(claims_embeddings)):
            for j in range(len(claims_embeddings)):
                if i != j:  # 자기 자신과의 유사도는 계산하지 않음
                    # 코사인 유사도 계산
                    sim = cosine_similarity([claims_embeddings[i]], [claims_embeddings[j]])[0][0]
                    similarity_matrix[i][j] = sim
        
        # 각 특허의 평균 유사도 계산
        avg_similarities = []
        for idx, similarities in enumerate(similarity_matrix):
            # 0이 아닌 값들의 평균 계산 (자기 자신 제외)
            non_zero_similarities = [sim for i, sim in enumerate(similarities) if i != idx and sim > 0]
            avg_sim = sum(non_zero_similarities) / len(non_zero_similarities) if non_zero_similarities else 0
            avg_similarities.append(avg_sim)
            logger.info(f"[TechEvaluationAgent] Patent {patents[idx].get('patent_id')}: avg similarity={avg_sim}")
        
        # 전체 평균 유사도 계산
        global_avg_similarity = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
        logger.info(f"[TechEvaluationAgent] Global average similarity: {global_avg_similarity}")
        
        # 독창성 점수 계산: 유사도가 낮을수록 독창성 높음 (5점 만점)
        originality_scores = []
        for idx, avg_sim in enumerate(avg_similarities):
            if avg_sim == 0:  # 유사한 특허가 없는 경우
                score = 5.0
            elif global_avg_similarity == 0:  # 모든 특허의 평균 유사도가 0인 경우
                score = 2.5
            else:
                # 유사도 범위가 좁으므로(예: 0.8~0.9) 점수 계산 방식 수정
                # min_sim = min(avg_similarities)
                # max_sim = max(avg_similarities)
                # 유사도 범위를 [0, 1] 대신 실제 관찰된 범위로 정규화
                min_sim = min(avg_similarities) if avg_similarities else 0
                max_sim = max(avg_similarities) if avg_similarities else 1
                
                if max_sim - min_sim > 0.001:  # 범위가 의미 있을 정도로 클 경우
                    # 정규화된 유사도: 0 (가장 낮은 유사도) ~ 1 (가장 높은 유사도)
                    normalized_sim = (avg_sim - min_sim) / (max_sim - min_sim)
                    # 독창성 점수: 5 (가장 낮은 유사도) ~ 1 (가장 높은 유사도)
                    score = 5.0 - 4.0 * normalized_sim
                else:
                    # 모든 유사도가 거의 같을 경우 중간값 반환
                    score = 3.0
            
            originality_scores.append(score)
            logger.info(f"[TechEvaluationAgent] Patent {patents[idx].get('patent_id')}: originality score={score}")
        
        return self._normalize_scores_to_five(originality_scores)
    
    def _summarize_tech(self, claims_text):
        """특허 청구항에서 기술 요약 생성"""
        prompt = f"다음 특허 청구항을 분석하여 핵심 기술을 간결하게 요약해주세요:\n\n{claims_text}"
        return self._call_openai(prompt)

    def evaluate(self, patent_list, keywords=None):
        """
        특허 리스트에 대한 기술 평가 수행
        
        Args:
            patent_list: 평가할 특허 리스트
            keywords: 검색에 사용된 키워드 리스트 (기술 트렌드 분석용)
        
        Returns:
            Dict: tech_eval_results 키를 포함하는 딕셔너리 - 종합 보고서 포함
        """
        logger.info(f"[TechEvaluationAgent] Evaluating {len(patent_list)} patents")
        if keywords:
            logger.info(f"[TechEvaluationAgent] Using keywords: {keywords}")
        
        # 결과 저장 리스트
        results = []
        
        # 키워드가 없으면 기본값 사용
        if not keywords:
            keywords = ["인공지능", "특허"]
        
        # 1. 기술 트렌드 키워드 추출
        trend_keywords = self._extract_trend_keywords(keywords)
        logger.info(f"[TechEvaluationAgent] Tech trend keywords: {trend_keywords}")
        
        # 2. 각 특허의 트렌드 적합성 점수 계산 (BM25 사용)
        trend_fit_scores = self._calculate_trend_fit_scores(patent_list, trend_keywords)
        
        # 3. 각 특허의 독창성 점수 계산 (청구항 유사도 기반)
        originality_scores = self._calculate_originality_scores(patent_list)
        
        # 전체 평균 점수 계산
        avg_trend_fit_score = sum(trend_fit_scores) / len(trend_fit_scores) if trend_fit_scores else 0
        avg_originality_score = sum(originality_scores) / len(originality_scores) if originality_scores else 0
        
        # 4. 각 특허의 기술 요약 및 결과 조합
        for i, patent in enumerate(patent_list):
            # patent 객체에 claims 키가 없을 경우를 대비 (KIPRIS 조회 실패 등)
            claims_text = patent.get("claims")
            if not claims_text:
                logger.warning(f"[TechEvaluationAgent] No claims found for patent {patent.get('patent_id', 'Unknown')}. Using limited info.")
                tech_summary_str = "기술 정보를 요약할 수 없습니다 (청구항 누락)."
            else:
                # OpenAI API 직접 호출로 요약 생성
                tech_summary_str = self._summarize_tech(claims_text)
            
            # 해당 특허의 점수 가져오기
            originality_score = originality_scores[i] if i < len(originality_scores) else 0.0
            trend_fit_score = trend_fit_scores[i] if i < len(trend_fit_scores) else 0.0
            
            # 특허 제목과 요약, 출원일, 출원인 가져오기
            patent_id = patent.get("patent_id", "ID 없음")
            title = patent.get("title", "제목 없음")
            abstract = patent.get("abstract", "요약 없음")
            filing_date = patent.get("filing_date", "출원일 정보 없음")
            applicant = patent.get("applicant", "출원인 정보 없음")
            
            # 종합 보고서 생성
            analysis_report = self._generate_patent_analysis_report(
                patent_id=patent_id,
                title=title,
                abstract=abstract,
                tech_summary=tech_summary_str,
                trend_fit_score=trend_fit_score,
                originality_score=originality_score,
                trend_keywords=trend_keywords,
                filing_date=filing_date,
                applicant=applicant
            )

            results.append({
                "patent_id": patent_id,
                "title": title,
                "abstract": abstract,
                "filing_date": filing_date,
                "applicant": applicant,
                "originality_score": round(originality_score, 2),
                "trend_fit_score": round(trend_fit_score, 2),
                "tech_summary": tech_summary_str,
                "trend_keywords": trend_keywords,
                "analysis_report": analysis_report
            })
        
        # 종합 분석 보고서 생성
        overall_analysis = self._generate_overall_analysis(
            results=results,
            trend_keywords=trend_keywords,
            avg_trend_fit_score=avg_trend_fit_score,
            avg_originality_score=avg_originality_score,
            keywords=keywords
        )
            
        logger.info(f"[TechEvaluationAgent] Evaluation complete for {len(results)} patents")
        return {
            "tech_eval_results": {
                "patents": results,
                "overall_analysis": overall_analysis,
                "trend_keywords": trend_keywords,
                "avg_trend_fit_score": round(avg_trend_fit_score, 2),
                "avg_originality_score": round(avg_originality_score, 2),
                "search_keywords": keywords
            }
        }
        
    def _generate_patent_analysis_report(self, patent_id, title, abstract, tech_summary, trend_fit_score, originality_score, trend_keywords, filing_date="", applicant=""):
        """
        개별 특허 분석 보고서 생성
        """
        trend_keywords_str = ", ".join(trend_keywords)
        
        report = f"""
# 특허 기술 평가 보고서: {patent_id}

## 기본 정보
- **특허 ID**: {patent_id}
- **특허 제목**: {title}
- **출원일**: {filing_date}
- **출원인**: {applicant}

## 기술 평가 점수
- **트렌드 적합성 점수**: {trend_fit_score}/5.0
- **독창성 점수**: {originality_score}/5.0

## 기술 요약
{tech_summary}

## 특허 초록
{abstract}

## 트렌드 키워드
{trend_keywords_str}

## 종합 평가
"""
        
        # 트렌드 적합성 평가
        if trend_fit_score >= 4.0:
            report += "- 이 특허는 현재 기술 트렌드와 매우 높은 관련성을 보입니다.\n"
        elif trend_fit_score >= 3.0:
            report += "- 이 특허는 현재 기술 트렌드와 상당한 관련성을 가지고 있습니다.\n"
        elif trend_fit_score >= 2.0:
            report += "- 이 특허는 현재 기술 트렌드와 일부 관련성을 가지고 있습니다.\n"
        else:
            report += "- 이 특허는 현재 기술 트렌드와 관련성이 낮습니다.\n"
            
        # 독창성 평가
        if originality_score >= 4.0:
            report += "- 이 특허는 매우 높은 독창성을 가지고 있습니다.\n"
        elif originality_score >= 3.0:
            report += "- 이 특허는 상당한 독창성을 가지고 있습니다.\n"
        elif originality_score >= 2.0:
            report += "- 이 특허는 적정 수준의 독창성을 가지고 있습니다.\n"
        else:
            report += "- 이 특허는 상대적으로 독창성이 낮습니다.\n"
            
        # 종합 권고
        if trend_fit_score >= 3.5 and originality_score >= 3.5:
            report += "\n**최종 평가**: 이 특허는 높은 트렌드 적합성과 독창성을 겸비하고 있어 매우 가치가 높을 것으로 평가됩니다."
        elif trend_fit_score >= 3.5:
            report += "\n**최종 평가**: 이 특허는 현재 기술 트렌드와 잘 부합하지만, 독창성 측면에서 개선이 필요합니다."
        elif originality_score >= 3.5:
            report += "\n**최종 평가**: 이 특허는 높은 독창성을 가지고 있지만, 현재 기술 트렌드와의 연관성이 부족합니다."
        else:
            report += "\n**최종 평가**: 이 특허는 트렌드 적합성과 독창성 모두 개선이 필요합니다."
        
        return report
        
    def _generate_overall_analysis(self, results, trend_keywords, avg_trend_fit_score, avg_originality_score, keywords):
        """
        전체 특허 종합 분석 보고서 생성
        """
        # 결과를 점수에 따라 정렬
        trend_sorted_patents = sorted(results, key=lambda x: x['trend_fit_score'], reverse=True)
        originality_sorted_patents = sorted(results, key=lambda x: x['originality_score'], reverse=True)
        
        # 상위 3개 특허 추출 (리스트 길이가 3 미만인 경우 고려)
        top_trend_patents = trend_sorted_patents[:min(3, len(trend_sorted_patents))]
        top_originality_patents = originality_sorted_patents[:min(3, len(originality_sorted_patents))]
        
        # 상위 특허 ID와 점수 문자열 생성
        top_trend_str = '\n'.join([f"- {p['patent_id']} ({p['title']}): {p['trend_fit_score']}/5.0" for p in top_trend_patents])
        top_originality_str = '\n'.join([f"- {p['patent_id']} ({p['title']}): {p['originality_score']}/5.0" for p in top_originality_patents])
        
        # 검색 키워드 및 트렌드 키워드 문자열
        keywords_str = ', '.join(keywords)
        trend_keywords_str = ', '.join(trend_keywords)
        
        report = f"""
# 특허 기술 분석 종합 보고서

## 검색 정보
- **검색 키워드**: {keywords_str}
- **분석된 특허 수**: {len(results)}

## 기술 트렌드 분석
- **발견된 트렌드 키워드**: {trend_keywords_str}
- **평균 트렌드 적합성 점수**: {avg_trend_fit_score:.2f}/5.0

## 독창성 분석
- **평균 독창성 점수**: {avg_originality_score:.2f}/5.0

## 트렌드 적합성 상위 특허
{top_trend_str}

## 독창성 상위 특허
{top_originality_str}

## 종합 분석
"""
        # 종합 분석 내용 - 전체 특허 포트폴리오 분석
        if avg_trend_fit_score >= 4.0:
            report += "- 검색된 특허들은 전반적으로 현재 기술 트렌드와 매우 높은 관련성을 보입니다.\n"
        elif avg_trend_fit_score >= 3.0:
            report += "- 검색된 특허들은 전반적으로 현재 기술 트렌드와 상당한 관련성을 가지고 있습니다.\n"
        else:
            report += "- 검색된 특허들은 전반적으로 현재 기술 트렌드와의 관련성이 낮습니다.\n"
            
        if avg_originality_score >= 4.0:
            report += "- 검색된 특허들은 전반적으로 매우 높은 독창성을 가지고 있습니다.\n"
        elif avg_originality_score >= 3.0:
            report += "- 검색된 특허들은 전반적으로 상당한 독창성을 가지고 있습니다.\n"
        else:
            report += "- 검색된 특허들은 전반적으로 독창성이 낮습니다.\n"
        
        # 특허 포트폴리오 추천
        if avg_trend_fit_score >= 3.5 and avg_originality_score >= 3.5:
            report += "\n**최종 평가**: 검색된 특허 포트폴리오는 높은 트렌드 적합성과 독창성을 겸비하고 있어 매우 가치가 높을 것으로 평가됩니다. 이 기술 영역은 혁신적이면서도 시장 트렌드에 부합하는 특성을 가지고 있습니다."
        elif avg_trend_fit_score >= 3.5:
            report += "\n**최종 평가**: 검색된 특허 포트폴리오는 현재 기술 트렌드와 잘 부합하지만, 독창성 측면에서 개선이 필요합니다. 더 혁신적인 기술 개발을 통해 경쟁력을 강화할 필요가 있습니다."
        elif avg_originality_score >= 3.5:
            report += "\n**최종 평가**: 검색된 특허 포트폴리오는 높은 독창성을 가지고 있지만, 현재 기술 트렌드와의 연관성이 부족합니다. 시장 트렌드에 맞는 응용 분야를 발굴하여 상업적 가치를 높일 필요가 있습니다."
        else:
            report += "\n**최종 평가**: 검색된 특허 포트폴리오는 트렌드 적합성과 독창성 모두 개선이 필요합니다. 새로운 혁신적 기술 개발과 시장 트렌드 분석을 통해 특허 전략을 재수립할 필요가 있습니다."
        
        return report 