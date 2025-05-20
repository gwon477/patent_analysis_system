import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import requests
from langchain_community.tools.tavily_search import TavilySearchResults
import openai
import re

# 로깅 설정
logger = logging.getLogger(__name__)

class MarketSearchTool:
    """Tavily 웹 검색 API를 사용한 시장 정보 검색 도구"""
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Tavily API 키
        """
        self.api_key = api_key
        self.search_tool = TavilySearchResults(api_key=api_key)
    
    def search(self, query: str, search_type: str = "market_size") -> List[Dict[str, Any]]:
        """
        주어진 쿼리와 검색 유형에 따라 Tavily 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            search_type: 검색 유형 ("market_size" 또는 "growth_rate")
            
        Returns:
            검색 결과 목록
        """
        try:
            if search_type == "market_size":
                full_query = f"{query} market size billion dollars global market"
            elif search_type == "growth_rate":
                full_query = f"{query} market CAGR growth rate percentage annual"
            else:
                full_query = query
                
            logger.info(f"[MarketSearchTool] Searching for: {full_query}")
            search_results = self.search_tool.invoke(full_query)
            logger.info(f"[MarketSearchTool] Found {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"[MarketSearchTool] Error during search: {e}")
            return []

def expand_keyword(keyword: str, openai_api_key: str, num_expansions: int = 3) -> List[str]:
    """
    OpenAI API를 사용하여 주어진 키워드를 확장합니다.
    
    Args:
        keyword: 확장할 키워드
        openai_api_key: OpenAI API 키
        num_expansions: 생성할 관련 키워드 수
        
    Returns:
        확장된 키워드 목록 (원본 키워드 포함)
    """
    try:
        # proxies 관련 설정을 제거하고 필수 매개변수만 전달
        client = openai.OpenAI(api_key=openai_api_key, default_headers=None, max_retries=2)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 시장 조사 및 키워드 확장 전문가입니다. 주어진 기술/제품 키워드와 관련된 시장 용어와 산업 키워드를 제안해주세요."},
                {"role": "user", "content": f"""다음 기술/제품과 관련된 시장 검색에 유용한 관련 키워드 {num_expansions}개를 제공해주세요:

1. 관련 시장 용어나 산업 카테고리를 포함할 것
2. 검색 시 발견 가능성이 높은 키워드를 선택할 것
3. 원래 키워드보다 더 넓은 시장 범주나 응용 분야도 고려할 것
4. 쉼표로 구분하여 키워드만 나열할 것 (다른 설명 없이)

원본 키워드: {keyword}"""}
            ],
            temperature=0.3,
        )
        
        expanded_keywords_text = response.choices[0].message.content.strip()
        expanded_keywords = [kw.strip() for kw in expanded_keywords_text.split(',')]
        
        # 원본 키워드가 리스트에 없으면 추가
        if keyword not in expanded_keywords:
            expanded_keywords.insert(0, keyword)
        
        return expanded_keywords[:num_expansions+1]  # 원본 + 확장 키워드 (최대 num_expansions개)
    except Exception as e:
        logger.error(f"[KeywordExpansion] Error expanding keyword '{keyword}': {e}")
        return [keyword]  # 오류 발생 시 원본 키워드만 반환

def extract_market_size(text: str) -> float:
    """
    텍스트에서 시장 규모(억 달러)를 추출합니다.
    
    Args:
        text: 시장 정보 텍스트
        
    Returns:
        추출된 시장 규모 (추출 실패 시 0.0)
    """
    # 패턴: 숫자 + 단위(billion, million 등)
    patterns = [
        r'(\d+\.?\d*)\s*billion\s*dollars',         # 10 billion dollars, 10.5 billion dollars
        r'(\d+\.?\d*)\s*billion\s*USD',             # 10 billion USD
        r'(\d+\.?\d*)\s*B\s*USD',                   # 10B USD
        r'USD\s*(\d+\.?\d*)\s*billion',             # USD 10 billion
        r'(\d+\.?\d*)\s*억\s*달러',                 # 10억 달러
        r'(\d+\.?\d*)\s*billion',                   # 10 billion
        r'market\s*size.*?(\d+\.?\d*)\s*billion',   # market size ... 10 billion
        r'market\s*size.*?\$\s*(\d+\.?\d*)\s*billion', # market size ... $10 billion
        r'\$\s*(\d+\.?\d*)\s*billion',              # $10 billion
        r'(\d+\.?\d*)\s*bn',                        # 10 bn
        r'market\s*value.*?(\d+\.?\d*)\s*billion',  # market value ... 10 billion
        r'market\s*worth.*?(\d+\.?\d*)\s*billion',  # market worth ... 10 billion
        # million 단위는 billion으로 변환 (1/1000)
        r'(\d+\.?\d*)\s*million\s*dollars',         # 10 million dollars -> 0.01 billion
        r'(\d+\.?\d*)\s*million\s*USD',             # 10 million USD -> 0.01 billion
        r'(\d+\.?\d*)\s*M\s*USD',                   # 10M USD -> 0.01 billion
        r'\$\s*(\d+\.?\d*)\s*million',              # $10 million -> 0.01 billion
        r'(\d+\.?\d*)\s*million',                   # 10 million -> 0.01 billion
    ]
    
    # 텍스트 전처리 (소문자 변환)
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                # million 단위인 경우 billion으로 변환
                if 'million' in pattern or 'M ' in pattern:
                    return value / 1000.0
                return value
            except ValueError:
                continue
    
    return 0.0

def extract_growth_rate(text: str) -> float:
    """
    텍스트에서 성장률(CAGR %)을 추출합니다.
    
    Args:
        text: 시장 정보 텍스트
        
    Returns:
        추출된 성장률 (추출 실패 시 0.0)
    """
    # 패턴: CAGR + 숫자 + %
    patterns = [
        r'CAGR\s*of\s*(\d+\.?\d*)\s*%',             # CAGR of 10%, CAGR of 10.5%
        r'CAGR\s*(\d+\.?\d*)\s*%',                  # CAGR 10%
        r'grow\s*at\s*(\d+\.?\d*)\s*%',             # grow at 10%
        r'growth\s*rate\s*of\s*(\d+\.?\d*)\s*%',    # growth rate of 10%
        r'(\d+\.?\d*)\s*%\s*CAGR',                  # 10% CAGR
        r'증가율[\s\w]*(\d+\.?\d*)\s*%',            # 증가율 10%
        r'성장률[\s\w]*(\d+\.?\d*)\s*%',            # 성장률 10%
        r'annual\s*growth\s*rate.*?(\d+\.?\d*)\s*%', # annual growth rate ... 10%
        r'projected\s*to\s*grow.*?(\d+\.?\d*)\s*%', # projected to grow ... 10%
        r'compound\s*annual\s*growth\s*rate.*?(\d+\.?\d*)\s*%', # compound annual growth rate ... 10%
        r'expected\s*to\s*grow.*?(\d+\.?\d*)\s*%',  # expected to grow ... 10%
        r'growth.*?(\d+\.?\d*)\s*percent',          # growth ... 10 percent
        r'growth.*?(\d+\.?\d*)\s*%',                # growth ... 10%
        r'expand\s*at.*?(\d+\.?\d*)\s*%',           # expand at ... 10%
    ]
    
    # 텍스트 전처리 (소문자 변환)
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                # 성장률이 비상식적으로 높은 경우 제한 (100% 이상은 오류일 가능성 높음)
                if value > 100:
                    continue
                return value
            except ValueError:
                continue
    
    return 0.0

def normalize_scores(value: float, min_val: float, max_val: float, new_min: float = 1.0, new_max: float = 5.0) -> float:
    """
    값을 정규화하여 지정된 범위(기본값: 1.0-5.0)로 변환합니다.
    
    Args:
        value: 정규화할 값
        min_val: 원래 값의 최소 범위
        max_val: 원래 값의 최대 범위
        new_min: 새 범위의 최소값
        new_max: 새 범위의 최대값
        
    Returns:
        정규화된 값
    """
    if min_val == max_val:
        return (new_min + new_max) / 2  # 범위가 없으면 중간값 반환
    
    if value <= min_val:
        return new_min
    
    if value >= max_val:
        return new_max
        
    normalized = (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return round(normalized, 1)  # 소수점 첫째 자리까지 반올림

def cache_search_results(cache_key: str, data: Any, cache_dir: str = "./data/market_cache"):
    """
    검색 결과를 캐시에 저장합니다.
    
    Args:
        cache_key: 캐시 키
        data: 저장할 데이터
        cache_dir: 캐시 디렉토리
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key.replace(' ', '_')}.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"[Cache] Saved search results to {cache_file}")
    except Exception as e:
        logger.error(f"[Cache] Error saving cache: {e}")

def load_cached_results(cache_key: str, cache_dir: str = "./data/market_cache") -> Dict[str, Any]:
    """
    캐시에서 검색 결과를 로드합니다.
    
    Args:
        cache_key: 캐시 키
        cache_dir: 캐시 디렉토리
        
    Returns:
        캐시된 데이터 (없으면 빈 딕셔너리)
    """
    try:
        cache_file = os.path.join(cache_dir, f"{cache_key.replace(' ', '_')}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"[Cache] Loaded from cache: {cache_file}")
            return data
        else:
            logger.info(f"[Cache] No cache found for {cache_key}")
            return {}
    except Exception as e:
        logger.error(f"[Cache] Error loading cache: {e}")
        return {} 