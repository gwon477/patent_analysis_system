import json

def cached_market_search(query, cache_path="./data/market_info_cache.json"):
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    if query in cache:
        return cache[query]
    # 실전에서는 PyTrends 등 실제 시장 데이터 API 연동
    result = {"query": query, "market_size": "120억 달러", "cagr": "22%"}
    cache[query] = result
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return result 