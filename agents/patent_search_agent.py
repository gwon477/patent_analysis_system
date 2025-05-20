import os
import requests
import xml.etree.ElementTree as ET
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.chunking import split_claims_to_chunks

from dotenv import load_dotenv
load_dotenv()

KIPRIS_SEARCH_API = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"
KIPRIS_ABSTRACT_API = "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/patentAbstractInfo"
KIPRIS_CLAIM_API = "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/patentClaimInfo"
KIPRIS_BIBLIO_API = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getBibliographySumryInfoSearch"
KIPRIS_SERVICE_KEY = os.getenv("KIPRIS_SERVICE_KEY")  # 반드시 .env에 저장

class PatentSearchAgent:
    def __init__(self, db_path="./data/embeddings/", embedding_model_name="jhgan/ko-sroberta-multitask"):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.db_path = db_path
        self.chroma = Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_function
        )

    def _search_application_numbers(self, keyword, year=0):
        print(f"[PatentSearchAgent] _search_application_numbers: {keyword}, {year}")
        #print(f"[PatentSearchAgent] KIPRIS_SERVICE_KEY: {KIPRIS_SERVICE_KEY}")
        if not KIPRIS_SERVICE_KEY:
            print("[PatentSearchAgent ERROR] KIPRIS_SERVICE_KEY is not set. Please check your .env file.")
            return []

        params = {
            "word": keyword,
            "year": year,
            "ServiceKey": KIPRIS_SERVICE_KEY
        }
        app_nums = []
        try:
            resp = requests.get(KIPRIS_SEARCH_API, params=params, timeout=15)
            #print(f"[PatentSearchAgent DEBUG] KIPRIS API Response Status Code for keyword '{keyword}': {resp.status_code}")
            #print(f"[PatentSearchAgent DEBUG] KIPRIS API Response Text for keyword '{keyword}': {resp.text[:500]}...") # 첫 500자만 출력
            resp.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            try:
                root = ET.fromstring(resp.text)
                items = root.find(".//items")
                if items is not None:
                    for item in items.findall("item"):
                        num = item.findtext("applicationNumber")
                        if num:
                            app_nums.append(num)
                else:
                    print(f"[PatentSearchAgent DEBUG] No 'items' tag found in KIPRIS API response for keyword '{keyword}'.")
            except ET.ParseError as e:
                print(f"[PatentSearchAgent ERROR] XML ParseError for keyword '{keyword}': {e}")
                print(f"[PatentSearchAgent DEBUG] Faulty XML Response Text: {resp.text[:1000]}...") # 파싱 오류 시 XML 앞부분 출력
                return [] # 파싱 오류 시 빈 리스트 반환

        except requests.exceptions.RequestException as e:
            print(f"[PatentSearchAgent ERROR] KIPRIS API request failed for keyword '{keyword}': {e}")
            return [] # 요청 실패 시 빈 리스트 반환
        
        # 검색된 특허 번호 출력
        print(f"[PatentSearchAgent] _search_application_numbers found for '{keyword}': {app_nums}")
        return app_nums

    def _fetch_abstract(self, application_number):

        if not KIPRIS_SERVICE_KEY:
            print("[PatentSearchAgent ERROR] KIPRIS_SERVICE_KEY is not set for _fetch_abstract.")
            return ""
        params = {
            "applicationNumber": application_number,
            "accessKey": KIPRIS_SERVICE_KEY
        }
        resp = requests.get(KIPRIS_ABSTRACT_API, params=params, timeout=10)
        if resp.status_code != 200:
            return ""
        try:
            root = ET.fromstring(resp.text)
            return root.findtext(".//astrtCont", default="")
        except Exception:
            return ""

    def _fetch_bibliography(self, application_number):
        """특허 번호로 출원일(applicationDate)과 발명 제목(inventionTitle) 정보를 가져옵니다."""
        if not KIPRIS_SERVICE_KEY:
            print("[PatentSearchAgent ERROR] KIPRIS_SERVICE_KEY is not set for _fetch_bibliography.")
            return {}, ""
        
        params = {
            "applicationNumber": application_number,
            "ServiceKey": KIPRIS_SERVICE_KEY
        }
        
        try:
            resp = requests.get(KIPRIS_BIBLIO_API, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"[PatentSearchAgent ERROR] KIPRIS Biblio API failed with status code {resp.status_code} for {application_number}")
                return {}, ""
            
            try:
                root = ET.fromstring(resp.text)
                application_date = root.findtext(".//applicationDate", default="")
                invention_title = root.findtext(".//inventionTitle", default="")
                
                # 추가로 출원인 정보도 가져올 수 있다면 확장 가능
                applicant = root.findtext(".//applicantName", default="")
                
                return {
                    "application_date": application_date,
                    "invention_title": invention_title,
                    "applicant": applicant
                }, resp.text
                
            except ET.ParseError as e:
                print(f"[PatentSearchAgent ERROR] XML ParseError for bibliography of {application_number}: {e}")
                return {}, ""
                
        except requests.exceptions.RequestException as e:
            print(f"[PatentSearchAgent ERROR] KIPRIS Biblio API request failed for {application_number}: {e}")
            return {}, ""

    def _fetch_claims(self, application_number):
        if not KIPRIS_SERVICE_KEY:
            print("[PatentSearchAgent ERROR] KIPRIS_SERVICE_KEY is not set for _fetch_claims.")
            return ""
        params = {
            "applicationNumber": application_number,
            "accessKey": KIPRIS_SERVICE_KEY
        }
        resp = requests.get(KIPRIS_CLAIM_API, params=params, timeout=10)
        if resp.status_code != 200:
            return ""
        try:
            root = ET.fromstring(resp.text)
            claim_texts = []
            for ci in root.findall(".//claimInfo"):
                text = ci.findtext("claim", "")
                if text:
                    claim_texts.append(text.strip())
            return "\n".join(claim_texts)
        except Exception:
            return ""

    def _fetch_metadata(self, application_number):
        """특허 번호로 메타데이터를 가져옵니다. 이제 출원일과 발명 제목 정보도 포함됩니다."""
        # 기본 메타데이터 설정
        metadata = {
            "patent_id": application_number,
            "title": "",
            "applicant": "",
            "filing_date": "",
        }
        
        # 서지 정보(출원일, 발명 제목 등) 가져오기
        biblio_data, _ = self._fetch_bibliography(application_number)
        
        # 메타데이터 업데이트
        if biblio_data:
            metadata["title"] = biblio_data.get("invention_title", "")
            metadata["filing_date"] = biblio_data.get("application_date", "")
            metadata["applicant"] = biblio_data.get("applicant", "")
            
        return metadata

    def search_and_store(self, keywords, year_range=(0,0), max_patents_per_keyword=5, **kwargs):
        print(f"[PatentSearchAgent] Searching for patents with keywords: {keywords}")
        all_app_nums = set()
        for keyword in keywords:
            # KIPRIS API는 단일 연도만 지원하므로, 연도 범위를 어떻게 처리할지 결정 필요.
            # 여기서는 year_range의 첫 번째 값을 사용하거나, 최근 연도를 기준으로 검색하도록 단순화.
            # 또는 각 연도별로 검색을 반복할 수 있음.
            # 우선은 year_range[0] (시작년도) 또는 기본값 0을 사용하도록 함.
            # 실제 API가 특정 기간 검색을 지원하지 않는다면, 최근 N년 등으로 로직 변경 필요.
            # subscript.md 에서는 year 인자를 사용하고 있었음. 여기서는 year_range[0]을 사용.
            search_year = year_range[0] if year_range and len(year_range) > 0 else 0 
            app_nums_for_keyword = self._search_application_numbers(keyword, search_year)
            all_app_nums.update(app_nums_for_keyword)
        
        patent_list = []
        docs = []

        # 중복 제거된 출원번호에 대해서만 정보 조회
        # processed_app_nums = list(all_app_nums)[:max_patents_per_keyword * len(keywords)] # 전체 결과 수 제한 로직 제거
        # all_app_nums는 set이므로 순서를 보장하고 중복을 제거한 리스트로 변환하여 사용
        final_app_nums_to_process = sorted(list(all_app_nums)) # 정렬 추가 (선택 사항이지만 일관성을 위해)

        print(f"[DEBUG] final_app_nums_to_process: {final_app_nums_to_process}")
        
        # 모든 검색된 고유 출원번호에 대해서 정보 조회
        for app_num in final_app_nums_to_process: 
            abstract = self._fetch_abstract(app_num)
            claims = self._fetch_claims(app_num)
            meta = self._fetch_metadata(app_num) # 이제 title, applicant, filing_date가 API에서 가져온 값으로 채워짐
            
            # meta의 title이 비어있을 경우에만 키워드로 대체
            if not meta['title']:
                meta['title'] = keywords[0] if keywords else "N/A"
            
            meta.update({
                "abstract": abstract,
                "claims": claims
            })
            patent_list.append(meta)

            base_text = claims if claims else abstract
            if not base_text: # 청구항과 초록 모두 없는 경우
                print(f"[PatentSearchAgent] Skipping {app_num} due to no content for claims or abstract.")
                continue
            
            chunks = split_claims_to_chunks(base_text)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "patent_id": app_num,
                        "chunk_index": idx,
                        "chunk": chunk, # 중복 저장되지만, 검색 시 편의를 위해 유지
                        "abstract": abstract,
                        "title": meta.get("title", ""),
                        "claims": claims,
                        "applicant": meta.get("applicant", ""),
                        "filing_date": meta.get("filing_date", ""),
                        "source": "KIPRIS"
                    }
                )
                docs.append(doc)

        if docs:
            print(f"[PatentSearchAgent] {len(docs)} claim/abstract chunks to embed/store...")
            self.chroma.add_documents(docs)
            print(f"[PatentSearchAgent] Embedding and storage complete.")
        else:
            print("[PatentSearchAgent] No documents to embed.")

        return {
            "patent_list": patent_list, # 실제 특허 메타데이터 리스트
            "retriever": self.chroma.as_retriever(search_kwargs={"k": 5}), # Langchain Retriever
        } 