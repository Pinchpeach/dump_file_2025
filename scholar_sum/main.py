# scholar_ins_optical.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import requests
import pandas as pd
from transformers import pipeline

# 1️⃣ 검색 키워드 설정
SEARCH_KEYWORDS = ["Neutron Star", "Optical Observation"]

# 2️⃣ Semantic Scholar API 설정
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,url"

# 3️⃣ 논문 검색 함수
def search_papers(query, limit=5):
    response = requests.get(
        BASE_URL,
        params={
            "query": query,
            "limit": limit,
            "fields": FIELDS
        }
    )
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

# 4️⃣ 논문 요약기 준비 (T5 모델 사용)
summarizer = pipeline("summarization", model="t5-base")

# 5️⃣ 메인 실행
def main():
    all_papers = []

    for keyword in SEARCH_KEYWORDS:
        print(f"🔍 Searching papers for: {keyword}")
        papers = search_papers(keyword)

        for paper in papers:
            title = paper.get('title', 'No Title')
            abstract = paper.get('abstract', 'No Abstract Available')
            url = paper.get('url', 'No URL')

            if abstract != 'No Abstract Available' and len(abstract.split()) > 30:
                summary = summarizer(abstract, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
            else:
                summary = "Abstract too short or missing for summarization."

            all_papers.append({
                "Keyword": keyword,
                "Title": title,
                "Abstract": abstract,
                "Summary": summary,
                "URL": url
            })

    # 6️⃣ 결과 저장
    df = pd.DataFrame(all_papers)
    df.to_csv("scholar_sum/papers_ins_optical.csv", index=False)
    print("✅ Results saved to 'papers_ins_optical.csv'")

if __name__ == "__main__":
    main()
