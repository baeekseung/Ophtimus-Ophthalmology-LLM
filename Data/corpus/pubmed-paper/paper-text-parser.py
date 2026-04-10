import os
import json
import re
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader

# 현재 파일 위치를 작업 디렉토리로 설정
current_file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_file_path))

# 경로 설정
DOCUMENTS_DIR = "./Documents"
CLASSIFIER_RESULTS_PATH = "./paper-layout-classifier-results.json"
OUTPUT_EXCEL_PATH = "./parsed-papers.xlsx"


def load_classifier_results(json_path: str) -> dict[str, str]:
    """레이아웃 분류 결과를 로드하여 {파일명: 레이아웃} 딕셔너리로 반환"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["file"]: item["layout"] for item in data["results"]}


def get_loader(file_path: str, layout: str):
    """레이아웃에 따라 적절한 LangChain 로더 반환
    - 1단 : PDFPlumberLoader
    - 2단 : PyMuPDFLoader
    """
    if "single_column" in layout or "1단" in layout:
        return PDFPlumberLoader(file_path)
    else:
        return PyMuPDFLoader(file_path)


def remove_urls(text: str) -> str:
    """텍스트에서 URL 제거"""
    return re.sub(r"http\S+", "", text)


def process_pdf(file_path: str, layout: str) -> list[dict]:
    """PDF 파일을 로드하고 페이지별 텍스트를 행 리스트로 반환"""
    loader = get_loader(file_path, layout)
    documents = loader.load()

    return [
        {
            "file": os.path.basename(file_path),
            "layout": layout,
            "page": doc.metadata.get("page", doc.metadata.get("page_number", "")),
            "contents": remove_urls(doc.page_content),
        }
        for doc in documents
    ]


def process_all_pdfs(documents_dir: str, classifier_results: dict[str, str]) -> pd.DataFrame:
    """모든 PDF를 처리하고 결과를 하나의 DataFrame으로 반환"""
    all_rows = []
    total = len(classifier_results)

    for idx, (filename, layout) in enumerate(classifier_results.items(), start=1):
        file_path = os.path.join(documents_dir, filename)

        if not os.path.exists(file_path):
            print(f"[{idx}/{total}] 파일 없음 (건너뜀): {filename}")
            continue

        print(f"[{idx}/{total}] 처리 중: {filename} ({layout})")
        rows = process_pdf(file_path, layout)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # 레이아웃 분류 결과 로드
    classifier_results = load_classifier_results(CLASSIFIER_RESULTS_PATH)
    print(f"총 {len(classifier_results)}개 문서 처리 시작\n")

    # 모든 PDF 처리
    df = process_all_pdfs(DOCUMENTS_DIR, classifier_results)

    # 단일 엑셀 파일로 저장
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\n완료: {OUTPUT_EXCEL_PATH} ({len(df)}행 저장)")
