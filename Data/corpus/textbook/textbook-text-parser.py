import os
import re
import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders import PDFPlumberLoader

# 현재 파일 위치를 작업 디렉토리로 설정
current_file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_file_path))

# 경로 설정
DOCUMENTS_DIR = "./Documents"
OUTPUT_EXCEL_PATH = "./loaded-textbooks.xlsx"


def remove_urls(text: str) -> str:
    """텍스트에서 URL 제거"""
    return re.sub(r"http\S+", "", text)


def load_pdf_pages(file_path: str) -> list[dict]:
    """PDF 파일을 페이지 단위로 lazy 로드하여 행 리스트로 반환

    lazy_load()를 사용하여 500페이지 이상의 대용량 PDF를
    한 번에 메모리에 올리지 않고 페이지 단위로 순회한다.
    """
    loader = PDFPlumberLoader(file_path)
    filename = os.path.basename(file_path)
    rows = []

    for doc in tqdm(loader.lazy_load(), desc=f"  {filename}", unit="page"):
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", ""))
        contents = remove_urls(doc.page_content)
        rows.append({
            "file": filename,
            "page": page_num,
            "contents": contents,
        })

    return rows


def process_all_pdfs(documents_dir: str) -> pd.DataFrame:
    """Documents/ 디렉토리 내 모든 PDF를 처리하고 결과를 하나의 DataFrame으로 반환"""
    pdf_files = sorted([f for f in os.listdir(documents_dir) if f.endswith(".pdf")])

    if not pdf_files:
        print("처리할 PDF 파일이 없습니다.")
        return pd.DataFrame(columns=["file", "page", "contents"])

    print(f"총 {len(pdf_files)}개 PDF 처리 시작\n")
    all_rows = []

    for idx, filename in enumerate(pdf_files, start=1):
        file_path = os.path.join(documents_dir, filename)
        print(f"[{idx}/{len(pdf_files)}] 처리 중: {filename}")
        rows = load_pdf_pages(file_path)
        all_rows.extend(rows)
        print(f"  → {len(rows)}페이지 로드 완료\n")

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = process_all_pdfs(DOCUMENTS_DIR)

    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"완료: {OUTPUT_EXCEL_PATH}")
    print(f"  처리 파일 수 : {df['file'].nunique()}개")
    print(f"  총 페이지 수 : {len(df)}페이지")
    print(f"  빈 페이지 수 : {(df['contents'].str.strip() == '').sum()}페이지")
