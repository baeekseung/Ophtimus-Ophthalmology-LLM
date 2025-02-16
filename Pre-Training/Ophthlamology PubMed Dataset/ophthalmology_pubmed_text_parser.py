import os
import pandas as pd
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader

# 현재 파일의 경로를 기준으로 작업 디렉토리 설정
current_file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_file_path))

# 입력 및 출력 디렉토리 경로 설정
directory_path = "./Ophthalmology PubMed Papers"
output_directory = "./Parsed Ophthalmology PubMed"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load(), fitz.open(file_path)


def extract_text_from_pdf(pdf_document, top_margin=0, bottom_margin=1):
    all_chunks = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        page_height = page.rect.height
        page_width = page.rect.width

        # 페이지의 텍스트를 지정된 영역에서 추출
        rect = fitz.Rect(
            0, page_height * top_margin, page_width, page_height * bottom_margin
        )
        extracted_text = page.get_text("text", clip=rect)

        metadata = {"page": page_num + 1}
        all_chunks.append({"contents": extracted_text, "metadata": metadata})
    return all_chunks


def split_documents(documents):
    # 페이지 전체를 하나의 청크로 처리
    return [
        {"contents": doc.page_content, "metadata": doc.metadata} for doc in documents
    ]


def remove_long_urls(data):
    for item in data:
        if "contents" in item:
            item["contents"] = remove_urls(item["contents"])
    return data


def remove_urls(text):
    import re

    return re.sub(r"http\S+", "", text)


def save_as_excel(data, filename):
    data = remove_long_urls(data)
    df = pd.DataFrame([{"contents": item["contents"]} for item in data])
    df.to_excel(filename, index=False)


def process_pdfs_in_directory(directory_path, output_directory):
    count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            documents, pdf_document = load_pdf(file_path)
            chunks = split_documents(documents)
            all_chunks = [
                {"contents": chunk["contents"], "metadata": chunk["metadata"]}
                for chunk in chunks
            ]
            output_excel_file = os.path.join(
                output_directory, f"{os.path.splitext(filename)[0]}.xlsx"
            )
            save_as_excel(all_chunks, output_excel_file)
            if count % 100 == 0:
                print(f"Processed {count} files")
            count += 1


if __name__ == "__main__":
    process_pdfs_in_directory(directory_path, output_directory)
