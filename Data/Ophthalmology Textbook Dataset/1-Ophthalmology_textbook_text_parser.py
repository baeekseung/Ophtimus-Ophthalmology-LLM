import os
import pandas as pd
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

BCSC_num = 13
start_page = 23
end_page = 450

directory_path = f"./Ophthalmology TextBook/wills-eye-manual.pdf"
excel_file = f"./Data/wills-eye-manual.xlsx"

def load_pdf(directory_path):
    loader = PyMuPDFLoader(directory_path)
    return loader.load(), fitz.open(directory_path)

def extract_text_from_pages(
    pdf_document, start_page, end_page, top_margin=0.1, bottom_margin=0.9
):
    all_chunks = []
    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        page_height = page.rect.height
        page_width = page.rect.width

        rect = fitz.Rect(
            0, page_height * top_margin, page_width, page_height * bottom_margin
        )
        extracted_text = page.get_text("text", clip=rect)

        metadata = {"page": page_num + 1}
        all_chunks.append({"contents": extracted_text, "metadata": metadata})
    return all_chunks

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4076, chunk_overlap=512)
    return text_splitter.split_documents(documents)

def convert_parquet_to_excel(parquet_file, excel_file):
    df = pd.read_parquet(parquet_file)
    df.to_excel(excel_file, index=False)
    print(f"Parquet file converted to Excel and saved as {excel_file}")

def main():
    documents, pdf_document = load_pdf(directory_path)
    all_chunks = extract_text_from_pages(pdf_document, start_page, end_page)
    chunks = split_documents(documents)
    all_chunks.extend(
        {"contents": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    )
    df = pd.DataFrame(all_chunks)
    df.to_excel(excel_file, index=False)
    print(f"Excel 파일로 저장되었습니다: {excel_file}")

if __name__ == "__main__":
    main()
