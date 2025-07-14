import openai
import pandas as pd
import os
import time
import threading
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

openai.api_key = ""

def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4096,
        n=1,
        stop=None,
    )
    return response['choices'][0]['message']['content']

def create_map_prompt(docs):
    return f"""The following is a partial content of the ophthalmology paper.
    {docs}
    Write a summary of the given content, focusing on information related to eye diseases and treatments. Do not include information about the authors and references. If the information is not relevant to eye diseases and treatments, don't include it in your summary.
    Answer:"""

def create_reduce_prompt(docs):
    return f"""The following is a Series of Ophthalmology Paper Summaries.
    {docs}
    Please generate a detailed summary of the following text in multiple paragraphs without using headings or subheadings, focusing on the content related to eye diseases and treatments. Please summarize in as much detail as possible.
    Answer:"""

def split_text(pdf_path):
    print(f"PDF name : {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Documents Num : {len(docs)}")
    return docs

def summarize_pdf(split_docs):
    intermediate_summaries = []
    for doc in split_docs:
        prompt = create_map_prompt(doc.page_content)
        summary = call_openai_api(prompt)
        intermediate_summaries.append(summary)

    combined_prompt = create_reduce_prompt(" ".join(intermediate_summaries))
    final_summary = call_openai_api(combined_prompt)
    return final_summary

def get_pdf_filenames(folder_path):
    all_files = os.listdir(folder_path)
    pdf_files = [os.path.splitext(file)[0] for file in all_files if file.endswith('.pdf')]
    return pdf_files

def summarize_with_timeout(docs, timeout=60):
    summary = [None]

    def target():
        summary[0] = summarize_pdf(docs)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Timeout reached, moving to next PDF.")
        thread.join()  # Clean up the thread
        return None
    else:
        return summary[0]

PDFList = get_pdf_filenames("./Dataset/Fold1")
a = len(PDFList)
c = 1
for pdf in PDFList:
    start_time = time.time()
    print(f"{c} / {a}")
    pdf_path = f"./Dataset/Fold1/{pdf}.pdf"
    docs = split_text(pdf_path)
    if len(docs) < 20:
        print("sufficient documents")
        summary = summarize_with_timeout(docs, timeout=180)
        if summary is None:
            c += 1
            print("\nFinish summary-----------------------------------------------------------------------------------------------")
            continue
        c += 1
    else:
        print("Too many documents")
        c += 1
        print("\nFinish summary-----------------------------------------------------------------------------------------------")
        continue

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame({"Summary": [summary]})

    # DataFrame을 Excel 파일로 저장
    excel_path = f"./Dataset/Summary/{pdf}_summary.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"Summary has been saved to {excel_path}")
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    print("\nFinish summary-----------------------------------------------------------------------------------------------")
