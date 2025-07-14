import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv
import os
import argparse

# Hugging Face 로그인
def login_to_huggingface(token):
    HfFolder.save_token(token)
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

# 엑셀 → DataFrame
def excel_to_dataframe(excel_path):
    df = pd.read_excel(excel_path)
    print(f"Loaded {excel_path} with columns: {list(df.columns)}")
    return df

# Hugging Face 업로드
def upload_to_huggingface(df, dataset_name, private=False):
    df = df.astype(str)  # 문자열 변환
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(dataset_name, private=private)
    print(f"Uploaded to Hugging Face as {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload Excel data to Hugging Face dataset.")
    parser.add_argument('--excel_path', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the Hugging Face dataset repo (e.g. username/repo)')
    args = parser.parse_args()

    # 환경변수 로드 및 로그인
    load_dotenv()
    token = os.getenv("HF_TOKEN_write")
    if not token:
        raise ValueError("환경변수 'HF_TOKEN_write'가 설정되어 있지 않습니다.")
    login_to_huggingface(token)

    # 실행
    df = excel_to_dataframe(args.excel_path)
    upload_to_huggingface(df, args.repo_name)

if __name__ == "__main__":
    main()

# python hf_upload.py --excel_path ./Test_ex.xlsx --repo_name BaekSeungJu/Test_dataset
