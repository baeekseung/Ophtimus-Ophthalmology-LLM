import pandas as pd
import json
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv("HF_TOKEN_write")


# 현재 파일의 디렉토리로 작업 경로 변경
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


# Hugging Face 로그인
def login_to_huggingface(token):
    HfFolder.save_token(token)
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")


login_to_huggingface(huggingface_token)

dataPath = "./Refined Ophthalmology Textbooks/Ophthalmology Textbook Pretraining Dataset.xlsx"  # 엑셀 파일 경로

exel_row_name = ["text"]
hf_row_name = ["text"]


# 엑셀 파일을 데이터프레임으로 변환하는 함수
def excel_to_dataframe(excel_path):
    df = pd.read_excel(excel_path)
    print(f"Loaded data from {excel_path}")
    return df


# 데이터프레임을 JSON 데이터로 변환하는 함수
def dataframe_to_json(df):
    json_data = df.fillna("").to_dict(orient="records")
    print("Converted DataFrame to JSON data")
    return json_data


# 엑셀 파일을 데이터프레임으로 변환
df = excel_to_dataframe(dataPath)

# 데이터프레임을 JSON 데이터로 변환
json_data = dataframe_to_json(df)


# JSON 데이터를 Dataset으로 변환하고 Hugging Face에 업로드
def upload_to_huggingface(json_data, dataset_name, private=False):
    # 데이터를 딕셔너리 형태로 변환 (모든 값을 문자열로 변환)
    dataset_dict = {
        hf_row_name[0]: [str(entry[exel_row_name[0]]) for entry in json_data]
    }

    # Dataset 객체로 변환
    dataset = Dataset.from_dict(dataset_dict)
    dataset.push_to_hub(dataset_name, private=private)
    print(f"Uploaded data to Hugging Face as {dataset_name}")


# BaekSeungJu/Ophthalmic_sLM_FT_Dataset
dataset_name = "BaekSeungJu/Ophthalmology_Textbook-Corpus"  # Hugging Face에 업로드할 데이터셋 이름을 입력
upload_to_huggingface(json_data, dataset_name)
