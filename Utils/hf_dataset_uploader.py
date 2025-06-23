import pandas as pd
from datasets import Dataset
from huggingface_hub import login, HfApi
import os
from dotenv import load_dotenv

load_dotenv()
huggingface_token = os.getenv("HF_TOKEN_write")
login(token=huggingface_token)

# 현재 파일의 디렉토리로 작업 경로 변경
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

df = pd.read_excel("./Test_ex.xlsx")

hf_dataset = Dataset.from_pandas(df)

repo_id = "baeekseung/test_dataset"
hf_dataset.push_to_hub(repo_id)

print(f"✅ 데이터셋이 업로드되었습니다: https://huggingface.co/datasets/{repo_id}")
