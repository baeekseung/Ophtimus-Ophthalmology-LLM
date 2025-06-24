#!/usr/bin/env python
import os
from dotenv import load_dotenv
from huggingface_hub import login
import datasets
from transformers import AutoTokenizer
import numpy as np

def login_hf():
    """
    .env 파일에서 HF 토큰을 로드하고 Hugging Face에 로그인합니다.
    """
    load_dotenv()
    HF_TOKEN_read = os.getenv("HF_TOKEN_read")
    if not HF_TOKEN_read:
        raise ValueError("HF_TOKEN_read가 .env 파일에 설정되어 있지 않습니다.")
    login(token=HF_TOKEN_read)

def load_dataset_parquet(parquet_file="./Pre-Training-Dataset/Preprocessed_PT_Dataset.parquet"):
    """
    Parquet 형식의 전처리된 데이터셋을 불러옵니다.
    """
    dataset = datasets.load_dataset("parquet", data_files=parquet_file, split="train")
    print("불러온 데이터셋:")
    print(dataset)
    return dataset

def load_tokenizer(model_path="meta-llama/Llama-3.1-8B"):
    """
    주어진 모델 경로에서 토크나이저를 불러옵니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer

def tokenization(example, tokenizer):
    """
    입력 예제의 "text" 컬럼을 토크나이징하고, bos, eos 토큰을 추가합니다.
    또한 토큰 개수를 'num_tokens'로 기록합니다.
    """
    tokens = tokenizer.tokenize(example["text"])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
    example["input_ids"] = token_ids
    example["num_tokens"] = len(token_ids)
    return example

def process_dataset(dataset, tokenizer):
    """
    데이터셋에 토크나이징 함수를 적용합니다.
    """
    dataset = dataset.map(lambda x: tokenization(x, tokenizer), load_from_cache_file=False)
    print("토크나이징 후 데이터셋:")
    print(dataset)
    return dataset

def print_sample(dataset, index=3):
    """
    지정한 인덱스의 샘플 데이터를 출력합니다.
    """
    sample = dataset[index]
    print("샘플 텍스트:", sample["text"][:30])
    print("\n샘플 input_ids:", sample["input_ids"][:30])
    print("\n샘플 num_tokens:", sample["num_tokens"])

def pack_data(dataset, max_seq_length=4096):
    """
    데이터셋의 모든 input_ids를 연결하여 전체 시퀀스를 만들고,
    max_seq_length 단위로 자른 후 재구성합니다.
    """
    # 모든 input_ids를 연결
    all_input_ids = np.concatenate(dataset["input_ids"])
    print("연결된 input_ids 총 길이:", len(all_input_ids))
    
    # 전체 길이를 max_seq_length의 배수로 맞춤
    total_length = len(all_input_ids) - (len(all_input_ids) % max_seq_length)
    print("전체 길이 (max_seq_length 배수):", total_length)
    
    all_input_ids = all_input_ids[:total_length]
    print("자른 후 shape:", all_input_ids.shape)
    
    # 시퀀스 단위로 재구성 및 int32 형 변환
    input_ids_reshaped = all_input_ids.reshape(-1, max_seq_length).astype(np.int32)
    print("재구성된 input_ids shape:", input_ids_reshaped.shape)
    return input_ids_reshaped

def create_packaged_dataset(input_ids_reshaped):
    """
    재구성한 input_ids 배열을 Dataset 객체로 변환합니다.
    """
    input_ids_list = input_ids_reshaped.tolist()
    packaged_dataset = datasets.Dataset.from_dict({"input_ids": input_ids_list})
    print("패키징된 데이터셋:")
    print(packaged_dataset)
    return packaged_dataset

def save_dataset(dataset, directory="./Pre-Training-Dataset", filename="packaged_pretrain_Dataset.parquet"):
    """
    데이터셋을 parquet 파일로 지정한 경로에 저장합니다.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    dataset.to_parquet(file_path)
    print(f"데이터셋이 {file_path}에 저장되었습니다.")

def main():
    # 1. Hugging Face 로그인
    login_hf()
    
    # 2. 데이터셋 로드
    dataset = load_dataset_parquet()
    
    # 3. 토크나이저 불러오기
    tokenizer = load_tokenizer()
    
    # 4. 토크나이징 적용
    dataset = process_dataset(dataset, tokenizer)
    
    # 5. 샘플 데이터 출력
    print_sample(dataset)
    
    # (선택사항) 데이터셋 전체 토큰 수 계산
    total_tokens = np.sum(dataset["num_tokens"])
    print("전체 토큰 수:", total_tokens)
    
    # 6. 시퀀스 패킹
    input_ids_reshaped = pack_data(dataset, max_seq_length=4096)
    
    # 7. 재구성한 데이터셋 생성
    packaged_dataset = create_packaged_dataset(input_ids_reshaped)
    
    # 8. 패키징된 데이터셋 저장
    save_dataset(packaged_dataset)

if __name__ == "__main__":
    main()
