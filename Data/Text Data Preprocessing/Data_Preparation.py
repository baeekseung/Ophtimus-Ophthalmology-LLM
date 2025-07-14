#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

import os
import re
from dotenv import load_dotenv
from datasets import load_dataset

# fastText 모델을 위한 임포트
from fasttext.FastText import _FastText

def load_env_and_login():
    """.env 파일에서 Hugging Face 토큰을 불러와 로그인합니다."""
    load_dotenv()
    HF_TOKEN_read = os.getenv("HF_TOKEN_read")
    if not HF_TOKEN_read:
        raise ValueError("HF_TOKEN_read가 .env 파일에 설정되어 있지 않습니다.")
    from huggingface_hub import login
    login(token=HF_TOKEN_read)

def load_and_prepare_dataset():
    """
    Hugging Face Hub에서 Ophthalmology PubMed Corpus를 불러오고,
    'text' 컬럼만 선택하여 dataset 객체를 반환합니다.
    """
    pretraining_dataset = load_dataset("BaekSeungJu/Ophthalmology-PubMed-Corpus")
    print("데이터셋 로드 완료:")
    print(pretraining_dataset)

    # dataset이 DatasetDict인지 단일 Dataset인지에 따라 처리
    if isinstance(pretraining_dataset, dict):
        dataset = pretraining_dataset["train"].select_columns(["text"])
    else:
        dataset = pretraining_dataset.select_columns(["text"])

    # 샘플 텍스트 출력
    sample = dataset[9]["text"]
    print("\n샘플 텍스트 (인덱스 9):")
    print(sample)

    return dataset

def find_duplicates(paragraphs):
    """
    단락 리스트 내 중복 요소(문장) 수와 중복된 문자 수를 반환합니다.
    """
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in paragraphs:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars

def paragraph_repetition_filter(x):
    """
    한 텍스트 내에서 중복 단락 비율이 너무 높은 경우 False를 반환합니다.
    """
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())
    if not paragraphs:
        return False
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False
    if len(text) > 0 and (char_duplicates / len(text)) > 0.2:
        return False
    return True

def deduplication(ds):
    """
    동일한 텍스트가 여러 번 포함된 예제를 제거합니다.
    """
    unique_text = set()
    def dedup_func(x):
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True
    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    return ds

def english_language_filter(ds):
    """
    fastText를 이용하여 영어 텍스트만 남기도록 필터링합니다.
    fastText 모델 파일은 './models/L2_language_model.bin'에 있어야 합니다.
    """
    model_path = './models/L2_language_model.bin'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"fastText 언어 모델 파일이 {model_path}에 존재하지 않습니다.")
    model = _FastText(model_path)

    def is_english(x):
        language, score = model.predict(x['text'].replace("\n", ""))
        # fastText 결과의 형식이 '__label__en' 형태라고 가정
        language = language[0].split("__")[-1]
        return score > 0.4 and language == "en"

    ds = ds.filter(is_english, load_from_cache_file=False, num_proc=1)
    return ds

def save_dataset(dataset):
    """
    전처리된 데이터셋을 parquet 파일로 저장합니다.
    """
    directory = "./Pre-Training-Dataset"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, "Preprocessed_pretrain_Dataset.parquet")
    # Dataset 객체가 DatasetDict라면 "train" split 저장, 그렇지 않으면 그대로 저장합니다.
    if isinstance(dataset, dict):
        dataset["train"].to_parquet(file_path)
    else:
        dataset.to_parquet(file_path)
    print(f"\n전처리된 데이터셋이 {file_path}에 저장되었습니다.")

def main():
    load_env_and_login()

    dataset = load_and_prepare_dataset()

    # 초기 데이터셋 행 수 출력
    try:
        initial_rows = dataset.num_rows
    except AttributeError:
        initial_rows = len(dataset)
    print(f"\n초기 행 수: {initial_rows}")

    # 1. 중복 단락 필터 적용
    dataset = dataset.filter(paragraph_repetition_filter, load_from_cache_file=False)
    try:
        filtered_rows = dataset.num_rows
    except AttributeError:
        filtered_rows = len(dataset)
    print(f"중복 단락 필터 후 행 수: {filtered_rows}")

    # 2. 중복 예제 제거
    dataset = deduplication(dataset)
    try:
        dedup_rows = dataset.num_rows
    except AttributeError:
        dedup_rows = len(dataset)
    print(f"중복 제거 후 행 수: {dedup_rows}")

    # 3. 영어 텍스트 필터 적용
    dataset = english_language_filter(dataset)
    try:
        english_rows = dataset.num_rows
    except AttributeError:
        english_rows = len(dataset)
    print(f"영어 텍스트 필터 후 행 수: {english_rows}")

    # 최종 데이터셋 정보 출력
    print("\n최종 데이터셋 정보:")
    print(dataset)

    # 전처리된 데이터셋 저장
    save_dataset(dataset)

if __name__ == "__main__":
    main()
