from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
import torch

import os
from dotenv import load_dotenv
load_dotenv()   # os.getenv("HUGGINGFACE_READ_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=os.getenv("HF_TOKEN_read"), use_fast=False)
chat = [
    {"role": "system", "content": "You are an expert ophthalmologist. Please provide accurate and medically sound answers to the user's ophthalmology-related question."},
    {"role": "user", "content": "{instruction}"}
]
prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
prompt = PromptTemplate.from_template(prompt_str)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

hf_model = AutoModelForCausalLM.from_pretrained(
    "BaekSeungJu/Ophtimus-Llama-8B",
    quantization_config=bnb_config,
    token=os.getenv("HF_TOKEN_read"),
)

pipe = pipeline("text-generation", model=hf_model, tokenizer=tokenizer, max_new_tokens=512)
model = HuggingFacePipeline(pipeline=pipe)
chain = prompt | model | StrOutputParser()

import pandas as pd
excel_path = "TopicwiseEval_Dataset.xlsx"
# 엑셀 파일에서 'Question' 열 읽기
df = pd.read_excel(excel_path)
questions = df["Question"].dropna().tolist()


