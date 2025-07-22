from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
import torch
import re

import os
from dotenv import load_dotenv
load_dotenv()   # os.getenv("HUGGINGFACE_READ_TOKEN")

from datasets import load_dataset
Topicwise_dataset = load_dataset("BaekSeungJu/EQA-Ophthal-Dataset", split="train")
# print(Topicwise_dataset[0])

import pandas as pd

def Create_hf_4bit_model_chain(tokenizer_name,model_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=os.getenv("HF_TOKEN_read"), use_fast=False)
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
        model_name,
        quantization_config=bnb_config,
        token=os.getenv("HF_TOKEN_read"),
    )

    pipe = pipeline("text-generation", model=hf_model, tokenizer=tokenizer, max_new_tokens=512)
    model = HuggingFacePipeline(pipeline=pipe)
    return prompt | model | StrOutputParser()

chain = Create_hf_4bit_model_chain("meta-llama/Llama-3.1-8B-Instruct", "BaekSeungJu/Ophtimus-Llama-8B")

results = []  # 결과 저장용 리스트

count = 1
for Data_ in Topicwise_dataset:
    print(f"Processing data {count} of {len(Topicwise_dataset)}")
    question = Data_["Question"]
    answer = Data_["Answer"]

    responce_ = chain.invoke({"instruction": question})

    collected_answer = re.search(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(<\|eot_id\|>|$)",
    responce_, re.DOTALL).group(1)

    # 결과를 리스트에 저장
    results.append({
        "question": question,
        "answer": answer,
        "responce": responce_,
        "collected_answer": collected_answer
    })
    count += 1

results_df = pd.DataFrame(results)
results_df.to_excel("EQA-Ophthal-Results.xlsx", index=False)