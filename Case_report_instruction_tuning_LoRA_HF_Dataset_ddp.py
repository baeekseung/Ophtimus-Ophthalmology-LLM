from huggingface_hub import login
import os
import pandas as pd
import torch
import logging
import sys
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from peft import LoraConfig
from accelerate import PartialState

os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ----------------------
# 0) Distributed training settings
# ----------------------
device_string = PartialState().process_index
# ----------------------
# 1) 모델/토크나이저 로드
# ----------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
    device_map={'':device_string},
    torch_dtype=torch.float32,
    # max_memory={5:"80GiB", 6: "80GiB", 7: "80GiB", 8: "80GiB", 9: "80GiB"},
    use_cache=False  # 수정: gradient checkpointing과 충돌 방지
)
print(f"Model '{model_name}' loaded successfully.", flush=True)

# Special token 설정
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_247|>"})
model.config.pad_token_id = tokenizer.pad_token_id
EOS_TOKEN = tokenizer.eos_token

# ----------------------
# 2) 데이터 로드 및 전처리
# ----------------------

def format_case_report(row):
    row["instruction"] = (
        "Based on the provided patient information and symptoms, diagnose the condition and suggest the appropriate treatment method.\n\n"
        f"Patient Information: {row['Patient Information']}"
    )
    row["response"] = f"Diagnosis and Treatment: {row['Diagnosis and Treatment']}"
    return row

def format_eqa(row):
    row["instruction"] = f"Answer the following ophthalmology-related question:\n\nQuestion: {row['question']}"
    row["response"] = f"Answer: {row['answer']}"
    return row

def MCQA_format_chat_template(row):
    def MCQA_Alphabet_Selection_func(answer, option_a, option_b, option_c, option_d, option_e):
        options = [option_a, option_b, option_c, option_d, option_e]
        if answer not in options:
            print(f"[Warning] Answer '{answer}' not found in options: {options}", flush=True)
            return "N/A"
        return "ABCDE"[options.index(answer)]

    row["instruction"] = (
        f"Question:\n{row['question']}\n\nOptions:\n"
        f"A) {row['option_a']}\nB) {row['option_b']}\nC) {row['option_c']}\nD) {row['option_d']}\nE) {row['option_e']}"
    )
    row["response"] = (
        f"Explanation:\n{row['explanation']}\n\n"
        f"Correct Answer: ({MCQA_Alphabet_Selection_func(row['answer'], row['option_a'], row['option_b'], row['option_c'], row['option_d'], row['option_e'])}) {row['answer']}"
    )
    return row

# 데이터셋 로드
dataset_train = load_dataset("MinWook1125/Ophthalmology-Case-Report-Train", split="train").shuffle(seed=42)
dataset_test = load_dataset("MinWook1125/Ophthalmology-Case-Report-Test", split="train").shuffle(seed=42)
dataset_eqa = load_dataset("MinWook1125/Ophthalmology-EQA-v3", split="train").train_test_split(test_size=0.15, seed=42)
dataset_mcqa = load_dataset("MinWook1125/Ophthalmology-MCQA-v3", split="train").map(MCQA_format_chat_template, num_proc=4).train_test_split(test_size=0.15, seed=42)

# 데이터 전처리
dataset_train = dataset_train.map(format_case_report, num_proc=4)
dataset_test = dataset_test.map(format_case_report, num_proc=4)
dataset_eqa_train = dataset_eqa["train"].map(format_eqa, num_proc=4)
dataset_eqa_test = dataset_eqa["test"].map(format_eqa, num_proc=4)
dataset_mcqa_train = dataset_mcqa["train"]
dataset_mcqa_test = dataset_mcqa["test"]

# ----------------------
# 3) 데이터셋 통합 및 변환
# ----------------------

def convert_to_alpaca_format(instruction, response):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n\n"
        "You are an expert ophthalmologist. Please provide accurate and medically answers to the user's ophthalmology-related question.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"### Response:\n{response}<|eot_id|>"
    )

train_dataset = concatenate_datasets([dataset_train, dataset_eqa_train, dataset_mcqa_train]).map(
    lambda x: {"text": convert_to_alpaca_format(x["instruction"], x["response"])}, num_proc=4
)

test_dataset = concatenate_datasets([dataset_test, dataset_eqa_test, dataset_mcqa_test]).map(
    lambda x: {"text": convert_to_alpaca_format(x["instruction"], x["response"])}, num_proc=4
)

# ----------------------
# 4) 학습 설정 및 실행
# ----------------------
collator = DataCollatorForCompletionOnlyLM(response_template="### Response:\n", tokenizer=tokenizer, mlm=False)

peft_config = LoraConfig(lora_alpha=16, lora_dropout=0, r=16, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"])

local_output_dir = "./LLM/Case_report_Ophthal_Llama3_1_instruct_LoRA_batch_32_MCQA_EQA_CR"

training_arguments = SFTConfig(
        output_dir=local_output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        num_train_epochs=100,
        eval_steps=100,  # 평가 주기 증가
        save_steps=100,  # 저장 주기 증가
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        logging_steps=5,  # 로그 출력 주기 증가
        logging_dir="./logs",  # 로그 저장 디렉토리
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant_with_warmup",
        fp16=True,  # AMP 사용
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,
)

trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset,
        data_collator=collator,
        peft_config=peft_config,
        args=training_arguments,
)

print("Starting training...", flush=True)
trainer.train()
torch.cuda.synchronize()
print("Training completed!", flush=True)
