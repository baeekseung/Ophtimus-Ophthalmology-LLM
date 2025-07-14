import os
import json
from dotenv import load_dotenv
import torch

# 1. 환경 변수 로드 및 Huggingface 로그인
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN_read")
from huggingface_hub import login
login(token=huggingface_token)

# 2. 4-bit 양자화 설정
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# 3. PEFT LoRA 설정
from peft import LoraConfig
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 4. 모델 및 토크나이저 로드
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
model.config.pad_token_id = tokenizer.pad_token_id  # 예: 128004

# 5. 데이터셋 로드 및 전처리
from datasets import load_dataset, concatenate_datasets

# EQA 데이터셋 전처리
EQA_dataset = load_dataset("BaekSeungJu/Ophthalmology-EQA-v3", split="train")
EQA_dataset = EQA_dataset.shuffle(seed=42)
def EQA_format_chat_template(row):
    system_instruction = "You are an expert ophthalmologist. Please provide accurate and medically answers to the user's ophthalmology-related question."
    row_json = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row
mapped_EQA_dataset = EQA_dataset.map(EQA_format_chat_template, num_proc=4)
split_EQA_dataset = mapped_EQA_dataset.train_test_split(test_size=0.01, seed=42)
train_EQA_dataset = split_EQA_dataset['train']
test_EQA_dataset = split_EQA_dataset['test']
print(f"Train EQA dataset size: {len(train_EQA_dataset)}")
print(f"Test EQA dataset size: {len(test_EQA_dataset)}")

# MCQA 데이터셋 전처리
MCQA_dataset = load_dataset("BaekSeungJu/Ophthalmology-MCQA-v3", split="train")
MCQA_dataset = MCQA_dataset.shuffle(seed=42)

MCQA_dataset2 = load_dataset("BaekSeungJu/Ophthal_MCQA_v2", split="train")
MCQA_dataset2 = MCQA_dataset.shuffle(seed=42)

def MCQA_format_chat_template(row):
    system_instruction = (
        "You are an expert ophthalmologist. Please provide accurate and "
        "medically sound answers to the user's ophthalmology-related question."
    )
    # 헬퍼 함수들
    def MCQA_Alphabet_Selection_func(answer, option_a, option_b, option_c, option_d, option_e):
        if answer == option_a:
            return "A"
        elif answer == option_b:
            return "B"
        elif answer == option_c:
            return "C"
        elif answer == option_d:
            return "D"
        elif answer == option_e:
            return "E"
        return ""
    def MCQA_Instruction_formatting_func(question, option_a, option_b, option_c, option_d, option_e):
        if option_c == "":
            return f"Question:\n{question}\n\nOptions:\nA) {option_a}\nB) {option_b}"
        elif option_e == "":
            return f"Question:\n{question}\n\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}"
        else:
            return f"Question:\n{question}\n\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nE) {option_e}"
    def MCQA_Response_formatting_func(option_a, option_b, option_c, option_d, option_e, explanation, answer):
        letter = MCQA_Alphabet_Selection_func(answer, option_a, option_b, option_c, option_d, option_e)
        return f"Explanation:\n{explanation}\n\nAnswer:\n{letter}) {answer}"
    row_json = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": MCQA_Instruction_formatting_func(
            row["question"], row["option_a"], row["option_b"], row["option_c"], row["option_d"], row["option_e"]
        )},
        {"role": "assistant", "content": MCQA_Response_formatting_func(
            row["option_a"], row["option_b"], row["option_c"], row["option_d"], row["option_e"],
            row["explanation"], row["answer"]
        )}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


mapped_MCQA_dataset = MCQA_dataset.map(MCQA_format_chat_template, num_proc=4)
split_MCQA_dataset = mapped_MCQA_dataset.train_test_split(test_size=0.01, seed=42)
train_MCQA_dataset = split_MCQA_dataset['train']
test_MCQA_dataset = split_MCQA_dataset['test']
print(f"Train MCQA dataset size: {len(train_MCQA_dataset)}")
print(f"Test MCQA dataset size: {len(test_MCQA_dataset)}")

mapped_MCQA_dataset2 = MCQA_dataset2.map(MCQA_format_chat_template, num_proc=4)
split_MCQA_dataset2 = mapped_MCQA_dataset2.train_test_split(test_size=0.01, seed=42)
train_MCQA_dataset2 = split_MCQA_dataset2['train']
test_MCQA_dataset2 = split_MCQA_dataset2['test']
print(f"Train MCQA dataset2 size: {len(train_MCQA_dataset2)}")
print(f"Test MCQA dataset2 size: {len(test_MCQA_dataset2)}")

# 전체 학습/평가 데이터셋 결합 및 셔플링
train_dataset = concatenate_datasets([train_EQA_dataset, train_MCQA_dataset, train_MCQA_dataset2])
test_dataset = concatenate_datasets([test_EQA_dataset, test_MCQA_dataset, test_MCQA_dataset2])
train_dataset = train_dataset.shuffle(seed=42)
print(f"Combined Train dataset size: {len(train_dataset)}")
print(f"Combined Test dataset size: {len(test_dataset)}")

# 예시로 두 번째 샘플 출력
print(train_dataset[1]["text"])

# 6. Data Collator 설정
from trl import DataCollatorForCompletionOnlyLM
data_collator_param = {}
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
data_collator_param["data_collator"] = collator

# 7. 출력 디렉토리 생성
local_output_dir = "../../Ophtimus_LoRA/Ophtimus_8B_Instruct_checkpoint"
os.makedirs(local_output_dir, exist_ok=True)

# 8. DeepSpeed 설정 파일 생성 (ds_config.json)
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 8,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True}
    },
    "optimizer": {"type": "AdamW", "params": {"lr": 2e-4, "weight_decay": 0.01}},
    "scheduler": {"type": "WarmupLR", "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-4, "warmup_num_steps": 10}},
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}
ds_config_file = "./ds_config.json"
if not os.path.exists(ds_config_file):
    with open(ds_config_file, "w") as f:
        json.dump(ds_config, f, indent=2)
print(f"DeepSpeed config saved to {ds_config_file}")

# 9. 학습 인자 설정 (DeepSpeed 옵션 추가)
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

training_arguments = SFTConfig(
    output_dir=local_output_dir,
    report_to="tensorboard",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    num_train_epochs=5,
    eval_steps=25,
    # save_steps=100,
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    max_seq_length=2048,
    lr_scheduler_type="constant_with_warmup",
    seed=42,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': True},
    deepspeed=ds_config_file  # DeepSpeed 설정 파일 경로 지정
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    args=training_arguments,
    **data_collator_param
)

# 10. 학습 수행
train_stats = trainer.train()

# 11. 학습 로그 저장 (CSV)
import pandas as pd
lossPD = pd.DataFrame(trainer.state.log_history)
lossPD.to_csv(f"./{local_output_dir}/loss.csv", index=False)
