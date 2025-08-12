import os
import json
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import pandas as pd
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

huggingface_token = os.getenv("HF_TOKEN_read")
login(token=huggingface_token)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = AutoModelForCausalLM.from_pretrained(
    "BaekSeungJu/Ophtimus-8B-Base",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("BaekSeungJu/Ophtimus-8B-Base")
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_247|>"})
else:
    tokenizer.pad_token = "<|reserved_special_token_247|>"
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<think>", "</think>"]
})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

EOS_TOKEN = tokenizer.eos_token

def reasoning_prompt(system_prompt, question, response):
    alpaca_format_str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}"""

    return alpaca_format_str

system_prompt_text = (
    "You are an expert ophthalmologist. Please provide accurate and medically sound answers "
    "to the user's ophthalmology-related question."
)

# batched=True에 맞춘 포맷팅 함수
def prompt_formatting_func(batch):
    qs = batch["question"]
    rs = batch["chain_response"]
    texts = [
        reasoning_prompt(system_prompt_text, q, r) + EOS_TOKEN
        for q, r in zip(qs, rs)
    ]
    return {"text": texts}

raw = load_dataset("BaekSeungJu/Ophthal-Reasoning-Dataset", split="train").shuffle(seed=42)

train_dataset = raw.map(prompt_formatting_func, batched=True, desc="format-train")

print(f"Train: {len(train_dataset)}")
print("Sample:\n", train_dataset[0]["text"])
print("--------------------------------")

# 7. 출력 디렉토리 생성
local_output_dir = "./Ophtimus_8B_Reasoning_checkpoint"
os.makedirs(local_output_dir, exist_ok=True)

training_arguments = SFTConfig(
    output_dir=local_output_dir,
    report_to="tensorboard",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    num_train_epochs=5,
    eval_steps=25,
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
    dataset_text_field="text",
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    peft_config=peft_config,
    args=training_arguments,
)

# 10. 학습 수행
train_stats = trainer.train()

# 11. 학습 로그 저장 (CSV)
import pandas as pd
lossPD = pd.DataFrame(trainer.state.log_history)
lossPD.to_csv(f"./{local_output_dir}/loss.csv", index=False)