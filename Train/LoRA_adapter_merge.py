from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv("HF_TOKEN_read")

from huggingface_hub import login
login(token=huggingface_token)

from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

fine_tuned_model_path = f"../../Ophtimus_LoRA/Ophtimus_3B_Instruct_checkpoint/checkpoint-1584"

from peft import PeftConfig, PeftModel
base_and_adapter_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)

base_and_adapter_model = base_and_adapter_model.merge_and_unload()

save_path = f"../../Ophtimus/Ophtimus-3B-Instruct-v1/1epoch"  # 저장할 경로를 지정

import os
os.makedirs(save_path, exist_ok=True)

base_and_adapter_model.save_pretrained(save_path)
