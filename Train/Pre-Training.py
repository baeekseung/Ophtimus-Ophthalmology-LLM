import sys
import torch
from transformers import AutoModelForCausalLM
import datasets
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import transformers
from transformers import Trainer, TrainerCallback
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN_read = os.getenv("HF_TOKEN_read")

from huggingface_hub import login
login(token=HF_TOKEN_read)

os.environ["TORCH_USE_CUDA_DSA"] = "1"

class CustomDataset(Dataset):
    def __init__(self, args, split="train"):
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet", data_files=args.dataset_name, split=split
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class CustomArguments(transformers.TrainingArguments):
    # 데이터셋 및 출력 경로 관련 인자
    dataset_name: str = field(
        default="./Pre-Training-Dataset/packaged_pretrain_Dataset.parquet"
    )
    max_seq_length: int = field(default=1024)
    output_dir: str = field(default="./training_output")

    # 핵심 학습 설정
    seed: int = field(default=0)
    optim: str = field(default="adamw_torch")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    save_strategy: str = field(default="epoch")

    # 추가 학습 설정
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0)
    warmup_steps: int = field(default=5)
    lr_scheduler_type: str = field(default="linear")
    gradient_checkpointing: bool = field(default=True)
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    dataloader_num_workers: int = field(default=2)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=4)

    # DeepSpeed 설정 (DeepSpeed config 파일 경로 지정)
    deepspeed: str = field(
        default="deepspeed_config.json", metadata={"help": "Path to DeepSpeed config file"}
    )

    # 로깅 설정
    logging_steps: int = field(default=50)
    report_to: str = field(default="none")


class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)


def main():
    parser = transformers.HfArgumentParser(CustomArguments)
    (args,) = parser.parse_args_into_dataclasses(sys.argv[1:])
    print(args)

    print("Is CUDA available?", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    train_dataset = CustomDataset(args=args)
    print("Input shape: ", train_dataset[0]["input_ids"].shape)

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    loss_logging_callback = LossLoggingCallback()

    trainer = Trainer(
        model=pretrained_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[loss_logging_callback],
    )

    trainer.train()


if __name__ == "__main__":
    main()