# train.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

# ============================================================
# 配置区 —— 你只需要改这里
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"# 本地模型路径
DATA_PATH  = "data/train.jsonl"
OUTPUT_DIR = "output/qwen2.5-7b-lora"
MAX_SEQ_LEN = 1024        # 序列长度，先用 1024 跑通
NUM_EPOCHS  = 3
BATCH_SIZE  = 2            # 每卡 batch
GRAD_ACCUM  = 8            # 有效 batch = 2 * 8 = 16
LR = 2e-4

USE_QLORA = True          # <<<< 显存不够就改成 True
# ============================================================

# ---------- 1. Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- 2. 模型加载 ----------
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

model.config.use_cache = False          # 训练时关闭 KV cache
model.enable_input_require_grads()      # LoRA 必须

# ---------- 3. LoRA 配置 ----------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # 低秩维度
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.05,
    target_modules=[        # 要加 LoRA 的层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 你会看到类似：trainable params: 20,971,520 || all params: 7,636,xxx,xxx || trainable%: 0.27%

# ---------- 4. 数据集 ----------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ---------- 5. 训练参数 ----------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    optim="adamw_8bit",
    max_length=MAX_SEQ_LEN,
    disable_tqdm=True,
)

# ---------- 6. Trainer ----------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ---------- 7. 开始训练 ----------
trainer.train()

# ---------- 8. 保存 LoRA adapter ----------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA adapter 已保存到 {OUTPUT_DIR}")
