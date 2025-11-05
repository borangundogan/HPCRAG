# -*- coding: utf-8 -*-
"""
Fine-tuning LLaMA-3-8B-Instruct on HPC + Code datasets
T4-SAFE | SINGLE GPU | ZERO OOM | FULL DATA
Author: Boran Gündoğan
"""

import subprocess, sys, os, torch, re
from datasets import load_dataset, concatenate_datasets

# ---------------- Dependency Management ---------------- #
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

def install_git(repo):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "--no-cache-dir", "--no-deps", repo
    ])

def ensure_dependencies():
    pkgs = [
        "torch>=2.3.0", "transformers>=4.43.0", "datasets",
        "accelerate", "bitsandbytes", "trl>=0.9.4", "unsloth_zoo" 
    ]
    for p in pkgs:
        try:
            __import__(p.split(">=")[0])
        except ImportError:
            install(p)
    install_git("git+https://github.com/unslothai/unsloth.git")

ensure_dependencies()
print("Dependencies installed. Restart runtime if first run.")

# ---------------- Unsloth Import (MUST BE FIRST) ---------------- #
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---------------- Configuration ---------------- #
MODEL_NAME = "unsloth/Llama-3-8B-Instruct"
MAX_SEQ_LEN = 2048
SUBSAMPLE = 8_000

PROMPT_TEMPLATE = """You are an expert HPC (High-Performance Computing) coding assistant with deep knowledge of parallel programming, GPU acceleration, distributed systems, and scientific computing. You provide accurate, efficient, and production-ready code solutions with clear explanations.

<Problem>
{prompt}
</Problem>

<Solution>
{completion}
</Solution>"""
EOS_TOKEN = "<|eot_id|>"

# ---------------- Model (4-bit QLoRA) ---------------- #
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")

# ---------------- Dataset Loading ---------------- #
print("Loading datasets...")
ds_hpc   = load_dataset("hpcgroup/hpc-instruct", split="train")
ds_evol  = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train").shuffle(seed=42).select(range(SUBSAMPLE))
ds_magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train").shuffle(seed=42).select(range(SUBSAMPLE))

combined = concatenate_datasets([ds_hpc, ds_evol, ds_magic])
print(f"Raw combined size: {len(combined):,} examples")

# ---------------- Robust Formatting ---------------- #
def format_combined(examples):
    texts = []
    batch_size = len(next(iter(examples.values())))
    for i in range(batch_size):
        prompt = (
            examples.get("problem_statement", [None]*batch_size)[i] or
            examples.get("problem", [None]*batch_size)[i] or
            examples.get("instruction", [None]*batch_size)[i] or ""
        ).strip()
        completion = (
            examples.get("solution", [None]*batch_size)[i] or
            examples.get("output", [None]*batch_size)[i] or ""
        ).strip()
        if prompt and completion:
            completion = re.sub(r"```(?:c|cpp|python)?", "```", completion)
            texts.append(PROMPT_TEMPLATE.format(prompt=prompt, completion=completion) + EOS_TOKEN)
    return {"text": texts}

print("Formatting all data...")
formatted = combined.map(
    format_combined,
    batched=True,
    remove_columns=combined.column_names,
    num_proc=1,
)
print(f"Formatted: {len(formatted):,} examples")

# ---------------- Train/Valid Split ---------------- #
split = formatted.train_test_split(test_size=0.02, seed=42)
train_ds, valid_ds = split["train"], split["test"]
print(f"Train: {len(train_ds):,}, Valid: {len(valid_ds):,}")

# ---------------- Trainer (T4 SINGLE GPU SAFE) ---------------- #
from transformers import TrainingArguments
from trl import SFTTrainer

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    max_steps=400,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="hpc_t4_safe",
    report_to="none",
    dataloader_num_workers=1,
    torch_compile=False,
    disable_tqdm=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=args,
    packing=True,  # Moved to SFTTrainer
)

trainer.train()
print("Training complete.")

# ---------------- Save ---------------- #
SAVE_DIR = "/content/llama3-8b-hpc-t4" if 'google.colab' in sys.modules else "llama3-8b-hpc-t4"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved: {SAVE_DIR}")

# ---------------- Inference Test ---------------- #
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

test_query = "Write a CUDA kernel that adds two float arrays element-wise."
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": test_query}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
print("\nSAMPLE OUTPUT:")
print(tokenizer.decode(output[0], skip_special_tokens=False))

# ---------------- Colab Download ---------------- #
if 'google.colab' in sys.modules:
    print("\nZipping model...")
    zip_path = "/content/hpc_t4_model.zip"
    subprocess.run(["zip", "-r", zip_path, SAVE_DIR], check=True)
    try:
        from google.colab import files
        files.download(zip_path)
    except:
        print("Download via Files panel.")