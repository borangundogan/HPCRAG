# -*- coding: utf-8 -*-
"""
Test script for the fine-tuned HPC model (LLaMA-3-8B-HPC)
Author: Boran Gündoğan
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. Load model and tokenizer
    MODEL_PATH = "model/content/merged_model/"   # adjust if folder name differs

    print(f"Loading model from: {MODEL_PATH}")
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Running on device: {device}")

    # Load model (no quantization needed for CPU/MPS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",      # MPS backend daha kararlı çalışır
        trust_remote_code=True,           # Unsloth patch’leri varsa gereklidir
        load_in_4bit=False                # quantization’ı zorla kapat
    )

    # 2. Example HPC / code question
    query = """Write an OpenMP C program that performs parallel matrix multiplication
with dynamic scheduling, and explain how it balances workload among threads."""

    sys_prompt = (
        "You are an expert HPC (High-Performance Computing) assistant.\n"
        "You provide efficient, scalable, and well-commented code with short explanations.\n\n"
        f"<Question>\n{query}\n</Question>\n"
    )

    # 3. Tokenize & generate
    inputs = tokenizer(sys_prompt, return_tensors="pt").to(device)
    print("Generating response...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # 4. Decode and display
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("========== HPC Model Response ==========")
    print(response)
    print("========================================")

if __name__ == "__main__":
    main()
