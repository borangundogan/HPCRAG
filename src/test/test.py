# -*- coding: utf-8 -*-
"""
Test script for the fine-tuned HPC model (LLaMA-3-8B-HPC)
Author: Boran Gündoğan
"""

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def main():
    # 1. Load model and tokenizer
    MODEL_PATH = "model/llama3-8b-hpc-t4"   # adjust if your folder has a different name

    print(f"Loading model from: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=4096,
        dtype=None,           # automatically selects fp16 / bf16 if GPU supports
        load_in_4bit=False,   # use merged weights (16-bit)
        device_map="auto",
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # 2. Example HPC / code question
    query = """Write an OpenMP C program that performs parallel matrix multiplication
with dynamic scheduling, and explain how it balances workload among threads."""

    sys_prompt = f"""You are an expert HPC (High-Performance Computing) assistant.
You provide efficient, scalable, and well-commented code with short explanations.

<Question>
{query}
</Question>
"""

    # 3. Tokenize & Generate
    messages = [{"role": "user", "content": sys_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    print("Generating response...\n")
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,         # lower = more deterministic output
        top_p=0.9,
        repetition_penalty=1.1,
        use_cache=True,
    )

    # ------------------------------------------------------------------
    # 4. Decode and display
    # ------------------------------------------------------------------
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("========== HPC Model Response ==========")
    print(response)
    print("========================================")

if __name__ == "__main__":
    main()
