# HPC-Llama: Fine-Tuned LLaMA-3 for High-Performance Code Generation

## Overview
**HPC-Llama** is an expert code generation model built on **unsloth/Llama-3.2-8B-Instruct** and fine-tuned with **QLoRA** on HPC-oriented and general programming datasets (HPC-Instruct, Evol-Instruct, Magicoder). The model produces structured reasoning and optimized solutions for MPI, CUDA, OpenMP, and related high-performance workloads.

---

## Features
- **Parameter-efficient training:** QLoRA (4-bit LoRA, `r=16`) via Unsloth.
- **Structured outputs:** Responses are emitted in a predictable XML-style template:
  ```xml
  <Explanation>...</Explanation>
  <Code Solution>...</Code Solution>
  ```
- **End-to-end tooling:** Training, evaluation, retrieval, and FastAPI inference scripts.
- **RAG-ready:** Integrates with an HPC documentation corpus to provide grounded answers.

---

## Project Structure
```text
├── train_hpc_llama.py        # QLoRA fine-tuning pipeline on HPC + code data
├── evaluate_hpc.py           # Evaluation script for Pass@k and ComputeEval
├── retrieval.py              # RAG logic over HPC documentation
├── app.py                    # FastAPI service for structured code QA
├── llama3-8b-hpc-mix/        # Merged fine-tuned model (safetensors)
├── outputs_hpc_mix_8b/       # Training logs and checkpoints
├── data/                     # HPC documentation corpus and vector index
└── README.md
```

---

## Requirements
- Python 3.10+
- CUDA-enabled GPU (T4 or higher recommended)
- Poetry/uv or pip environment capable of installing Unsloth and Transformers stacks

---

## Installation
Install dependencies and apply Unsloth patches directly from the training script:

```bash
python train_hpc_llama.py --install-deps-only
```

---

## Evaluation
Functional correctness and latency are measured with both general and HPC-specific benchmarks:

- **Pass@k** for multi-sample code generation accuracy.
- **ComputeEval** for CUDA/GPU parallelism validation.
- **Latency and formatting metrics** captured per evaluation run.

Run the evaluation suite:

```bash
python evaluate_hpc.py
```

Artifacts (logs, metrics, generated samples) are saved under `outputs_hpc_mix_8b/benchmarks/`.

---

## Inference and Deployment
- The merged model is served through **vLLM** for high-throughput inference:
  ```bash
  vllm serve llama3-8b-hpc-mix --port 8080
  ```
- **FastAPI** (`app.py`) exposes `/code/generate`, `/code/rag`, and `/health` endpoints with Pydantic validation.
- RAG mode augments prompts with retrieved HPC documentation to improve grounding.

---

## Example API Call (RAG Endpoint)
```bash
curl -X POST http://localhost:8000/code/rag \
     -H "Content-Type: application/json" \
     -d '{
           "question": "How do I use shared memory in a CUDA kernel for matrix multiplication?",
           "language": "CUDA C"
         }'
```

---

## License
This project is released for research and educational purposes. Ensure compliance with the licenses for Llama-3.2-8B-Instruct, Unsloth, and all datasets incorporated into the fine-tuning pipeline.
