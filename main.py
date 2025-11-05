from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "model/llama3-8b-hpc-merged"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,   
    device_map="auto"
)

# 3️⃣ Örnek HPC prompt
prompt = """Explain how OpenMP dynamic scheduling improves load balancing in a parallel loop."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
