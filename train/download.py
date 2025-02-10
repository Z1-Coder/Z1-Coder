from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')

print(model)
