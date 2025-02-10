from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct',cache_dir='/home/jovyan/shared/yilunz/z1')

print(model)
