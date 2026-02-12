import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "out-gpt2-shakespeare"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "How to peel a banana? \n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
