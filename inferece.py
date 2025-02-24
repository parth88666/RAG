from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./finetuned-llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

prompt = "What is the latest news about AI startups?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

output = model.generate(input_ids, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))

