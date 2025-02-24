import os
import json
import torch
import bitsandbytes as bnb
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Load API key from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if not hf_token:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACE_API_KEY in .env")

# Load Q&A dataset
def load_qa_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"question": d["question"], "context": d["context"], "answer": d["answer"]} for d in data]

qa_data = load_qa_dataset("qa_dataset.json")

# Convert to HF dataset format
dataset = load_dataset("json", data_files="qa_dataset.json", split="train")

# Model & Tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to mistralai/Mistral-7B if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure QLoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        [f"Question: {q}\nContext: {c}\nAnswer: {a}" for q, c, a in zip(examples["question"], examples["context"], examples["answer"])],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-llm",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    fp16=True,
    optim="adamw_bnb_8bit",
    save_total_limit=2,
    report_to="none"
)

# Train model
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer
)

trainer.train()

# Save fine-tuned model
trainer.model.save_pretrained("./finetuned-llm")
tokenizer.save_pretrained("./finetuned-llm")

print("Fine-tuning completed! 🚀")

