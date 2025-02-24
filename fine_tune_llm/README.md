# Fine-Tuning an LLM with QLoRA for Q&A

This project fine-tunes a **LLaMA-2-7B** or **Mistral-7B** model using **QLoRA** to generate answers from text summaries. The dataset consists of **multiple `.txt` files**, converted into a Q&A format before training.

## 📌 Features
✅ Converts `.txt` summaries into **structured Q&A pairs**  
✅ Fine-tunes **LLaMA-2/Mistral** using **QLoRA** (4-bit quantization)  
✅ Optimized training with **`bitsandbytes`**, **LoRA**, and **`transformers`**  
✅ Saves the fine-tuned model for **real-time inference**  

---

## 🚀 1. Installation
Run the following to install dependencies:

```bash
pip install transformers peft bitsandbytes datasets accelerate trl
```

---

## 📂 2. File Structure
```
.
├── articles/                  # Folder with raw .txt summaries
├── qa_dataset.json            # Generated Q&A dataset (Step 1)
├── fine_tune.py               # Fine-tuning script (Step 2)
├── inference.py               # Script to test the fine-tuned model
├── .env                       # API keys for OpenAI & Hugging Face
└── README.md                  # This file
```

---

## 🔹 3. Step 1: Convert Summaries into Q&A Pairs
Run `generate_qa.py` to convert `.txt` files into a structured dataset:

```bash
python generate_qa.py
```
This creates `qa_dataset.json` for training.

---

## 🔹 4. Step 2: Fine-Tune the Model with QLoRA
Start fine-tuning with:

```bash
python fine_tune.py
```
The script:
- Loads **LLaMA-2/Mistral**
- Uses **QLoRA (4-bit quantization)** for efficiency
- Trains on **`qa_dataset.json`**
- Saves the fine-tuned model in `./finetuned-llm/`

---

## 🔹 5. Step 3: Run Inference on the Fine-Tuned Model
Test the trained model using `inference.py`:

```bash
python inference.py
```
Example output:
```
User: What is the latest news about AI startups?
Model: AI startups are focusing on generative models for automation...
```

---

## 🔑 API Keys
Create a `.env` file with your **OpenAI & Hugging Face** keys:

```
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-huggingface-key
```

---

## 🔧 Choosing the Right Model: Mistral-7B vs. LLaMA-2-7B
- **Mistral-7B** is a highly efficient model with better performance per parameter and strong generalization, making it ideal for **low-resource fine-tuning and fast inference**.
- **LLaMA-2-7B** is well-suited for **knowledge-intensive tasks** and offers strong reasoning abilities, making it a good choice for complex question-answering.
- If you need **faster inference and efficiency**, go with **Mistral-7B**.
- If you require **deeper reasoning and contextual understanding**, **LLaMA-2-7B** is preferable.

---

## 🔧 Customization
- Change `MODEL_NAME` in `fine_tune.py` to another LLM (e.g., `mistralai/Mistral-7B`)
- Adjust `chunk_size` in `generate_qa.py` for different text splits
- Modify `num_train_epochs` in `fine_tune.py` for longer/shorter training

---

## 🎯 Next Steps
- Deploy the model via **FastAPI/Gradio**
- Experiment with **longer training & better hyperparameters**
- Try fine-tuning **other models (Gemma-7B, Mixtral-8x7B)**

🚀 **Happy fine-tuning!**


