📢 Introduction
This repository combines retrieval-based and fine-tuned LLM approaches to enhance AI-powered question-answering. The RAG pipeline utilizes ChromaDB & LangChain to fetch relevant content, while the fine-tuned LLM (LLaMA-2-7B / Mistral-7B) generates responses tailored to structured Q&A datasets.

1️⃣ Retrieval-Augmented Generation (RAG) with ChromaDB & LangChain
A pipeline designed for efficient question-answering (QA) on TechCrunch articles using ChromaDB for retrieval and OpenAI GPT-4 / Hugging Face models for response generation.

🔹 Key Features
✅ Semantic search using ChromaDB
✅ Context-aware QA with LangChain
✅ Supports multiple documents for broader insights
✅ Custom LLM integration with Mistral-7B
✅ Secure API handling via .env

🔧 Setup & Usage
1️⃣ Clone the repo & install dependencies
2️⃣ Extract articles and run the Jupyter notebook
3️⃣ Query documents using GPT-4 or Hugging Face LLM

🔗 Notebook: VD_ChromaDB_+_Langchain_QA_Multiple_documents.ipynb
🔗 Custom LLM Script: Hugging_Face_LLm.py

2️⃣ Fine-Tuning an LLM with QLoRA
A streamlined approach to fine-tune LLaMA-2-7B / Mistral-7B on structured Q&A pairs generated from .txt summaries.

🔹 Key Features
✅ Q&A dataset generation from .txt files
✅ QLoRA-based fine-tuning (4-bit quantization)
✅ Optimized training with bitsandbytes, LoRA, transformers
✅ Saves fine-tuned models for real-time inference

🔧 Setup & Usage
1️⃣ Convert .txt summaries into structured Q&A pairs (generate_qa.py)
2️⃣ Fine-tune the LLM using QLoRA (fine_tune.py)
3️⃣ Run inference on the fine-tuned model (inference.py)

🔹 Choosing the Right Model:

Mistral-7B: Faster inference, efficient for low-resource setups
LLaMA-2-7B: Stronger reasoning and contextual understanding
