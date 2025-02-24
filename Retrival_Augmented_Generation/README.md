 Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB and LangChain to perform Question Answering (QA) on a collection of TechCrunch articles. Additionally, it includes support for both OpenAI GPT-4 and a custom LLM from Hugging Face for enhanced response generation.

🏗️ How It Works

Document Ingestion: Extracts TechCrunch articles from techcrunch_articles.zip or loads text files from the articles/ directory.

Text Processing: Splits documents into manageable chunks for better retrieval.

Vector Storage: Converts text into embeddings using LangChain and stores them in ChromaDB.

Query Processing: Accepts user questions and retrieves relevant content.

Response Generation:

Uses OpenAI GPT-4 for general queries.

Uses a custom Hugging Face LLM (e.g., mistralai/Mistral-7B-Instruct) for local processing.

🛠️ Installation

1. Clone the Repository

git clone https://github.com/parth88666/RAG.git
cd RAG



3. Set Up Environment Variables

Create a .env file in the project directory and add:

OPENAI_API_KEY=your_openai_api_key

Ensure you load the environment variables using dotenv in your script.

🚀 Running the Notebook

Extract techcrunch_articles.zip into a data/ folder.

Open Jupyter Notebook:

jupyter notebook VD_ChromaDB_+_Langchain_QA_Multiple_documents.ipynb

Run each cell to:

Load articles into ChromaDB.

Process queries using LangChain.

Generate AI-powered responses using OpenAI GPT-4 or the Hugging Face LLM.

🤖 Using the Hugging Face LLM Pipeline

The Hugging_Face_LLm.py script provides a custom LLM integration using a Hugging Face model for response generation.

1. Running the Hugging Face LLM script

python Hugging_Face_LLm.py

2. How It Works

Loads multiple text documents from the articles/ directory.

Splits documents into smaller chunks.

Embeds text using sentence-transformers/all-MiniLM-L6-v2.

Stores vectors in ChromaDB for efficient retrieval.

Loads a custom LLM from Hugging Face (mistralai/Mistral-7B-Instruct).

Creates a retrieval-based QA chain to answer questions.

Example Queries:

query = "What is the news about Pando?"
llm_response = qa_chain.run(query)
print("Response:", llm_response)

🏆 Features

✅ Efficient Semantic Search with ChromaDB✅ Context-Aware QA using LangChain✅ Supports Multiple Documents for broader topic coverage✅ Custom LLM Support using Hugging Face Models✅ OpenAI GPT-4 Integration for enhanced response generation✅ Secure API Key Handling using .env file

📌 Future Enhancements

🔍 Improve embedding model for better search accuracy📊 Add interactive UI for querying🚀 Optimize for real-time applications

