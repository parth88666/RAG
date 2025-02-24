# RAG
📌 Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB and LangChain to perform Question Answering (QA) on a collection of TechCrunch articles.

🏗️ How It Works

Document Ingestion: Extracts TechCrunch articles from techcrunch_articles.zip.

Text Processing: Splits articles into manageable chunks for better retrieval.

Vector Storage: Converts text into embeddings using LangChain and stores them in ChromaDB.

Query Processing: Accepts user questions and retrieves relevant content.

Response Generation: Uses a language model (e.g., OpenAI GPT-4) to generate context-aware answers.

🛠️ Installation

1. Clone the Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install Dependencies

pip install -r requirements.txt

3. Set Up Environment Variables

Create a .env file in the project directory and add:

OPENAI_API_KEY=your_openai_api_key

🚀 Running the Notebook

Extract techcrunch_articles.zip into a data/ folder.

Open VD_ChromaDB_+_Langchain_QA_Multiple_documents.ipynb in Jupyter Notebook:

jupyter notebook VD_ChromaDB_+_Langchain_QA_Multiple_documents.ipynb

Run each cell to:

Load articles into ChromaDB

Process queries using LangChain

Generate AI-powered responses

🏆 Features

✅ Efficient Semantic Search with ChromaDB✅ Context-Aware Question Answering using LangChain✅ Handles Multiple Documents for broad topic coverage✅ Secure API Key Handling with .env file

📌 Future Enhancements

🔍 Improve embedding model for better search accuracy

📊 Add interactive UI for querying

🚀 Optimize for real-time applications

