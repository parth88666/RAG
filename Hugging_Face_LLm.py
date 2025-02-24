#!/usr/bin/env python
# coding: utf-8

# ## Langchain + ChromaDB - Q&A Multiple files with Custom LLM

# - Multiple Files
# - ChromaDB
# - Custom LLM from Hugging Face

import os
from dotenv import load_dotenv
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

# Load multiple documents and process documents
loader = DirectoryLoader("./articles/", glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create a ChromaDB
persist_directory = "db"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory
)

# Persist the db to the disk
vectordb.persist()
vectordb = None

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Create a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Load custom LLM from Hugging Face
MODEL_NAME = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query example
query = "What is the news about Pando?"
llm_response = qa_chain.run(query)
print("Response:", llm_response)

# Helper function to display output
def process_llm_response(llm_response):
    print(llm_response["result"])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# Example Queries
query = "What is the news about Pando?"
llm_response = qa_chain.run(query)
process_llm_response(llm_response)

query = "What is the news about Databricks?"
llm_response = qa_chain.run(query)
process_llm_response(llm_response)

