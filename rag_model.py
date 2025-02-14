#%%
# Importing libraries
import os
#import faiss
#import openai
import pickle
import numpy as np
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

#from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import HuggingFacePipeline
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


#%%
# Load and process document(s)
def load_and_process_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    return texts

# Create vector database with FAISS
def create_vector_database(texts):
    # Load an open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    # Save FAISS index
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

# Load FAISS index
def load_vector_database():
    with open("faiss_index.pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

#%%
# Defining the model
model_name = "meta-llama/Llama-3.2-1B" #"mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer))

#%%
# RAG Retrieval & Response Generation
def rag_qa_system(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa_chain({"query": query})
    
    print("\n📌 **Generated Response:**")
    print(response["result"])
    
    print("\n📖 **Retrieved Documents:**")
    for doc in response["source_documents"]:
        print(f"- {doc.page_content[:200]}...")  # Displaying snippet

#%%
# Main Function
if __name__ == "__main__":
    file_path = "xyz.pdf"
    if not os.path.exists("faiss_index.pkl"):
        print("🔍 Creating vector database...")
        docs = load_and_process_documents(file_path)
        vector_db = create_vector_database(docs)
    else:
        print("📂 Loading existing vector database...")
        vector_db = load_vector_database()

    query = input("🔎 Enter your query: ")
    rag_qa_system(query, vector_db)

