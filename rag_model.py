#%%
# Importing libraries
import os
import torch
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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification

from langchain.prompts import ChatPromptTemplate


#%%
# Load and process document(s)
def load_and_process_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
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
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

llm = HuggingFacePipeline(pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Control response length
        temperature=0.8,     # Add some creativity while keeping responses focused
        do_sample=True,      # Enable sampling
        top_k=50,           # Limit vocabulary choices
        top_p=0.9,         # Nucleus sampling
        repetition_penalty=1.2,  # Add repetition penalty to avoid exact copying
        pad_token_id=tokenizer.eos_token_id
    ))

#%%
# Load a reranker model
reranker_model = "BAAI/bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(reranker_model)
reranker = pipeline("text-classification", model=reranker_model, tokenizer=tokenizer)

def rerank_results(query, retrieved_docs):
    texts = [query + " [SEP] " + doc.page_content for doc in retrieved_docs]
    scores = reranker(texts)
    sorted_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1]['score'], reverse=True)
    return [doc for doc, _ in sorted_docs]


#%%
# RAG Retrieval & Response Generation
def rag_qa_system(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    
    # Rerank the retrieved documents
    reranked_docs = rerank_results(query, retrieved_docs)[:3]

    # Format retrieved documents into a structured context
    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    # Define a structured prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Analyze the provided context carefully and generate a comprehensive answer to the user's question. 
        - Synthesize information from multiple sources in the context
        - Provide a coherent, well-structured response
        - Do not simply repeat the exact phrases from the context
        - Only use information present in the context
        - If the context doesn't contain relevant information, say so"""),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a detailed answer:"),
    ])

    formatted_prompt = prompt.format_messages(context=context, question=query)
    # Note: For Llama models, you might need to convert the formatted prompt to a string
    formatted_prompt = "".join([m.content for m in formatted_prompt])
    response = llm.invoke(formatted_prompt)

    print("\n📌 **Generated Response:**")
    print(response)

    print("\n📖 **Retrieved Documents Used:**")
    for doc in reranked_docs:
        print(f"- {doc.page_content[:200]}...")  # Displaying a snippet

#%%
# Main Function
if __name__ == "__main__":
    file_path = "/Users/karthik1705/Documents/UT Austin MS Docs/StatMethsII Syllabus 2025.pdf"
    if not os.path.exists("faiss_index.pkl"):
        print("🔍 Creating vector database...")
        docs = load_and_process_documents(file_path)
        vector_db = create_vector_database(docs)
    else:
        print("📂 Loading existing vector database...")
        vector_db = load_vector_database()

    query = input("🔎 Enter your query: ")
    rag_qa_system(query, vector_db)

