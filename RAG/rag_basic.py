import os
import chromadb
import torch
import requests
import json
import warnings
import logging
import time
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# 🚀 Suppress all warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 🚀 Suppress logs from unwanted libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# 🚀 Prevent LangChain from logging to console
logging.basicConfig(level=logging.CRITICAL, format="", force=True)

# 🚀 Stop Hugging Face logs & disable progress bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🚀 Suppress PyTorch warnings (optional)
os.environ["TORCH_SHOW_CPP_STACKTRACE"] = "0"

HF_API_TOKEN = "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"

# Load embedding model (using your existing model for consistency)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(DEVICE)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Collection names
collections = ["General_BankFAQs", "RBI_FAQs_basic", "RBI_FAQs_advanced"]

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-3B",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={
        "temperature": 0.01,  # ❄️ Almost no randomness
        "max_new_tokens": 100,  # ⏳ Reduce excessive text generation
        "top_p": 0.7,  # 🔍 Focus on high-confidence words
        "repetition_penalty": 1.7,  # 🚫 Prevents unnecessary rewording
    }
)

def search_all_collections(query, top_k=2):
    """Search across all collections and return the most relevant results."""
    query_embedding = embedding_model.encode(
        query, 
        device=DEVICE, 
        convert_to_numpy=True
    ).tolist()
    
    all_results = []
    
    # Query each collection
    for collection_name in collections:
        try:
            collection = chroma_client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format and store results
            for i in range(len(results["documents"][0])):
                all_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "collection": collection_name
                })
        except Exception as e:
            print(f"Error querying collection {collection_name}: {e}")
    
    # Sort by relevance (lower distance is better)
    all_results.sort(key=lambda x: x["distance"])
    
    # Return top results across all collections
    return all_results[:top_k]

def get_context(query, top_k=3):
    """Retrieve and format context for the query from vector database."""
    results = search_all_collections(query, top_k=top_k)
    
    if not results:
        return "No relevant information found."

    # Extract only the relevant Q&A
    contexts = [
        f"Q: {result['metadata']['question']}\nA: {result['metadata']['answer']}"
        for result in results
    ]
    
    return "\n\n".join(contexts)


# Define prompt template for loan advisor
loan_advisor_template = """{context}

Based on the above information, answer the following question:

Question: {query}  

Answer:"""



prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=loan_advisor_template
)

# Create a chain to generate answers
loan_advisor_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

def answer_question(query, language="english"):
    """Retrieve exact answers from the database or generate meaningful responses when needed."""
    
    # 🔹 Retrieve relevant results from the Vector DB
    results = search_all_collections(query, top_k=3)
    
    if results:
        context_text = get_context(query)  # ✅ Use retrieved context
    else:
        context_text = ""  # ❌ Do NOT pass 'No relevant information found.'

    # ✅ If an exact match or similar question exists, return the stored answer
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = [
        (result, cosine_similarity([query_embedding], [embedding_model.encode(result["metadata"]["question"], convert_to_numpy=True)])[0][0])
        for result in results
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    SIMILARITY_THRESHOLD = 0.75 if len(query.split()) < 6 else 0.60

    if similarities and similarities[0][1] >= SIMILARITY_THRESHOLD:
        best_match = similarities[0][0]
        return {"answer": best_match["metadata"]["answer"], "context": context_text}

    # ❌ No relevant data → Generate an answer using LLM
    formatted_input = loan_advisor_template.format(
        context=context_text if context_text else "No additional information available.",  
        query=query
    )

    # 🚀 Dynamically Adjust `max_new_tokens`
    token_length = len(query.split()) * 5  
    max_tokens = min(max(50, token_length), 150)  

    llm.model_kwargs["max_new_tokens"] = max_tokens

    # 🔹 Retry logic for API timeout
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = llm.invoke(formatted_input)
            
            # 🔥 Ensure structured output for LLM-generated responses
            cleaned_response = response.strip()

            # Remove prompt leakage if it appears
            if "You are a financial assistant." in cleaned_response:
                cleaned_response = cleaned_response.split("\n", 1)[-1].strip()

            return {"answer": cleaned_response, "context": context_text}
        
        except requests.exceptions.Timeout:
            print(f"⚠️ Hugging Face API Timeout. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)

        except Exception as e:
            if "504 Server Error" in str(e):
                print(f"⚠️ 504 Gateway Timeout. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Unexpected error: {e}")
                break

    # 🚨 If API still fails, use a backup model
    print("⚠️ Switching to backup model...")
    backup_llm = HuggingFaceHub(
        repo_id="microsoft/phi-3-mini-4k-instruct",
        huggingfacehub_api_token=HF_API_TOKEN,
        model_kwargs={
            "temperature": 0.01,
            "max_new_tokens": max_tokens,
            "top_p": 0.7,
            "repetition_penalty": 1.6,
        }
    )

    try:
        response = backup_llm.invoke(formatted_input)
        cleaned_response = response.strip()

        # Remove prompt leakage again if needed
        if "You are a financial assistant." in cleaned_response:
            cleaned_response = cleaned_response.split("\n", 1)[-1].strip()

        return {"answer": cleaned_response, "context": context_text}
    except Exception as e:
        print(f"❌ Backup model also failed: {e}")
        return {"answer": "Sorry, I am unable to process this request.", "context": "No relevant information available."}


# Example usage
if __name__ == "__main__":
    print("💼 Loan Advisor Chatbot 💼")
    print("Ask me anything about loans, eligibility, or financial literacy!")
    print("Type 'exit' to quit.")
    print("-" * 100)
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Loan Advisor. Goodbye!")
            break
            
        result = answer_question(query)
        print("\n🤖 Answer:", result["answer"])
        print("\n📚 Retrieved Context:")
        print(result["context"])
        print("-" * 100)