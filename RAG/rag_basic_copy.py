import os
import chromadb
import torch
import requests
import json
import warnings
import logging
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
    repo_id="meta-llama/Llama-3.2-1B",
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
loan_advisor_template = """You are a financial assistant.  
Your job is to return answers **EXACTLY as given in the context**—without adding anything new.  

- **If the answer is in the context, return it VERBATIM** with no changes.  
- **If the context does NOT contain the answer, reply with: "I don't have enough information."**  
- **DO NOT rephrase, generate opinions, add explanations, or infer answers.**  

Context:  
{context}

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
    """Retrieve an exact answer from the database. If not found, use LLM."""
    results = search_all_collections(query, top_k=3)

    if not results:
        return {"answer": "I don't have enough information.", "context": "No relevant information found."}

    # ✅ If an exact match exists, return it immediately
    for result in results:
        if query.strip().lower() == result["metadata"]["question"].strip().lower():
            return {"answer": result["metadata"]["answer"], "context": get_context(query)}

    # ❌ If no exact match, call LLM
    formatted_input = prompt.format(context=get_context(query), query=query)
    
    response = llm.invoke(formatted_input)  # ✅ Prevents LLM from showing the prompt template
    
    return {"answer": response.strip(), "context": get_context(query)}

# Example usage
if __name__ == "__main__":
    print("💼 Loan Advisor Chatbot 💼")
    print("Ask me anything about loans, eligibility, or financial literacy!")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Loan Advisor. Goodbye!")
            break
            
        result = answer_question(query)
        print("\n🤖 Answer:", result["answer"])
        print("\n📚 Retrieved Context:")
        print(result["context"])
        print("-" * 50)