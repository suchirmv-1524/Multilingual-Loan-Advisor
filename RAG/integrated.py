from flask import Flask, render_template, request, jsonify
import os
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# 🚀 Suppress all warnings and logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.CRITICAL, format="", force=True)

# 🚀 Prevent Hugging Face logs & disable progress bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)

# Hugging Face API token
HF_API_TOKEN = "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"

# Load embedding model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(DEVICE)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Collection names
collections = ["General_BankFAQs", "RBI_FAQs_basic", "RBI_FAQs_advanced"]

# Initialize LLM
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

def answer_question(query, language="english"):
    """Retrieve the best matching answer from the database. If no close match, use LLM."""
    
    # 🔹 Get relevant results from ChromaDB
    results = search_all_collections(query, top_k=3)
    
    if not results:
        return {"answer": "I don't have enough information.", "context": "No relevant information found."}
    
    # 🔹 Compute similarity scores between the input query and stored questions
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = []
    
    for result in results:
        stored_question = result["metadata"]["question"]
        stored_embedding = embedding_model.encode(stored_question, convert_to_numpy=True)
        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        similarities.append((result, similarity))
    
    # 🔹 Sort results by highest similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 🔹 Define a similarity threshold (adjust as needed)
    SIMILARITY_THRESHOLD = 0.75  # You can fine-tune this value

    # ✅ If a similar enough question exists, return that result
    if similarities[0][1] >= SIMILARITY_THRESHOLD:
        best_match = similarities[0][0]
        return {"answer": best_match["metadata"]["answer"], "context": get_context(query)}

    # ❌ If no close match, call LLM
    formatted_input = prompt.format(context=get_context(query), query=query)
    
    response = llm.invoke(formatted_input)  # ✅ Prevents LLM from showing the prompt template
    
    return {"answer": response.strip(), "context": get_context(query)}

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("input")
        language = data.get("language", "english")  # Default to English if not provided

        if not user_input:
            return jsonify({"error": "Please provide some input."}), 400

        # Get the answer and context
        result = answer_question(user_input, language=language)
        
        return jsonify({
            "response": result["answer"],
            "context": result["context"]
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
