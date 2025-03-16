from flask import Flask, render_template, request, jsonify
import requests
import os
import chromadb
import torch
import warnings
import logging
import time
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Constants
SARVAM_API_KEY = "c22aa186-2328-46cb-8f87-322e60d5089b"
SARVAM_API_URL_TRANSLATE = "https://api.sarvam.ai/translate"

headers = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}

# Suppress warnings and logs
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.CRITICAL, format="", force=True)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_SHOW_CPP_STACKTRACE"] = "0"

# Hugging Face API token
HF_API_TOKEN = "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"

# Load embedding model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(DEVICE)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Collection names
collections = ["General_BankFAQs", "RBI_FAQs_basic", "RBI_FAQs_advanced"]

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-3B",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={
        "temperature": 0.01,
        "max_new_tokens": 100,
        "top_p": 0.7,
        "repetition_penalty": 1.7,
    }
)

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
    """Retrieve exact answers from the database or generate meaningful responses when needed."""
    
    # Retrieve relevant results from the Vector DB
    results = search_all_collections(query, top_k=3)
    
    if results:
        context_text = get_context(query)  # Use retrieved context
    else:
        context_text = ""  # Do NOT pass 'No relevant information found.'

    # If an exact match or similar question exists, return the stored answer
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = [
        (result, cosine_similarity([query_embedding], [embedding_model.encode(result["metadata"]["question"], convert_to_numpy=True)])[0][0])
        for result in results
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    SIMILARITY_THRESHOLD = 0.75 if len(query.split()) < 6 else 0.60

    if similarities and similarities[0][1] >= SIMILARITY_THRESHOLD:
        best_match = similarities[0][0]
        return {"answer": best_match["metadata"]["answer"], "context": ""}  # Context removed from response

    # No relevant data → Generate an answer using LLM
    if results:
        formatted_input = f"{context_text}\n\nBased on the above information, answer the following question:\n\nQuestion: {query}\n\nAnswer:"
    else:
        formatted_input = f"Question: {query}\n\nAnswer in a clear and concise manner:"

    # Dynamically Adjust `max_new_tokens`
    token_length = len(query.split()) * 7  # Increase estimated token need  
    max_tokens = min(max(80, token_length), 500)  # Allow slightly longer responses  

    llm.model_kwargs["max_new_tokens"] = max_tokens  

    # Retry logic for API timeout
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = llm.invoke(formatted_input)
            
            # Ensure structured output for LLM-generated responses
            cleaned_response = response.strip()

            # Remove any FAQ-like structure in generated responses
            if "Q:" in cleaned_response or "A:" in cleaned_response:
                cleaned_response = cleaned_response.split("\n")[-1].strip()

            return {"answer": cleaned_response, "context": ""}  # Context removed from response
        
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

    # If API still fails, use a backup model
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

        # Remove FAQ-like structure again if needed
        if "Q:" in cleaned_response or "A:" in cleaned_response:
            cleaned_response = cleaned_response.split("\n")[-1].strip()

        return {"answer": cleaned_response, "context": ""}  # Context removed from response
    except Exception as e:
        print(f"❌ Backup model also failed: {e}")
        return {"answer": "Sorry, I am unable to process this request.", "context": ""}

def translate_text(input_text, source_lang='en-IN', target_lang='en-IN', mode="classic-colloquial", speaker_gender="Male", numerals_format="international", output_script=None):
    """
    Function to translate text using Sarvam AI API.
    """
    if not source_lang:
        source_lang = 'en-IN'  # Default to English if source_lang is None
    payload = {
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": speaker_gender,
        "mode": mode,
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "numerals_format": numerals_format,
        "input": input_text
    }
    
    if output_script:
        payload["output_script"] = output_script
    
    try:
        response = requests.post(SARVAM_API_URL_TRANSLATE, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get("translated_text", "Translation not available")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

@app.route("/")
def home():
    """
    Route for the home page.
    """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Route to handle chatbot interaction (text-to-text only).
    """
    try:
        data = request.json
        input_lang = data.get("language", "en-IN")  # Default to English if not provided
        user_input = data.get("input")

        if not user_input:
            return jsonify({"error": "Please provide some input."}), 400

        # Translate user input to English for processing
        if input_lang != "en-IN":
            translated_input = translate_text(user_input, source_lang=input_lang, target_lang="en-IN")
            if not translated_input:
                return jsonify({"error": "Translation failed. Please try again."}), 400
        else:
            translated_input = user_input

        # Get the answer from the loan advisor
        result = answer_question(translated_input)
        response_text = result["answer"]

        # Translate response to the input language
        if input_lang != "en-IN":
            translated_response = translate_text(response_text, source_lang="en-IN", target_lang=input_lang)
            if not translated_response:
                return jsonify({"error": "Translation failed. Please try again."}), 400
        else:
            translated_response = response_text

        # Return the response in JSON format
        return jsonify({"response": translated_response})
    except Exception as e:
        print(f"Error in /chat route: {e}")
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.route("/favicon.ico")
def favicon():
    """
    Route to ignore favicon requests.
    """
    return "", 404

if __name__ == "__main__":
    app.run(debug=True)
