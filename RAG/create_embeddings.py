import os
import json
import chromadb
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 🚀 Load the embedding model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(DEVICE)

# 🚀 ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./vector_db")  # Persistent storage

# 🚀 Define JSON file paths & corresponding ChromaDB partitions
json_files = {
    "datasets/General/BankFAQs.json": "General_BankFAQs",
    "datasets/RBI/RBI_FAQs_basic.json": "RBI_FAQs_basic",
    "datasets/RBI/RBI_FAQs_advanced.json": "RBI_FAQs_advanced",
}

def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)["faqs"]

def clean_text(text):
    """Ensure text is a valid string (convert lists to a single string)."""
    if isinstance(text, list):
        return " ".join(text)  # Convert list to a space-separated string
    return str(text)  # Ensure it's a string

def process_and_store(file_name, collection_name):
    """Process JSON and store embeddings in ChromaDB."""
    # 🔹 Load JSON data
    faqs = load_json(file_name)
    
    # 🔹 Create a ChromaDB collection for this category
    collection = chroma_client.get_or_create_collection(collection_name)

    # 🔹 Prepare data for embedding
    texts = [f"Q: {item['question']} A: {clean_text(item['answer'])}" for item in faqs]
    metadata = [{
        "question": item["question"], 
        "answer": clean_text(item["answer"]),  # Ensure it's a string
        "category": item.get("category", "N/A")
    } for item in faqs]
    
    # 🔹 Generate embeddings in batches (optimized)
    batch_size = 32  # Adjust based on memory
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {collection_name}"):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, batch_size=batch_size, device=DEVICE, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    
    # 🔹 Store embeddings in ChromaDB
    for idx, embedding in enumerate(embeddings):
        collection.add(
            ids=[str(idx)],  # Unique ID for each entry
            embeddings=[embedding.tolist()],
            metadatas=[metadata[idx]]
        )
    
    print(f"✅ Stored {len(embeddings)} embeddings in `{collection_name}`.")

# 🔥 Process all JSON files
for file_name, collection_name in json_files.items():
    process_and_store(file_name, collection_name)

print("🎉 All embeddings stored successfully in ChromaDB!")


