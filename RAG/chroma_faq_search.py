import chromadb
import torch
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Function to format the answer
def format_answer(answer):
    # Replace certain keywords (like '->') or split on punctuation marks to improve readability
    formatted_answer = answer.replace("->", "\n  ->")  # Indentation for steps
    #formatted_answer = formatted_answer.replace(".", ".\n")  # New line after each sentence
    return formatted_answer.strip()

# Function to search the FAQ database
def search_faq(collection_name, query, top_k=1):
    """Retrieve the most similar FAQ from a given ChromaDB collection."""
    # Ensure the collection exists
    collections = chroma_client.list_collections()
    if collection_name not in collections:
        print(f"❌ Collection `{collection_name}` does not exist.")
        return
    
    collection = chroma_client.get_collection(collection_name)
    query_embedding = embedding_model.encode([query], device="cuda" if torch.cuda.is_available() else "cpu", convert_to_numpy=True)[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    if results["documents"]:
        best_match = results["metadatas"][0][0]
        formatted_answer = format_answer(best_match['answer'])
        print(f"💡 Answer to your query :\n{formatted_answer}")
    else:
        print("❌ No relevant answer found.")

# Example usage:
query = "what is the meaning of mortgage"
search_faq("RBI_FAQs_basic", query)
