import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def loading(directory_path):
    print("Loading ...")
    return None

def chunking(text):
    print("Chunking ...")
    return None

def embedding(chunks):
    print("Embedding ...")
    return None

def vector_store(vectors, chunks):
    print("Creating vector store ...")
    return None

def retrieval(query_embedding, vector_db):
    print("Retrieving ...")
    return None

def generation(query, retrieved_chunks):
    print("Generating response ...")
    return None

print("Injestion started...")
text = loading()
chunks = chunking(text)
embeddings = embedding(chunks)
vectors = vector_store(embeddings, chunks)

print("Retrieval started...")
query = "What is the time of departure to Paris?"
query_embedding = embedding([query])
retrieved_chunks = retrieval(query_embedding, vectors)
response = generation(query, retrieved_chunks)
print(f"Response generated: {response}")
    