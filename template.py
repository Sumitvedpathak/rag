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

def retrieval(query, vector_db):
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
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_chunks = retrieval(query, vectors)
    response = generation(query, retrieved_chunks)
    print(f"Response generated: {response}")






import os
import shutil
from dotenv import load_dotenv

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DirecTORY_PATH = "resource/a5/"

if os.path.exists("vectors/a5"):
    shutil.rmtree("vectors/a5")

def ingestion():
    return None

def retrival(query, store):
    return None

def generation(query, retrieved_chunks):
    return None

store = ingestion()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = retrival(query, store)

    response = generation(query, response)
    print(f"\nPlain Response - {response}")