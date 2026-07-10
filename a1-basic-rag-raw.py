import os
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("LLM_MODEL")  # Default to gpt-4o if not set

FILE_PATH = "resource/421595-CO-2026.pdf"

def loading(file_path):
    """
    Load a PDF file and return its text content.

    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: The text content of the PDF file.
    """
    print("Loading ...")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunking(text):
    """
    Split the text into chunks of a specified size.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
    Returns:
        list: A list of text chunks.
    """
    print("Chunking ...")
    chunk_size = 500  # Define the size of each chunk
    overlap = 0
    length = len(text)
    start=0
    chunks=[]
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def embedding(chunks):
    """
    Generate embeddings for the given text.

    Args:
        text (str): The text to generate embeddings for.
    Returns:
        list: A list of embeddings for each chunk of text.
    """
    print("Embedding ...")
    client = OpenAI()
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return embeddings




def vector_store(vectors:list[list[float]],chunks:list[str]):
    """
    Vectorize the embeddings for further processing.

    Args:
        vectors (list[list[float]]): A list of embeddings.
        chunks (list[str]): A list of text chunks corresponding to the embeddings.
    Returns:
        list: A list of vectorized embeddings.
    """
    print("Vectorizing ...")
    vector_db:list[tuple[np.ndarray,str]] = []
    for vector, text in zip(vectors, chunks):
        vector_db.append((np.array(vector), text))
    return vector_db

def retrieval(query_embedding:list[list[float]], vector_db:list[tuple[np.ndarray,str]]):
    """
    Retrieve relevant text chunks based on a query.

    Args:
        query_embedding (list[list[float]]): The embedding of the query to search for.
    Returns:
        list: A list of relevant text chunks based on the query.
    """
    print("Retrieving ...")
    # Calculate cosine similarity between the query vector and each vector in the vector store
    query_vector = np.array(query_embedding[0])
    similarities = []
    for vector, text in vector_db:
        similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        similarities.append((similarity, text))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return the top 5 most similar text chunks
    return [text for _, text in similarities[:3]]

def generation(query:str, retrieved_chunks:list[str]):
    """
    Generate a response based on the query and retrieved text chunks.

    Args:
        query (str): The query to generate a response for.
        retrieved_chunks (list[str]): A list of relevant text chunks.
    Returns:
        str: The generated response based on the query and retrieved text chunks.
    """
    print("Generating ...")
    # client = OpenAI()
    client = Anthropic()
    context = " ".join(retrieved_chunks)
    system_prompt = """You are an AI assistant. You need to answer user's question based on the provided context only.
    Do not make up any information. If the answer is not present in the context, respond with "I don't know"."""
    prompt = f"""Here is the Context: {context} 
    You need to respond to user in below format
    Question: {query} 
    Answer:<your answer based on the context>
    Source of information: <specific chunk of text from the context that contains the answer>
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.content[0].text

print("Injestion started...")
text = loading(FILE_PATH)
chunks = chunking(text)
embeddings = embedding(chunks)
vectors = vector_store(embeddings, chunks)

print("Retrieval started...")
query = "What is the time of departure to Paris?"
query_embedding = embedding([query])
retrieved_chunks = retrieval(query_embedding, vectors)
response = generation(query, retrieved_chunks)
print(f"Response generated: {response}")
