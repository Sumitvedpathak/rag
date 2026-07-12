import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIRECTORY_PATH = "resource/a2/wiki_docs"

def loading(directory_path):
    print("Loading ...")
    loader = DirectoryLoader(directory_path, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    return docs

def chunking(docs):
    print("Chunking ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks

def embedding(chunks):
    print("Embedding ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = FAISS.from_documents(chunks, embeddings)
    return vectors

def vector_store(vectors):
    print("Creating vector store ...")
    vectors.save_local("vectors/a2")
    return vectors

def retrieval(query, vector_db):
    print("Retrieving ...")
    retrieved_chunks = vector_db.similarity_search(query, k=5)
    return retrieved_chunks

def generation(query, retrieved_chunks):
    print("Generating response ...")
    context = "\n\n".join([f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}" for chunk in retrieved_chunks])
    prompt = f""" You are an helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know."
    Context: {context}\n\nQuestion: {query}\nAnswer:
    Answer the question, then on a new line cite the source file(s) you actually used, in this format:
    Source: <file path from the label above>
    If information from multiple sources was used, list each one. If sources disagree, say so explicitly and cite each source separately rather than picking one."""
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4",
        input=prompt
    )
    return response.output_text

print("Injestion started...")
text = loading(DIRECTORY_PATH)
chunks = chunking(text)
embeddings = embedding(chunks)
vectors = vector_store(embeddings)

print("Retrieval started...")
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_chunks = retrieval(query, vectors)
    response = generation(query, retrieved_chunks)
    print(f"Response generated: {response}\n")
# query = "How many days of PTO do full-time employees accrue in their first two years?"
# retrieved_chunks = retrieval(query, vectors)
# response = generation(query, retrieved_chunks)
# print(f"Response generated: {response}")
