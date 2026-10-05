import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIRECTORY_PATH = "resource/a7"

chatHistory = []

def ingestion(directory_path):
    print("Loading ...")
    loader = DirectoryLoader(directory_path, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    print("Chunking ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    print("Embedding ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists("vectors/a7"):
        store = Chroma.from_documents(chunks, embeddings, persist_directory="vectors/a7")
    else:
        store = Chroma(persist_directory="vectors/a7", embedding_function=embeddings)

    print("Ingestion completed.")
    return store

def retrieval(query, store):
    print("Retrieving ...")
    return retrieved_chunks,condensed_query

def generation(query, retrieved_chunks):
    print("Generating response ...")
    return response.content

print("Ingestion Initiating!")
store = ingestion(DIRECTORY_PATH)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_chunks, condensed_query = retrieval(query, store)
    response = generation(condensed_query, retrieved_chunks)
    chatHistory.append({"query": query, "response": response})
    print(f"Response: {response}")
