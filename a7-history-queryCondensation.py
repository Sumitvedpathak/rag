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

def queryCondensation(query, chat_history):
    print("Condensing query ...")
    history_context = " ".join([f"User: {entry['query']}\nAssistant: {entry['response']}" for entry in chat_history])
    prompt = f"""You are an AI assistant. Given the following conversation history and a new user query, condense the query to its essential information for retrieval purposes.
    Conversation History: {history_context}
    New User Query: {query}
    Condensed Query:"""
    client = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)
    response = client.invoke(prompt)
    return response.content


def retrieval(query, store):
    print("Retrieving ...")
    condensed_query = queryCondensation(query, chatHistory[-5:])  # Use the last 5 entries of chat history for context
    print(f"Condensed Query: {condensed_query}")
    retrieved_chunks = store.similarity_search(condensed_query, k=5)
    return retrieved_chunks

def generation(query, retrieved_chunks):
    print("Generating response ...")
    context = " ".join([f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}" for chunk in retrieved_chunks])
    prompt = f""" You are an helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know."
    Context: {context}\n\nQuestion: {query}\nAnswer:
    Answer the question, then on a new line cite the source file(s) you actually used, in this format:
    Source: <file path from the label above>
    If information from multiple sources was used, list each one. If sources disagree, say so explicitly and cite each source separately rather than picking one."""
    client = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=500)
    response = client.invoke(prompt)
    return response.content

print("Ingestion Initiating!")
store = ingestion(DIRECTORY_PATH)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_chunks = retrieval(query, store)
    response = generation(query, retrieved_chunks)
    chatHistory.append({"query": query, "response": response})
    print(f"Response: {response}")
