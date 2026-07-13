import os
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIRECTORY_PATH = "resource/a3/"

if os.path.exists("vectors/a3"):
    shutil.rmtree("vectors/a3")

# Adding metadata to filter specific documents based on the product name. Narrowing the chunk retrival based on users requirement.
def loading():
    print("Loading ...")
    products = ["product_a", "product_b", "product_c"]
    docs = []
    for product in products:
        product_path = os.path.join(DIRECTORY_PATH, product)
        loader = DirectoryLoader(product_path, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        product_docs = loader.load()
        for doc in product_docs:
            # Extract the product name from the file path and add it to the metadata
            doc.metadata['product'] = product
            docs.append(doc)
    return docs

def chunking(text):
    print("Chunking ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(text)
    return chunks

def embedding(chunks):
    print("Embedding and storing ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma = Chroma.from_documents(chunks, embeddings,persist_directory="vectors/a3")
    # chroma = Chroma.from_documents(chunks, embeddings,distance_strategy="cos",persist_directory="vectors/a3")
    return chroma

# def vector_store(vectors, chunks):
#     print("Creating vector store ...")
#     # vectors.PersistentClient("vectors/a3")
#     return vectors

#Adding filter based on the metadata field "product" to the retrieval function. But the meta data should be added while loading the documents.
def retrieval(product, query, vector_db):
    print("Retrieving ...")
    plain_chunks = vector_db.similarity_search(query,k=5, filter={"product": product})
    print(f"Plain chunks retrieved: {plain_chunks}")
    mmr_chunks = vector_db.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.5, filter={"product": product})
    print(f"MMR chunks retrieved: {mmr_chunks}")
    return plain_chunks, mmr_chunks

def generation(query, retrieved_chunks):
    print("Generating response ...")
    context = " ".join([f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}" for chunk in retrieved_chunks])
    prompt = f""" You are an helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know." 
    Context: {context}\n\nQuestion: {query}\nAnswer:
    Answer the question, then on a new line cite the source file(s) you actually used, in this format:
    Source: <file path from the label above>"""
    client = ChatOpenAI(model="gpt-4", temperature=0)
    response = client.invoke(input=prompt)
    return response.content

print("Injestion started...")
text = loading()
chunks = chunking(text)
store = embedding(chunks)
# vectors = vector_store(embeddings, chunks)
print("Retrieval started...")
while True:
    product = input("\n\nEnter the product: ")
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    plain_chunks, mmr_chunks = retrieval(product, query, store)
    response = generation(query, plain_chunks)
    print(f"\nPlain Response generated: {response}")
    response = generation(query, mmr_chunks)
    print(f"\nMMR Response generated: {response}")
    