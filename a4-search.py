import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIRECTORY_PATH = "resource/a4/"

if os.path.exists("vectors/a4"):
    shutil.rmtree("vectors/a4")

def _llm(prompt):
    client = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0.7, max_tokens=500)
    response = client.invoke(prompt)
    return response.content

def _query_rewrite(query):
    print(f"Rewriting query ...")
    prompt = f""" Rewrite this as a precise, formal legal search query about lease and contract terms: '{query}'"""
    return _llm(prompt)

def _multi_query_rewrite(query) -> list[str]:
    print(f"Rewriting queries ...")
    prompt = f""" Generate 3 different phrasings of this query:\n{query}"""
    queries = _llm(prompt).split("\n")
    return queries

def _deduplicate(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk.metadata.get("chunk_id") or hash(chunk.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique

def _hyde(query):
    print(f"Generating hypothetical answer ...")
    prompt = f""" Generate a hypothetical answer to this question: '{query}'"""
    return _llm(prompt)


def ingestion():
    print("Ingesting ...")

    loader = DirectoryLoader(DIRECTORY_PATH, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_documents(chunks, embeddings, persist_directory="vectors/a4")
    return store

def retrival(query, store):
    print(f"Retrieving ...")
    plain_response = store.similarity_search(query, k=5)

    q_rewritten = _query_rewrite(query)
    qr_response = store.similarity_search(q_rewritten, k=5)
    print(f"Rewritten query: {q_rewritten}")

    qm_rewritten = _multi_query_rewrite(query)
    print(f"Multi rewritten queries: {qm_rewritten}")
    qm_resp = []
    for mquery in qm_rewritten:
        qm_resp.extend(store.similarity_search(mquery, k=5))
    qm_response = _deduplicate(qm_resp)

    qh_rewrite = _hyde(query)
    print(f"Hypothetical answer: {qh_rewrite}")
    qh_response = store.similarity_search(qh_rewrite, k=5)
    return plain_response, qr_response, qm_response, qh_response

def generation(query, retrieved_chunks):
    print(f"Generating response ...")
    context = " ".join([f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}" for chunk in retrieved_chunks])
    prompt = f""" You are an helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know."
    Context: {context}\n\nQuestion: {query}\nAnswer:"""
    return _llm(prompt)

store = ingestion()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    plain_response, qr_response, qm_response, qh_response = retrival(query, store)

    response = generation(query, plain_response)
    print(f"\nPlain Response - {response}")

    response = generation(query, qr_response)
    print(f"\nRewrite Query Response - {response}")

    response = generation(query, qm_response)
    print(f"\nMulti Query Response - {response}")

    response = generation(query, qh_response)
    print(f"\nHypothetical Response - {response}")
