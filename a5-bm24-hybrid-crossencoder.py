import os
import shutil
import re
from unittest import loader
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIRECTORY_PATH = "resource/a5/"

if os.path.exists("vectors/a5"):
    shutil.rmtree("vectors/a5")

class BM25Retriever:
    def __init__(self):
        self.chunks = self.load_chunks()
        tokenized_corpus = [self.tokenize(c.page_content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def load_chunks(self):
        loader = DirectoryLoader(DIRECTORY_PATH, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        return chunks

    def search(self, query, k=5):
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1][:k]
        return [
            {"chunk": self.chunks[i], "score": float(scores[i])}
            for i in ranked_indices
        ]

    def tokenize(self, text):
        return re.findall(r"[a-zA-Z0-9_/\.]+", text.lower())

def ingestion():
    loader = DirectoryLoader(DIRECTORY_PATH, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_documents(chunks, embeddings, persist_directory="vectors/a5")
    return store

def _bm25_search(query):
    retriever = BM25Retriever()
    return retriever.search(query,k=5)

def _dense_search(query, store):
    results = store.similarity_search_with_score(query, k=5)
    return [{"chunk": doc, "score": score} for doc, score in results]

def _chunk_id(chunk):
    """A stable identity for a chunk so dense and BM25 results can be matched up.
    Content hash is safer than list index — works even if the two result lists
    were built from independently-ordered iterations of the same chunk set."""
    return hash(chunk.page_content)


def rrf_rerank(dense_results, bm25_results, k=60, top_n=5):
    fused_scores = {}
    chunk_lookup = {}  # id -> actual chunk object, so we can return it later

    for rank, result in enumerate(dense_results):
        cid = _chunk_id(result["chunk"])
        chunk_lookup[cid] = result["chunk"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        cid = _chunk_id(result["chunk"])
        chunk_lookup[cid] = result["chunk"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1 / (k + rank + 1)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]

    return [
        {"chunk": chunk_lookup[cid], "fused_score": fused_scores[cid]}
        for cid in ranked_ids
    ]

def _hybrid_search(query, store):
    dense_results = _dense_search(query, store)
    bm25_results = _bm25_search(query)
    fused = rrf_rerank(dense_results, bm25_results, k=60, top_n=5)
    return fused

def _rerank(query, fused_chunks, top_n=5):
    # Placeholder for cross-encoder reranking logic
    pairs = [(query, r["chunk"].page_content) for r in fused_chunks]
    ce_scores = reranker.predict(pairs)  # one joint (query, chunk) score per pair

    for r, score in zip(fused_chunks, ce_scores):
        r["rerank_score"] = float(score)

    reranked = sorted(fused_chunks, key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_n]

def _cross_encoder_rerank(query, fused_chunks):
    print("Cross-encoding and reranking ...")
    fused_response = _hybrid_search(query, store)
    reranked = _rerank(query, fused_response, top_n=5)
    return fused_response, reranked
    

def retrival(query, store):
    print("Retrieving ...")
    sparseResponse = store.similarity_search(query, k=5)
    bm25Response = _bm25_search(query)
    fusedResponse = _hybrid_search(query, store)
    fused, reranked = _cross_encoder_rerank(query, fusedResponse)
    return reranked

def generation(query, retrieved_chunks):
    client = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0.7, max_tokens=500)
    prompt = f""" You are an helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know."
    Context:
    {' '.join([r["chunk"].page_content for r in retrieved_chunks])}
    Question:
    {query}
    """
    response = client.invoke(prompt)
    return response.content

store = ingestion()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = retrival(query, store)

    response = generation(query, response)
    print(f"\nPlain Response - {response}")