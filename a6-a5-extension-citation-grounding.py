import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,  TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI

load_dotenv()

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_/\.]+", text.lower())

def loading():
    loader = DirectoryLoader("resource/a5/", glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists("vectors/a5"):
        store = Chroma.from_documents(chunks, embeddings, persist_directory="vectors/a5")
    else:
        store = Chroma(persist_directory="vectors/a5", embedding_function=embeddings)

    tokenized_corpus = [tokenize(c.page_content) for c in chunks] #extracts words from chunks
    # indexes is done
    # 1. How many times each word appears in the corpus
    # 2. Identify unique words that does not repeat has a high IDF(Inverse Document Frequency) score
    # 3 - How many times each word appears in the chunk
    # all these stats are stored in this bm25 object, which is used to rank the chunks based on the query
    bm25 = BM25Okapi(tokenized_corpus)  

    return chunks, store, bm25

def _dense_search(query, store, k=5):
    results = store.similarity_search_with_score(query, k=k)
    return [
        {"chunk": doc, "score": float(score)}
        for doc, score in results
    ]

def _bm25_search(query, bm25, chunks, k=5):
    # Get words from the query and convert them to lowercase
    query_tokens = tokenize(query)
    # Get scores for the query tokens based on the BM25 algorithm
    scores = bm25.get_scores(query_tokens)
    # Get the indices of the top k scores in descending order for query relevance
    ranked_indices = scores.argsort()[::-1][:k]
    return [
        {"chunk": chunks[i], "score": float(scores[i])}
        for i in ranked_indices
    ] 

def _fusion(dense_chunks, sparse_chunks, k=60, top_k=5):
    fused_scores = {}
    chunk_lookup = {}

    for rank, result in enumerate(dense_chunks):
        chunk_id = hash(result["chunk"].page_content)
        chunk_lookup[chunk_id] = result["chunk"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)  # Dense score contribution

    for rank, result in enumerate(sparse_chunks):
        chunk_id = hash(result["chunk"].page_content)
        chunk_lookup[chunk_id] = result["chunk"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)  # Sparse score contribution

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    return [
        {"chunk": chunk_lookup[chunk_id], "score": fused_scores[chunk_id]} 
        for chunk_id in ranked_ids
    ]

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Load the reranker model

def _reranker(query, fused_chunks, top_n=5):
    # This populates query and chunk pairs for then encoder
    pairs = [(query, chunk["chunk"].page_content) for chunk in fused_chunks]
    # Gets the score based on the query and the chunk content, the higher the score the more relevant the chunk is to the query
    reranker_scores = reranker.predict(pairs)

    #adds the score in chunk dictionary for each chunk in fused_chunks
    for chunk, score in zip(fused_chunks, reranker_scores):
        chunk["rerank_score"] = float(score)

    #sort the chunks based on the rerank score in descending order and return the top_n chunks
    reranked_chunks = sorted(fused_chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

    return reranked_chunks

def hybrid_search_retrieval(query,store,bm25):
    print("Searching...")
    dense_chunks = _dense_search(query, store, 20)
    sparse_chunks = _bm25_search(query, bm25, chunks, 20)  # Ensure 'chunks' is passed correctly
    fused_chunks = _fusion(dense_chunks, sparse_chunks, k=60, top_k=20)
    reranked_chunks = _reranker(query, fused_chunks, top_n=5)
    return reranked_chunks

def generation(query, reranked_chunks):
    print("Generation...")
    client = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=500)
    summarized_chunks = []
    #cannot do it in one call, as the context is too long, so we summarize each chunk and then combine them to answer the query. 
    # Also risk of misalignment is reduced, as the chunks are summarized based on the query, so the context and meaning stay intact.
    for chunk in reranked_chunks:
        prompt = f""" You are an helpful assistant. You need to summarize the chunks based on the query provided(do not change query) so that the context and the meaning stay intact. Do not remove any find any keywords
        Query: {query}
        Context:
        {chunk['chunk'].page_content}
        """
        response = client.invoke(prompt)
        summarized_chunks.append(response.content)
    
    #This will give the full answer, but it will show the citations on which chunk the answer is based on.
    sys_prompt = f""" You are an helpful assistant. Answer the question using ONLY the context below.
    Cite your sources inline using [1], [2], etc., matching the numbered context blocks.
    Every factual claim must have a citation. If the context does not contain the answer, respond exactly with:
    "I don't know based on the available documentation."
    Do not use any outside knowledge."
    Context:
    {' '.join([f'[{i+1}-{c["chunk"].metadata.get("source")}] {chunk}' for i, (c,chunk) in enumerate(zip(reranked_chunks,summarized_chunks))])} # citation provided for each chunk with number.
    Question:
    {query}
    Display all file paths used in the answer, and if information from multiple sources was used, list each one. If sources disagree, say so explicitly and cite each source separately rather than picking one.
    [1] - <file path from the label above>
    [2] - <file path from the label above>
    ...
    """
    response = client.invoke(sys_prompt)

    return response.content

chunks, store, bm25 = loading()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = hybrid_search_retrieval(query, store, bm25)
    response = generation(query, response)
    print(f"\nPlain Response - {response}")