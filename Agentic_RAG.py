import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import csv
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from FlagEmbedding import FlagReranker
from langchain_experimental.text_splitter import SemanticChunker

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# LLM model setup
llm = Ollama(model="benedict/linkbricks-llama3.1-korean:8b", temperature=0)
# llm = OpenAI() #OpenAI

# File path for CSV data
file_path = './email_neo4j_vector.csv'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Embeddings setup
embeddings_model = HuggingFaceEmbeddings(
    model_name='nlpai-lab/KoE5',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)

# CSV Loader
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path=file_path,
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "doublequote": True,
        "skipinitialspace": True,
        "quoting": csv.QUOTE_ALL,
        "escapechar": "\\",
        "lineterminator": "\r\n",
    },
    autodetect_encoding=True
)

data = loader.load()

# Text Splitter
text_splitter = SemanticChunker(
    embeddings_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.85,
)

texts = []
metadatas = []
for item in data:
    try:
        split_texts = text_splitter.split_text(item.page_content)
        texts.extend(split_texts)
        metadatas.extend([item.metadata] * len(split_texts))
    except Exception as e:
        print(f"Error during text splitting: {str(e)}")
        continue

# Vector Store
vectorstore_cosine = FAISS.from_texts(
    texts,
    embedding=embeddings_model,
    metadatas=metadatas,
    distance_strategy=DistanceStrategy.COSINE
)
vectorstore_cosine.save_local('./db/faiss_cosine')

db_cosine = FAISS.load_local('./db/faiss_cosine', embeddings_model, allow_dangerous_deserialization=True)

# Retrievers
faiss_retriever = db_cosine.as_retriever(
    search_type="mmr",
    search_kwargs={"score_threshold": 0.789, "k": 40}
)

# Reranker
reranker = FlagReranker('upskyy/ko-reranker', use_fp16=True)

def rerank_documents(query, docs, top_n=3):
    passages = [doc.page_content for doc in docs]
    scores = reranker.compute_score([[query, passage] for passage in passages])
    reranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    print("\n=== Reranked Documents (Ranked by Relevance) ===")
    for idx, (doc, score) in enumerate(reranked_results[:top_n], 1):
        print(f"\n[Rank {idx}] Score: {score:.4f}")
        print("Content:", doc.page_content)
        if doc.metadata:
            print("Metadata:", doc.metadata)
        print("-" * 70)
    
    return [doc for doc, _ in reranked_results[:top_n]]

# Process question
def process_question(question):
    # Retrieve documents using FAISS retriever
    docs = faiss_retriever.get_relevant_documents(question)
    
    # Rerank the documents
    reranked_docs = rerank_documents(question, docs)
    
    return reranked_docs

# Example usage
if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        query = f"모욕으로 판단되는 것을 찾아줘"
        
        print("\n검색 중...")
        try:
            # 문서 검색 및 리랭킹 수행
            docs = faiss_retriever.get_relevant_documents(query)
            reranked_docs = rerank_documents(query, docs)
            
            # 검색 결과 요약
            print(f"\n총 {len(reranked_docs)}개의 관련 문서를 찾았습니다.")
            print("출처별 문서 수:")
            sources = {}
            for doc in reranked_docs:
                source = doc.metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            for source, count in sources.items():
                print(f"- {source}: {count}개")
                
        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {str(e)}")
            break