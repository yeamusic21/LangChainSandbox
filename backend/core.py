from typing import List, Dict, Any
import os 
from config.tp_secrets import Secrets
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

def run_llm(query: str):
    INDEX_NAME = "langchain-doc-index"
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name2
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    os.environ['LANGCHAIN_API_KEY'] = Secrets.langsmith_api_key
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'LangChain Doc Index'
    embeddings = OllamaEmbeddings(model="gemma2")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOllama(model="gemma2")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result

def run_llm_with_memory(query: str, chat_history: List[Dict[str, Any]] = []):
    INDEX_NAME = "langchain-doc-index"
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name2
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    os.environ['LANGCHAIN_API_KEY'] = Secrets.langsmith_api_key
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'LangChain Doc Index'
    embeddings = OllamaEmbeddings(model="gemma2")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOllama(model="gemma2")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    res = run_llm(query="What are LangChain callbacks?")
    print(res["answer"])