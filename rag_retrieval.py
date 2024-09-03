import os
from config.tp_secrets import Secrets
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


if __name__=="__main__":
    os.environ['LANGCHAIN_API_KEY'] = Secrets.langsmith_api_key
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'Intro to RAG'
    print("Retrieving...")
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    embeddings = OllamaEmbeddings(model="gemma2")
    llm = ChatOllama(model="gemma2")
    query = "What is Pinecone in mahcine Learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)
    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrival_chain.invoke(input={"input":query})
    print(result)