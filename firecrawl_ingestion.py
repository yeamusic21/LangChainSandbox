import os
from config.tp_secrets import Secrets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import FireCrawlLoader

def ingest_docs():
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name3
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    os.environ['FIRECRAWL_API_KEY'] = Secrets.firecrawl_api_key
    embeddings = OllamaEmbeddings(model="gemma2") # 3584 dim
    # langchain_documents_base_urls = [
    #     "https://python.langchain.com/v0.2/docs/integrations/chat/",
    #     "https://python.langchain.com/v0.2/docs/integrations/llms/",
    #     "https://python.langchain.com/v0.2/docs/integrations/text_embedding/",
    #     "https://python.langchain.com/v0.2/docs/integrations/document_loaders/"
    # ]
    langchain_documents_base_urls = [
        "https://python.langchain.com/v0.2/docs/integrations/chat/",
        "https://python.langchain.com/v0.2/docs/integrations/llms/"
    ]
    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url, 
            mode="crawl", 
            # params={
            #     "crawlerOptions": {"limit": 5}, 
            #     "pageOptions": {"onlyMainContent": True}, 
            #     "wait_until_done": True
            # }
        )
        docs = loader.load()
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name=Secrets.pinecone_index_name3
        )
        print("***** Loading {url} to vectorstore done *****")


if __name__ == "__main__":
    ingest_docs()