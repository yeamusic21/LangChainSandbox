import os
from config.tp_secrets import Secrets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs():
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name2
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    embeddings = OllamaEmbeddings(model="gemma2") # 3584 dim
    loader = ReadTheDocsLoader("langchain-docs/langchain-docs/api.python.langchain.com/en/latest/callbacks", encoding='utf8')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    # just updating the source in the metadata
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()