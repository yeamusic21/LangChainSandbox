
import os
from config.tp_secrets import Secrets
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
# https://python.langchain.com/v0.1/docs/integrations/text_embedding/ollama/
from langchain_community.embeddings import OllamaEmbeddings


if __name__=="__main__":
    print("Ingesting ... ")
    os.environ['INDEX_NAME'] = Secrets.pinecone_index_name
    os.environ['PINECONE_API_KEY'] = Secrets.pinecone_api_key
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OllamaEmbeddings(model="gemma2")
    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])


    