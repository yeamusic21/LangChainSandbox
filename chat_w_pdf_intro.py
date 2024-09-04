import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

if __name__=="__main__":
    print("Let's chat with a PDF!")
    loader = PyPDFLoader(file_path="2210.03629v3.pdf")
    documents = loader.load()
    # print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=3, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    # print(docs)
    embeddings = OllamaEmbeddings(model="gemma2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index_react")
    # vectorstore_reloaded = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOllama(model="gemma2")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    query = "Give me the gist of ReAct in 3 sentences"
    result = retrival_chain.invoke(input={"input":query})
    print(result["answer"])
