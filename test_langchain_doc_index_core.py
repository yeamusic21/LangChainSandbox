from backend.core import run_llm

if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])