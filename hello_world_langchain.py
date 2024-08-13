from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

if __name__=="__main__":
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

    llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm | StrOutputParser() # pipe is part of the langchain expression language

    information = ["""
        Elon Reeve Musk FRS ( born June 28, 1971) is a businessman and investor known for his key roles in space company SpaceX and automotive company Tesla, Inc. 
        Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the 
        founding of The Boring Company, xAI, Neuralink and OpenAI. He is one of the wealthiest people in the world; as of August 2024, Forbes estimates 
        his net worth to be US$241 billion.[
    """]

    res = chain.invoke(input={"information": information})

    print(res)
