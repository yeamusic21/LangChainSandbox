from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedIn import scrape_linkedin_profile

if __name__=="__main__":
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

    llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm | StrOutputParser() # pipe is part of the langchain expression language

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)
    
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
