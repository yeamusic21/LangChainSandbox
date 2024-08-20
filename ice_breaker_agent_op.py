from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedIn import scrape_linkedin_profile
from agents.linkedin_description_agent import content_lookup as linkedin_content_lookup
from agents.linkedin_lookup_agent import lookup as linkedin_url_lookup
from output_parsers import summary_parser

def ice_breaker_with(name: str) -> str:
    # linkedin_description = linkedin_content_lookup(name=name) 
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)
    linkedin_url= linkedin_url_lookup(name=name) 
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them

        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables="information", 
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    # llm = ChatOllama(model="llama3.1")
    llm = ChatOllama(model="gemma2")

    # chain = summary_prompt_template | llm | StrOutputParser() # pipe is part of the langchain expression language
    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    chain = summary_prompt_template | llm | summary_parser

    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)
    
    res = chain.invoke(input={"information": linkedin_data})

    print(res)

if __name__=="__main__":
    ice_breaker_with(name="Andrew Ng")
    

    
