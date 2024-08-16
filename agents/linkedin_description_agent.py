import os
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool 
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.tools import get_profile_content_tavily
from config.tp_secrets import Secrets

def content_lookup(name: str) -> str:

    ##### MAIN LLM
    # the main llm that will be used throughout

    llm = ChatOllama(model="llama3.1")

    ##### UNIQUE PROMPT
    # i.e. the question that I want solved

    template = """given the full name {name_of_person} I want you to get their most recent
        LinkedIn profile description.  Your answer should contain a brief LinkedIn profile description."""

    prompt_template = PromptTemplate(
        template=template, input_variables=['name_of_person']
    )

    ##### REACT AGENT
    # how the llm should solve the 'unique prompt' a.k.a. the question that I want solved

    tools_for_agent = [
        Tool(
            name="Call Tavity for LinkedIn profile description",
            func=get_profile_content_tavily,
            description="useful for when you need get a LinkedIn user description" # used to determine if this tool should be used or not
        )
    ]

    os.environ['LANGCHAIN_API_KEY'] = Secrets.langsmith_api_key
    react_prompt = hub.pull("hwchase17/react")
    # react_template = """ 

    #     Answer the following questions as best you can. You have access to the following tools:

    #     {tools}

    #     Use the following format:

    #     Question: the input question you must answer
    #     Thought: you should always think about what to do
    #     Action: the action to take, should be one of [{tool_names}]
    #     Action Input: the input to the action
    #     Observation: the result of the action
    #     ... (this Thought/Action/Action Input/Observation can repeat N times)
    #     Thought: I now know the final answer
    #     Final Answer: the final answer to the original input question

    #     Begin!

    #     Question: {input}
    #     Thought:{agent_scratchpad}

    # """

    # react_prompt = PromptTemplate(
    #     template=react_template, input_variables=['tools','tool_names','input','agent_scratchpad']
    # )

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt) # recipe

    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True) # orchestrator

    ##### PUT IT ALL TOGETHER!
    ##### RUN, CLEAN AND RETURN RESULT!

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]
    
    return linkedin_profile_url

