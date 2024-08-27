from typing import Union, List
from langchain.agents import tool
from langchain.tools import Tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_ollama import ChatOllama
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad.log import format_log_to_str
from react_langchain_callbacks import AgentCallbackHandler

@tool
def get_text_length(text:str) -> int:
    """Retruns the length of a text by characters"""
    text = text.strip("'\n").strip('"') # stripping away non alphabetic characters just in case
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name:str)->Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool 
    raise ValueError(f"Tool with name {tool_name} not found.")

if __name__ == "__main__":
    print("Let's dive deep into ReAct!")
    tools = [get_text_length]
    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """ 
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join([t.name for t in tools])
    )
    llm = ChatOllama(model="gemma2", stop=["\nObservation"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []
    agent = {"input": lambda x:x["input"], "agent_scratchpad": lambda x:format_log_to_str(x['agent_scratchpad'])} | prompt | llm | ReActSingleInputOutputParser()
    # start while loop
    agent_step=""
    while not isinstance(agent_step,AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input":"What is the length in characters of the text DOG?",
                "agent_scratchpad": intermediate_steps
            }
        )
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print("agent_step: ", agent_step)
            print("observation: ", f"{observation}")
            intermediate_steps.append((agent_step, str(observation)))
    if isinstance(agent_step, AgentFinish):
        print("Final Answer: ", agent_step.return_values)






    # print("------------- CALL 1")
    # print("Note that the first call only determines what needs to be done, and only afterwards do we apply a tool...")
    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #     {
    #         "input":"What is the length in characters of the text DOG?",
    #         "agent_scratchpad": intermediate_steps
    #     }
    # )
    # print("agent_step: ", agent_step)
    # print("agent_step type: ", type(agent_step))
    # if isinstance(agent_step, AgentAction):
    #     tool_name = agent_step.tool
    #     tool_to_use = find_tool_by_name(tools, tool_name)
    #     tool_input = agent_step.tool_input
    #     observation = tool_to_use.func(str(tool_input))
    #     print("observation: ", f"{observation}")
    #     intermediate_steps.append((agent_step, str(observation)))
    # print("------------- CALL 2")
    # print("Note that previously we had a first crack at 1) what tool to apply 2) applying that tool. ")
    # print("Now we're going to make a 2nd call where it's possible we obtained the answer after applying the 1st tool.")
    # print("If not, then we need to determine next steps ...")
    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #     {
    #         "input":"What is the length in characters of the text DOG?",
    #         "agent_scratchpad": intermediate_steps
    #     }
    # )
    # print("agent_step: ", agent_step)
    # print("agent_step type: ", type(agent_step))
    # if isinstance(agent_step, AgentFinish):
    #     print("Final Answer: ", agent_step.return_values)




