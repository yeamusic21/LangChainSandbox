from typing import Set
from backend.core import run_llm, run_llm_with_memory
import streamlit as st
from streamlit_chat import message

st.header("LangChain Udemy Course - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Ender your prompt here...")

if ("user_prompt_history" not in st.session_state and "chat_answers_history" not in st.session_state and "chat_history" not in st.session_state):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []
    

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm_with_memory(query=prompt, chat_history=st.session_state["chat_history"])
        # print("generated_response: ", generated_response)
        # sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        sources = set([doc.metadata["source"] for doc in generated_response["context"]])
        # formatted_response = (
        #     f"{generated_response["result"]} \n\n {create_sources_string(sources)}"
        # )
        formatted_response = (
            f"{generated_response["answer"]} \n\n {create_sources_string(sources)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        # st.session_state["chat_history"].append(("ai", generated_response["result"]))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
        message(user_query,is_user=True)
        message(generated_response)
