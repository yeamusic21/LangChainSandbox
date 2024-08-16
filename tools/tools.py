import os
from langchain_community.tools.tavily_search import TavilySearchResults
from config.tp_secrets import Secrets

def get_profile_url_tavily(name: str):
    """Searches for LinkedIn or Twitter Profile Page."""
    os.environ['TAVILY_API_KEY'] = Secrets.tavily_api_key
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["url"]

def get_profile_content_tavily(name: str):
    os.environ['TAVILY_API_KEY'] = Secrets.tavily_api_key
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["content"]
