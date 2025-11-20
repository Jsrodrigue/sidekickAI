import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def init_worker_llm(api_key: str = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)

def init_evaluator_llm(api_key: str = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)

def init_embeddings(api_key: str = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(openai_api_key=api_key)
