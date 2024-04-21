from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

'''
# could use this OLlama embeddings for local LLM 
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    # llama3: https://ollama.com/library/llama3
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

'''

def get_embedding_function():
    embeddings = embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings