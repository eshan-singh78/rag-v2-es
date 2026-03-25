from langchain_ollama import OllamaEmbeddings
import config


def get_embedding_function():
    return OllamaEmbeddings(model=config.embedding_model)
