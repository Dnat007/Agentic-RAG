from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name:str = "My Agentic RAG"
    environment:str = "development"
    debug:bool = True

    api:str = "0.0.0.0"
    api_port:int = 8000
    
    chunk_size:int = 1000
    chunk_overlap:int = 250
    
    retrieval_top_k:int = 20
    reranker_top_k:int = 5
    
    # ye mere agent ki limit h kitni baar try krega 
    max_retrieval_attempts:int = 2
    max_agent_iterations:int = 10
    max_tool_calls : int = 10
    
    max_context_tokens: int = 6000

    # its my cache part 
    redis_url: str = "redis://localhost:6379/0"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    

@lru_cache
def get_settings():
    return Settings()