from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    embed_model: str = Field(default="nomic-embed-text", alias="EMBED_MODEL")
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")

    chunk_size: int = Field(default=400, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=80, alias="CHUNK_OVERLAP")

    hnsw_m: int = Field(default=32, alias="HNSW_M")
    hnsw_ef_construction: int = Field(default=200, alias="HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(default=64, alias="HNSW_EF_SEARCH")
    topk: int = Field(default=15, alias="TOPK")
    topn_context: int = Field(default=3, alias="TOPN_CONTEXT")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    extra_seeds: str | None = Field(
        default="KSC|15.1|https://support.kaspersky.com/KSC/15.1/ru-RU/5022.htm,"
                "KATA|7.1|https://support.kaspersky.com/KATA/7.1/ru-RU/246841.htm",
        alias="EXTRA_SEEDS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "raw").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "clean").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "chunks").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "index").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "eval").mkdir(parents=True, exist_ok=True)
