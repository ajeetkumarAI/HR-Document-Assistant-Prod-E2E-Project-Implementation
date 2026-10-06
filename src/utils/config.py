"""Typed application settings.

Precedence (highest first): explicit init kwargs > environment variables > .env > config.yaml.
Nested keys use "__" in env vars, e.g. ``LLM__MODEL=gpt-6.1-sol``.

WHY TYPED SETTINGS?
-------------------
Every value is validated at startup. A typo like `top_k: twenty` or `reranker.provider: llmm`
stops the app immediately with a clear error, instead of failing on the first user request.
Code reads settings as attributes with autocomplete:  settings.llm.model,  settings.retrieval.top_k
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("RAG_CONFIG_PATH", PROJECT_ROOT / "config.yaml"))


class AppConfig(BaseModel):
    name: str = "hr-document-assistant"
    version: str = "2.0.0"
    environment: Literal["dev", "staging", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json: bool = True
    cors_origins: list[str] = ["*"]
    max_upload_mb: int = 25
    default_role: str = "employee"


class SecurityConfig(BaseModel):
    auth_enabled: bool = True
    rate_limit_per_minute: int = 60
    role_access: dict[str, list[str]] = {
        "employee": ["public"],
        "manager": ["public", "manager"],
        "hr_admin": ["public", "manager", "confidential"],
    }


class IngestionConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    supported_extensions: list[str] = [".pdf", ".docx", ".txt", ".md", ".html", ".csv"]
    default_metadata: dict[str, Any] = {}


class ChunkingConfig(BaseModel):
    strategy: Literal["structure_aware", "recursive"] = "structure_aware"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60
    min_chunk_tokens: int = 30
    add_context_header: bool = True

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_smaller(cls, v: int, info: Any) -> int:
        size = info.data.get("chunk_size_tokens", 400)
        if v >= size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v


class EmbeddingsConfig(BaseModel):
    provider: Literal["openai", "hashing"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 96


class HnswConfig(BaseModel):
    m: int = 16
    ef_construct: int = 200
    ef_search: int = 128
    full_scan_threshold: int = 10000
    on_disk: bool = False


class QuantizationConfig(BaseModel):
    enabled: bool = False
    always_ram: bool = True


class VectorDBConfig(BaseModel):
    provider: Literal["qdrant"] = "qdrant"
    mode: Literal["local", "server", "memory"] = "local"
    local_path: str = "data/qdrant"
    url: str = "http://localhost:6333"
    collection: str = "hr_documents"
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    hnsw: HnswConfig = HnswConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    keyword_indexes: list[str] = ["doc_id", "source", "department", "doc_type", "access_level"]
    integer_indexes: list[str] = ["effective_ts"]


class RetrievalConfig(BaseModel):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    top_k: int = 20
    prefetch_multiplier: int = 2
    dense_score_threshold: float | None = 0.2
    use_query_rewrite: bool = True


class RerankerConfig(BaseModel):
    provider: Literal["llm", "cohere", "none"] = "llm"
    cohere_model: str = "rerank-v3.5"
    top_n: int = 5
    score_threshold: float | None = None
    fail_open: bool = True


class LLMConfig(BaseModel):
    provider: Literal["openai", "fake"] = "openai"
    model: str = "gpt-6-luna"
    fallback_model: str | None = "gpt-5.4-mini"
    utility_model: str = "gpt-6-luna"
    reasoning_effort: str | None = "low"
    max_output_tokens: int = 1024
    temperature: float | None = None
    timeout_seconds: float = 60
    context_token_budget: int = 6000


class RetryConfig(BaseModel):
    max_attempts: int = 4
    initial_wait_seconds: float = 0.5
    max_wait_seconds: float = 20
    jitter_seconds: float = 0.5


class CacheConfig(BaseModel):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 86400
    embedding_cache: bool = True
    response_cache: bool = True
    semantic_cache: bool = True
    semantic_threshold: float = 0.95
    max_entries: int = 5000


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_turns: int = 6
    ttl_seconds: int = 3600


class GuardrailsConfig(BaseModel):
    max_query_chars: int = 2000
    block_prompt_injection: bool = True
    redact_pii_in_logs: bool = True


class TracingConfig(BaseModel):
    langsmith_enabled: bool = True
    project: str = "hr-document-assistant"


class Settings(BaseSettings):
    """Root settings object. Use :func:`get_settings` to access the singleton."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=CONFIG_PATH,
    )

    app: AppConfig = AppConfig()
    security: SecurityConfig = SecurityConfig()
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vectordb: VectorDBConfig = VectorDBConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    llm: LLMConfig = LLMConfig()
    retry: RetryConfig = RetryConfig()
    cache: CacheConfig = CacheConfig()
    memory: MemoryConfig = MemoryConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    tracing: TracingConfig = TracingConfig()

    # ---- secrets (from .env / environment) ----
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    cohere_api_key: SecretStr | None = None
    qdrant_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    # JSON map of api-key -> role, e.g. {"emp-key-123": "employee", "hr-key-456": "hr_admin"}
    api_keys: dict[str, str] = Field(default_factory=dict)

    @field_validator("openai_api_key", "cohere_api_key", "qdrant_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, v: Any) -> Any:
        """Treat empty values and the .env.example placeholders as 'not configured'."""
        if v is None:
            return None
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        raw = raw.strip()
        return None if not raw or "your-key" in raw else raw

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v) if v.strip() else {}
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


# lru_cache(maxsize=1) = build Settings once and reuse the same object everywhere (a singleton).
# Note: .env / config.yaml are therefore read only at startup -> restart after editing them.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
