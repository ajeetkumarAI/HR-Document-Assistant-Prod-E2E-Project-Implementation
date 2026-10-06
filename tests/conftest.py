"""Shared fixtures. Everything runs offline: in-memory Qdrant, hashing embedder, fake LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.llm.llm_client import FakeLLMClient
from src.pipeline.container import Container, build_container
from src.utils.config import Settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "raw"

EMPLOYEE_KEY, MANAGER_KEY, ADMIN_KEY = "test-employee-key", "test-manager-key", "test-admin-key"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict = {
        "app": {"environment": "test", "log_dir": str(tmp_path / "logs"), "log_json": True},
        "security": {"auth_enabled": True, "rate_limit_per_minute": 1000},
        "embeddings": {"provider": "hashing", "model": "hashing", "dimensions": 512},
        "vectordb": {"mode": "memory", "collection": "test_hr"},
        "reranker": {"provider": "none", "top_n": 4},
        "llm": {"provider": "fake"},
        "retrieval": {"dense_score_threshold": None, "top_k": 10},
        "cache": {"backend": "memory"},
        "tracing": {"langsmith_enabled": False},
        "api_keys": {EMPLOYEE_KEY: "employee", MANAGER_KEY: "manager", ADMIN_KEY: "hr_admin"},
        "openai_api_key": None,
        "langsmith_api_key": None,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def container(settings: Settings) -> Container:
    return build_container(settings, llm=FakeLLMClient())


@pytest.fixture
def indexed_container(container: Container) -> Container:
    report = container.ingestion.ingest_directory(SAMPLE_DIR)
    assert report.summary.get("failed", 0) == 0, report.to_dict()
    return container


@pytest.fixture
def client(settings: Settings, indexed_container: Container) -> TestClient:
    app = create_app(settings, container=indexed_container)
    with TestClient(app) as c:
        yield c


def auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}
