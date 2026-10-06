"""Index documents from the command line.

python -m scripts.ingest                      # index data/raw (unchanged files skipped)
python -m scripts.ingest --path ./policies    # another folder
python -m scripts.ingest --recreate           # drop + rebuild the collection (after changing embedding model)
python -m scripts.ingest --force              # re-embed everything even if unchanged
"""

from __future__ import annotations

import argparse
import json

from src.pipeline.container import build_container
from src.utils.config import get_settings
from src.utils.logger import setup_logging
from src.utils.tracing import configure_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HR documents into the vector store")
    parser.add_argument("--path", help="directory to ingest (default: ingestion.raw_data_dir)")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection first")
    parser.add_argument("--force", action="store_true", help="re-index unchanged documents")
    args = parser.parse_args()

    s = get_settings()
    setup_logging(s.app.log_level, s.resolve_path(s.app.log_dir), json_logs=False)
    configure_tracing(
        s.tracing.langsmith_enabled,
        s.langsmith_api_key.get_secret_value() if s.langsmith_api_key else None,
        s.tracing.project,
    )
    container = build_container(s)
    if args.recreate:
        container.store.recreate()
        container.response_cache.invalidate()
    report = container.ingestion.ingest_directory(s.resolve_path(args.path or s.ingestion.raw_data_dir), args.force)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Collection '{s.vectordb.collection}' now holds {container.store.count()} chunks")


if __name__ == "__main__":
    main()
