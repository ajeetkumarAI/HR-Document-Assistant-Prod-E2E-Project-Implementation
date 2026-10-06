"""Entry point.

python main.py                      # run the API (http://localhost:8000/docs)
uvicorn main:app --workers 4        # production style
"""

from __future__ import annotations

import uvicorn

from src.api.app import create_app
from src.utils.config import PROJECT_ROOT, get_settings

app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.environment == "dev",
        # Watch only source code. Watching the whole folder makes every write to logs/app.log
        # (and to .venv / data/qdrant) trigger a reload, which logs again -> endless loop.
        reload_dirs=[str(PROJECT_ROOT / "src")],
        log_config=None,  # keep our JSON logging
    )
