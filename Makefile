.PHONY: install dev run ui ingest eval test lint format docker-up docker-down

install:        ## production deps
	pip install -r requirements.txt
dev:            ## dev deps
	pip install -r requirements-dev.txt
run:            ## start the API on :8000
	python main.py
ui:             ## start the Streamlit UI on :8501 (API must be running)
	streamlit run ui/streamlit_app.py
ingest:         ## index data/raw
	python -m scripts.ingest
eval:           ## run golden-set evaluation
	python -m scripts.evaluate
test:
	pytest --cov=src --cov-report=term-missing
lint:
	ruff check . && ruff format --check .
format:
	ruff check . --fix && ruff format .
docker-up:
	docker compose up -d --build
docker-down:
	docker compose down
