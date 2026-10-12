# Development

- Install dependencies: `uv sync`
- Run the service: `uv run uvicorn nextbook.app:app --host 0.0.0.0 --port "${PORT:-8000}"`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
- Tests: `uv run pytest -q`
- Build catalog and recommendation artifacts: `uv run python scripts/build_data.py`
- Fetch resumable Open Library descriptions: `uv run python scripts/fetch_descriptions.py`
- Build held-out-user evaluation artifacts: `uv run python scripts/build_data.py --holdout-frac 0.1 --seed 42 --out data_eval/`
