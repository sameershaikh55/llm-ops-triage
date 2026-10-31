.PHONY: help download bootstrap split eval eval-dev eval-smoke

help:
	@echo "Available commands:"
	@echo "  make test        - Run fast unit tests for prompts and FastAPI service (pytest)"
	@echo "  make api         - Start FastAPI service locally on http://127.0.0.1:8000"
	@echo "  make eval        - Run official 100-ticket M0 baseline eval on TEST split -> results/baseline.json"
	@echo "  make eval-dev    - Run 100-ticket eval on DEV split -> results/dev_eval.json"
	@echo "  make eval-smoke  - Run quick 5-ticket smoke test on DEV split -> results/smoke_eval.json"
	@echo "  make download    - Download Bitext dataset and sample 200 stratified rows"
	@echo "  make bootstrap   - Bootstrap missing ground-truth labels with Groq"
	@echo "  make split       - Audit and split 200 rows into golden_dev.jsonl and golden_test.jsonl"

test:
	uv run pytest -v

api:
	uv run uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

download:
	uv run python -m src.download_data

bootstrap:
	uv run python -m src.bootstrap_labels

split:
	uv run python -m src.prepare_golden_set

eval:
	uv run python -m src.eval --split test --output results/baseline.json

eval-dev:
	uv run python -m src.eval --split dev --output results/dev_eval.json

eval-smoke:
	uv run python -m src.eval --split dev --limit 5 --output results/smoke_eval.json
