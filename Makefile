.PHONY: setup up down serve-tagger api n8n-import test lint

PY := .venv/bin/python

setup:  ## Create .venv (CPU only; vLLM runs from its own environment)
	~/.local/bin/uv venv --allow-existing --python 3.12 .venv
	~/.local/bin/uv pip install --python $(PY) -e ".[dev,sheets]"

up:  ## Start the mock catalog store and n8n
	docker compose up -d

down:
	docker compose down

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve-tagger:  ## Fine-tuned tagger on vLLM at :8000 (log: logs/vllm.log)
	scripts/serve_tagger.sh

api:  ## Agent API at http://127.0.0.1:8080 (docs: /docs, review page: /review/ui)
	$(PY) -m uvicorn catalog_agent.api:app --host 127.0.0.1 --port 8080


n8n-import:  ## Load the n8n workflow into the running n8n container and activate it
	docker compose exec -T n8n n8n import:workflow --input=/home/node/workflows/new_product_workflow.json
	docker compose exec -T n8n n8n update:workflow --id=catalogAgentNewProduct --active=true
	docker compose restart n8n
