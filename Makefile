UV ?= uv
COMPOSE ?= docker compose

.PHONY: up down logs sync migrate ingest seed demo ablation eval test lint typecheck test-cov eval-smoke eval-gate

up:
	$(COMPOSE) up -d --build
	@until $(COMPOSE) exec -T postgres pg_isready -U rag_user -d rag_db >/dev/null 2>&1; do sleep 1; done

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

sync:
	$(UV) sync

migrate: sync
	$(UV) run alembic upgrade head

ingest: sync
	$(UV) run python -m src.ingestion

seed:
	$(COMPOSE) up -d postgres redis
	@until $(COMPOSE) exec -T postgres pg_isready -U rag_user -d rag_db >/dev/null 2>&1; do sleep 1; done
	$(UV) sync
	$(UV) run python -m src.retrieval seed

# One-command local smoke demo: infra + parse/chunk + embed/seed + retrieval.
demo: up sync ingest seed
	$(UV) run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 3

# Offline retrieval comparison: dense only vs hybrid vs hybrid + rerank.
ablation: sync
	$(UV) run python -m src.eval --ablation

eval: sync
	$(UV) run python -m src.eval --limit 5

test:
	@for f in tests/test_*.py; do $(UV) run python "$$f" || exit 1; done

lint: sync
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck: sync
	$(UV) run mypy src

test-cov: sync
	$(UV) run pytest --cov=src --cov-report=term-missing --cov-fail-under=70

# Offline retrieval regression gate (no LLM/DB needed).
eval-smoke: sync
	$(UV) run python -m src.eval --ablation --limit 10 --output /tmp/eval_smoke.json
	$(UV) run python -m src.eval.gate --baseline data/eval/smoke_baseline.json --new /tmp/eval_smoke.json

# Full answer-eval gate against the checked-in baseline (needs LLM keys + seeded DB).
eval-gate: sync
	$(UV) run python -m src.eval --metrics heuristic --output /tmp/eval_full.json
	$(UV) run python -m src.eval.gate --baseline data/eval/baseline.json --new /tmp/eval_full.json
