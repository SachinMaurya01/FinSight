UV ?= uv
COMPOSE ?= docker compose

.PHONY: up sync ingest seed demo ablation eval test

up:
	$(COMPOSE) up -d
	@until $(COMPOSE) exec -T postgres pg_isready -U rag_user -d rag_db >/dev/null 2>&1; do sleep 1; done

sync:
	$(UV) sync

ingest: sync
	$(UV) run python -m src.ingestion

seed: up sync
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
