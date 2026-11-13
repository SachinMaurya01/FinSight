# FinSight

Grounded retrieval-augmented generation (RAG) over SEC Form 10-K filings. FinSight retrieves evidence from filing text and structured XBRL facts, answers with inline citations, falls back across LLM providers, verifies numeric and textual claims before returning them, abstains when evidence is missing, and routes investment-advice-like output to human review.

> **Status: reference implementation, not a production deployment.** The pipeline and its controls are real and tested. The corpus is two sample filings, the evaluation set is small, and the service has no multi-tenant isolation or horizontal scaling. This README separates what is measured from what is only implemented or planned.

**Quick links:** [Architecture](docs/architecture.svg) · [CLI demo](docs/demo.svg) · [Results](#evaluation) · [Experiments](docs/EXPERIMENTS.md) · [Security](#security) · [API](#api-reference) · [Roadmap](#roadmap)


## Doc Refernce:

- Item 1   Business-
- Item 1A  Risk Factors
- Item 7   Management's Discussion and Analysis
- Item 8   Financial Statements
- Item 7A  Market Risk

## Contents

- [Executive summary](#executive-summary)
- [Scope](#scope)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing and quality gates](#testing-and-quality-gates)
- [Security](#security)
- [Reliability and failure modes](#reliability-and-failure-modes)
- [Observability](#observability)
- [Production readiness](#production-readiness)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Project structure](#project-structure)
- [Components](#components)
- [Contributing](#contributing)

## Executive summary

- **Problem:** 10-K answers are buried in long, table-heavy filings. Generic chat-with-PDF tools lose citation traceability and numeric precision.
- **Approach:** section-scoped, metadata-filtered EDGAR chunks; hybrid retrieval (dense + BM25 fused with weighted RRF); cross-encoder reranking; query understanding with comparison decomposition; tiered model routing; provider fallback; numeric and claim verification; abstention; and a human-in-the-loop (HITL) gate for recommendation-like output.
- **Measured:** offline retrieval ablation on 28 citation-labeled questions. Hybrid + rerank reaches `MRR@8 = 0.9000` (dense-only `0.8661`) and context precision `0.7768` (dense-only `0.7411`).
- **Not yet measured:** RAGAS faithfulness (blocked by a dependency issue), end-to-end quality beyond a 3-question smoke run, LLM-based recommendation classification, and load/latency.
- **Production status:** FastAPI service with API-key auth, per-key rate and concurrency limits, request timeouts, HITL review endpoints backed by an audit table, structured logs, Prometheus metrics, Dockerfile, and CI gates. It is not yet hardened for untrusted public traffic (see [Known limitations](#known-limitations)).

## Scope

| In scope | Out of scope (today) |
|---|---|
| 10-K HTML ingestion for Items 1A, 7, and 8 | 10-Q, 8-K, proxies, transcripts, and full note-level coverage |
| EDGAR filing discovery and download with throttling and retries; idempotent `(accession, chunk_index)` upserts | Multi-company backfill orchestration and scheduling |
| Text and structured table chunks with ticker, CIK, form, year, section, and accession metadata | Full XBRL taxonomy coverage and rowspan/colspan reconstruction |
| SEC XBRL companyfacts for TTM diluted EPS | Concept alias coverage beyond the checked tickers |
| Dense + BM25 hybrid retrieval with metadata prefilters and reranking | Parent-child retrieval and learned sparse retrieval |
| Query understanding (regex, optional LLM) with ticker comparison decomposition | Multi-turn query rewriting and HyDE |
| Provider routing and fallback (OpenAI → Groq → Gemini → vLLM) | Tenant isolation and per-resource authorization |
| Numeric-aware verification, optional LLM claim judge, abstention | Calibrated NLI-grade entailment |
| HITL gate for recommendation-like output, with audit logging | Investment advice, portfolio execution, or recommendations of any kind |
| FastAPI service, Docker image, offline retrieval ablation, eval harness | Hosted UI, SLO dashboards, on-call runbooks |

## Architecture

![FinSight architecture](docs/architecture.svg)

```
Client (curl / UI)
   │  X-API-Key
   ▼
FastAPI ── auth → per-key rate limit → concurrency cap → timeout → request-id
   │
   ▼
LangGraph (compiled once in the app lifespan)
   classify ─► parse_query ─► retrieve ─► compress ─► call_llm (+tools)
      ─► verify_citations ─► [retry to retrieve, max 2] ─► human_review_gate (interrupt)
      ─► finalize
   │                                   │
   ▼                                   ▼
Postgres + pgvector               Redis (checkpoints, cache, pending reviews)
(chunks, embeddings, api_keys,    LLM providers (fallback chain)
 hitl_audit)
```

**Offline:** `EdgarClient` or `data/*.html` (plus `data/filings.json` accession and filed-at metadata) → `parse_edgar_file` (Items 1A, 7, 8; Item 8 tables serialized as structured rows) → `chunk_document` (1000/150 text chunks plus unsplit table chunks with CIK, form, year, and accession metadata) → `embed_chunks` (sentence-transformers, 384d, model and version tagged) → pgvector schema managed by Alembic, with HNSW and metadata indexes.

**Online:** `ask()` → complexity classification → `parse_query` (tickers, year, form, intent; comparison sub-queries) → filtered retrieval (dense top-20 and BM25 top-20 → RRF → rerank to top-8) → compression → LLM call (tier routing, fallback, tools; P/E uses SEC XBRL TTM diluted EPS) → citation verification (scale-aware numeric grounding and optional semantic judge; retry on failure) → abstention on missing or unverifiable evidence → HITL gate → final answer with sources.

**Request invariants**

1. The `src.graph` path always runs `verify_citations` before `finalize`. The low-level `src.llm` CLI is a debugging path and bypasses verification and HITL.
2. Recommendation-like output pauses at the HITL gate by default. `--no-hitl` and the non-interactive `ask()` helper auto-approve. Both are local and eval escape hatches, and the API does not expose them.
3. Every graph LLM call goes through the provider fallback chain. Failures are logged with provider and reason.
4. Evidence carries `ticker`, `cik`, `form_type`, `fiscal_period`, `fiscal_year`, `section`, `accession`, `filed_at`, `chunk_index`, `chunk_type`, and `unit_scale`.
5. Dense and BM25 apply the same metadata filters to the same candidate set before fusion.
6. Missing evidence, low retrieval confidence, out-of-corpus filters, and exhausted verification return `status="abstained"` with a caveat and no sources. The system does not guess.
7. Recommendation-like answers, including pending-review drafts, carry the fixed "not investment advice" template.

## Design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Hybrid retrieval with weighted RRF | Dense handles paraphrases; BM25 handles exact figures, tickers, and rare terms | Lexical matches can crowd out narrative chunks (`eval_013`) |
| Cross-encoder rerank on fused candidates | Better ordering once candidates are in the pool | Adds latency; cannot recover chunks missing from the candidate set |
| Complexity-based model routing | Controls cost and latency for simple queries | Routing accuracy is not yet measured on labeled queries |
| Provider fallback chain with per-provider tier models | Availability under rate limits, timeouts, and auth failures; a fallback never silently serves a much weaker tier model | A fallback can change answer style; no cross-provider consistency test |
| Citation verification before return | Blocks ungrounded claims and figures from reaching users | Lexical and numeric checks are deterministic; the semantic judge is opt-in and not calibrated |
| HITL gate via LangGraph `interrupt` | Compliance control that survives API restarts when Redis is used | Requires Redis for durable pauses; the in-memory checkpointer is for development |
| Recommendation gate = regex OR optional LLM classifier | Regex is cheap and always on; the LLM catches phrasing the regex misses | On the synthetic set, regex precision is 0.80 and recall 0.67 (see [Evaluation](#recommendation-classifier)) |
| Fail-closed HITL audit | A decision is applied only after its audit row is written; if the write fails, the API returns 503 | A resume that fails after the audit write leaves an audit row with no applied decision |
| API keys stored as SHA-256 digests | A database leak does not expose usable keys | Keys cannot be recovered; they must be re-issued if lost |
| Local embeddings by default | Offline reproducibility and no embedding API cost | 384d MiniLM is a quality ceiling; not benchmarked against stronger embedders |

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.13 is pinned in `.python-version`)
- Docker with Compose v2
- Provider keys are optional. The retrieval demo and offline tests run without them.

### Local development

```bash
cp .env.example .env          # fill in provider keys only if you need LLM answers
make demo                     # compose up, sync deps, parse, embed, seed pgvector, sample query
```

Manual equivalent:

```bash
docker compose up -d postgres redis
uv sync --frozen
uv run alembic upgrade head
uv run python -m src.ingestion
uv run python -m src.retrieval seed
uv run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 3
```

Full pipeline from the CLI (needs at least one provider key):

```bash
uv run python -m src.graph ask "What were Apple's total net sales in fiscal 2023?" --k 5
uv run python -m src.graph ask "What's the current P/E ratio for AAPL given the latest filing?" --k 5
uv run python -m src.graph ask "Build a bull and bear case for AAPL, should I invest?" --k 5
```

### Run the API

API authentication is on by default. Create a key before the first request. The raw key is printed once and is not stored.

```bash
uv run python -m src.api.auth create --name local-dev
uv run uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

With Docker, the `api` service is part of the compose file. Create a key inside the container:

```bash
docker compose up -d --build
docker compose exec api python -m src.api.auth create --name local-dev
```

Smoke test (replace the placeholder with your key; never commit it):

```bash
export FINSIGHT_API_KEY="<your-key>"
curl -s http://localhost:8000/v1/ask \
  -H "X-API-Key: $FINSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What were Apple'\''s total net sales in fiscal 2023?", "k": 5}'
```

### Ingest and seed the sample corpus with Docker

```bash
docker compose --profile ingest run --rm ingest
```

## API reference

Base path `/v1` for business endpoints. Health, readiness, and metrics are unversioned. All business endpoints require `X-API-Key: <key>` or `Authorization: Bearer <key>` when `API_AUTH_ENABLED=true`.

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `POST` | `/v1/ask` | Run one question; returns a cited answer, an abstention, a degraded result, or a pending review | Key |
| `GET` | `/v1/ask/stream` | Same pipeline as Server-Sent Events: `start`, `node`, `token`, `pending_review`, `error`, `done` | Key |
| `GET` | `/v1/reviews/pending` | List recommendation drafts waiting for a human decision | Key |
| `POST` | `/v1/reviews/{thread_id}` | Approve or reject a pending draft; resumes the paused graph | Key |
| `GET` | `/health` | Liveness: the process is up | None |
| `GET` | `/ready` | Readiness: Postgres, Redis, and at least one LLM provider are usable; returns 503 otherwise | None |
| `GET` | `/metrics` | Prometheus exposition | None (restrict at the network layer) |

**`POST /v1/ask` request**

```json
{
  "query": "What was Apple's total net sales in fiscal 2023?",
  "k": 5,
  "ticker": "AAPL",
  "fiscal_year": 2023,
  "thread_id": "optional-client-thread-id"
}
```

| Field | Rule |
|---|---|
| `query` | 3–1000 characters |
| `k` | integer 1–20, default 5 |
| `ticker` | optional; matches `^[A-Z.\-]{1,10}$` |
| `fiscal_year` | optional; 2000–2100 |
| `thread_id` | optional; 1–128 characters; generated if omitted |

Unknown fields are rejected.

**Response**

```json
{
  "thread_id": "3f2a…",
  "status": "completed",
  "answer": "Apple's total net sales were $383.3 billion [1].",
  "sources": ["aapl 10-K FY2023 | Item 8 | table"],
  "provider": "openai",
  "tier": "normal",
  "verification": {"passed": true, "failed_claims": [], "verified_claims": [], "reasons": []},
  "reason": null,
  "caveat": null
}
```

`status` is one of:

| Status | Meaning |
|---|---|
| `completed` | Answer returned; `verification.passed=false` adds a caveat |
| `pending_review` | Recommendation-like draft paused for a reviewer; `answer` is the draft with the disclaimer |
| `abstained` | Withheld: no evidence, out-of-corpus filter, low rerank confidence, or exhausted verification; `sources` is empty |
| `degraded` | No provider or pipeline could complete; `reason` explains why |

**HITL review**

```bash
curl -s http://localhost:8000/v1/reviews/pending -H "X-API-Key: $FINSIGHT_API_KEY"
curl -s -X POST http://localhost:8000/v1/reviews/<thread_id> \
  -H "X-API-Key: $FINSIGHT_API_KEY" -H "Content-Type: application/json" \
  -d '{"approved": true, "note": "Reviewed against filing"}'
```

A rejection returns the withheld message instead of the draft. Every decision is written to `hitl_audit` before the graph resumes. If that write fails, the endpoint returns 503 and the decision is not applied.

**Error codes**

| Code | Cause |
|---|---|
| 401 | Missing or invalid API key |
| 404 | Review requested for a thread that is not pending |
| 422 | Request validation failed |
| 429 | Per-key rate limit or concurrency cap exceeded (`Retry-After` header set) |
| 503 | Authentication store unavailable, audit store unavailable (review), or readiness failure |
| 504 | Request exceeded `REQUEST_TIMEOUT_SECONDS` |

## Configuration

Settings are loaded from environment variables and `.env` (see `src/config.py`). Nested settings use `__`, for example `MODEL_ROUTING__CHEAP__MODEL_NAME`. Complex values are JSON, for example `FALLBACK_CHAIN=["openai","groq"]`. Never commit `.env`. `.env.example` contains placeholders only.

### Application and security

| Variable | Purpose | Default |
|---|---|---|
| `ENVIRONMENT` | `development`, `staging`, or `production`; production rejects `API_AUTH_ENABLED=false` | `development` |
| `API_AUTH_ENABLED` | Require API keys on business endpoints | `true` |
| `API_KEY_PEPPER` | Optional server-side pepper mixed into key digests. Set it in production, and never log it | unset |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Default per-key fixed-window limit (keys may override) | `60` |
| `MAX_CONCURRENT_REQUESTS_PER_KEY` | In-flight request cap per key | `4` |
| `REQUEST_TIMEOUT_SECONDS` | Work timeout for ask, stream, and review | `60` |
| `RECOMMENDATION_CLASSIFIER_ENABLED` | Add the cheap-tier LLM classifier to the regex gate | `false` |
| `LOG_QUERY_TEXT` | Include redacted query text in logs; off means length and hash only | `false` |
| `LOG_RETENTION_DAYS` | Operational log retention target (enforced by the log shipper) | `30` |
| `AUDIT_RETENTION_DAYS` | HITL audit retention target used by the purge job | `365` |

### Storage, cache, and checkpoints

| Variable | Purpose | Default |
|---|---|---|
| `POSTGRES_DATABASE_URL` | pgvector, metadata, API keys, and audit log | `postgresql://rag_user:rag_password@localhost:5432/rag_db` (local dev only) |
| `REDIS_URL` | Checkpoints, embedding cache, pending reviews, rate-limit counters | `redis://localhost:6379/0` |
| `CHECKPOINTER_TYPE` | `redis` or `in_memory`. Use `redis` for durable HITL pauses | `in_memory` |

### Retrieval and ingestion

| Variable | Purpose | Default |
|---|---|---|
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | Embedding backend | `sentence_transformers` / `all-MiniLM-L6-v2` / `384` |
| `EMBEDDING_VERSION` | Vector-space tag; queries filter on it so spaces are not mixed | `v1` |
| `RERANKER_MODEL` / `RERANKER_DEVICE` | Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `cpu` |
| `DENSE_TOP_K` / `BM25_TOP_K` / `FUSION_TOP_K` / `RERANK_TOP_K` | Retrieval depths | `20` / `20` / `20` / `8` |
| `RRF_K`, `DENSE_WEIGHT`, `LEXICAL_WEIGHT` | Fusion parameters | `60`, `0.6`, `0.4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text chunking | `1000` / `150` |
| `CONTEXT_TOKEN_BUDGET` / `MAX_COMPRESSED_CHUNKS` | Prompt budget after compression | `12000` / `8` |
| `QUERY_UNDERSTANDING_ENABLED` / `QUERY_UNDERSTANDING_MODEL` | Optional LLM query parser (regex is always the fallback) | `false` / `gpt-4o-mini` |
| `SEC_EDGAR_USER_AGENT` | Required SEC User-Agent including a contact email | `FinSightResearch dev@example.com` (replace it) |
| `SEC_REQUEST_THROTTLE_SECONDS` | Local EDGAR throttle | `0.5` |
| `SEC_SUBMISSIONS_BASE_URL` / `SEC_ARCHIVE_BASE_URL` | EDGAR endpoints | SEC URLs |
| `XBRL_ENABLED`, `SEC_COMPANYFACTS_BASE_URL`, `SEC_CACHE_TTL_SECONDS` | XBRL facts for P/E | `true`, SEC URL, `86400` |
| `SEC_MAX_ATTEMPTS` / `SEC_BACKOFF_SECONDS` | SEC retry policy | `3` / `0.5` |

### Model routing, verification, and resilience

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, `VLLM_BASE_URL`, `VLLM_MODEL` | Provider credentials and endpoints (injected at runtime) | unset / local defaults |
| `FALLBACK_CHAIN` | Provider order | `["openai","groq","gemini","vllm"]` |
| `MODEL_ROUTING__{CHEAP,MEDIUM,EXPENSIVE}__*` | Tier to model mapping | `gpt-4o-mini` / `gpt-4o` / `gpt-4-turbo` |
| `FALLBACK_TIER_MODELS` | Per-provider model for each tier; startup validation requires every provider in the chain to have one | see `src/config.py` |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_ATTEMPTS` | Per-attempt timeout and retry count | `30` / `3` |
| `RETRY_BACKOFF_MULTIPLIER` / `RETRY_BACKOFF_MAX_SECONDS` | Exponential backoff with jitter | `0.5` / `8` |
| `CIRCUIT_BREAKER_FAILURES` / `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | Per-provider breaker | `5` / `30` |
| `VERIFICATION_MAX_RETRIES` | Bounded retries to retrieval | `2` |
| `SEMANTIC_VERIFICATION_ENABLED` | Cheap-tier LLM claim judge | `false` |
| `ABSTENTION_MIN_RERANK_SCORE` | Abstain below this top rerank score; empty disables | `-12.0` |
| `HITL_ENABLED`, `HITL_INTERRUPT_NODE` | Recommendation gate | `true`, `human_review_gate` |
| `ENABLE_TOOLS` | Tool calls (price, ratios) | `true` |

### Observability and evaluation

| Variable | Purpose | Default |
|---|---|---|
| `LOG_LEVEL` | Log level | `INFO` |
| `TRACING_ENABLED` | Per-request `trace_completed` events | `true` |
| `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Optional Langfuse export (`uv sync --extra langfuse`) | `false` |
| `EVAL_DATASET_PATH` | Eval dataset | `data/eval/eval_dataset.json` |
| `RAGAS_METRICS`, `JUDGE_MODEL`, `EVAL_SEED` | RAGAS metrics, judge model, reproducibility seed | see `src/config.py` |

## Evaluation

### Datasets

| File | Contents | Size |
|---|---|---:|
| `data/eval/eval_dataset.json` | Answerable questions with section-level citation labels, plus out-of-corpus abstention cases; 3 complexity tiers, 4 recommendation-framed questions | 32 (30 answerable, 2 abstention) |
| `data/eval/prompt_injection_dataset.json` | Hostile passages injected into the top-ranked chunk | 2 |
| `data/eval/recommendation_dataset.json` | Synthetic answers labeled recommendation or not (`source: synthetic`) | 24 (12 positive, 12 negative) |

Notes:

- 28 answerable questions have section-level citation labels. Two are tool-only and are excluded from retrieval scoring.
- The retrieval ablation is offline and deterministic: local embeddings and reranker, `k=8`, checked-in chunks.
- The end-to-end baseline is a 3-question run through the full graph with Groq and the heuristic metrics backend.
- Raw artifacts: [`data/eval/retrieval_ablation.json`](data/eval/retrieval_ablation.json), [`data/eval/baseline.json`](data/eval/baseline.json), [`data/eval/smoke_baseline.json`](data/eval/smoke_baseline.json).
- Retrieval experiments and keep/drop decisions: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (E1 chunking, E2 embedder, E3 rerank depth, E4 query rewriting).

### Retrieval ablation: dense vs hybrid vs hybrid + rerank

Reproduce: `uv run python -m src.eval --ablation`

| Configuration | Context precision | Context recall | MRR | Δ MRR vs dense |
|---|---:|---:|---:|---:|
| Dense only | 0.7411 | 0.9464 | 0.8661 | — |
| Hybrid (dense + BM25, RRF) | 0.7455 | 0.9464 | 0.9167 | +0.0506 |
| Hybrid + cross-encoder rerank | 0.7768 | 0.9821 | 0.9000 | +0.0339 |

`MRR` is the mean reciprocal rank of the first relevant chunk. Precision and recall use section-level labels.

- With Item 8 table chunks in the index, reranking improves precision and recall but slightly lowers MRR versus hybrid (`0.9167 → 0.9000`). Table chunks add valid evidence and some ranking noise.
- Recall@8 is not saturated (`0.9821`), so missed Item 8 citations stay visible.
- These are retrieval-only numbers. They are not RAGAS, faithfulness, or answer-correctness scores.

### End-to-end smoke baseline (n=3, heuristic)

| Run | n | Context precision | Context recall | Faithfulness | Answer relevance | Numeric judge |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic end-to-end (Groq) | 3 | 0.9167 | 1.0000 | 1.0000 | 0.9107 | 1.0000 |

This is a smoke test, not a quality claim. All three examples passed citation verification, which is not evidence of robustness at n=3.

### Recommendation classifier

Reproduce: `uv run python -m src.eval.recommendation` (heuristic only, offline). Add `--llm` to score the production gate (heuristic OR LLM) with provider keys.

| Gate | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| Regex heuristic (default) | 0.8000 | 0.6667 | 0.7273 | Misses 4 of 12 positives, e.g. "a reasonable hold", "the better buy"; 2 false positives on "no investment recommendation" wording |
| Heuristic OR LLM | Not measured | Not measured | — | Requires `--llm` and provider keys |

The dataset is synthetic and small. Treat these numbers as a floor for the regex gate, not as a production estimate.

### Acceptance criteria

PRD targets are used where defined. "Current" is measured only where evidence exists.

| Gate | Target | Current | Status |
|---|---:|---:|---|
| Retrieval MRR@8 | ≥ 0.90 | 0.9000 | Met on 28 questions |
| Context precision@8 | ≥ 0.80 | 0.7768 | Not met on section-level labels |
| Context recall@8 | ≥ 0.80 | 0.9821 | Met |
| RAGAS faithfulness | ≥ 0.85 | Not measured | Blocked by RAGAS dependency issue |
| Citation judge | ≥ 90% grounded | 3/3 examples (14/14 claims) | Not statistically meaningful |
| Fallback success under provider failure | 100% | Unit-tested | No end-to-end chaos test |
| Routing accuracy (simple → cheap) | ≥ 90% | Not measured | No labeled routing set |
| HITL interception | 100% of recommendation-like output | Regex 0.67 recall on synthetic set | Not met |
| Abstention on unanswerable or out-of-corpus questions | 100% abstain with reason | 2 cases wired | End-to-end run needs DB and provider keys |
| Prompt-injection block rate | 100% | 2 cases wired | End-to-end run needs DB and provider keys |

### Metrics backends

- `--metrics heuristic` (default): deterministic, offline proxies. Fast, but **not RAGAS**.
- `--metrics ragas`: RAGAS (`faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`). Requires a working RAGAS install and a judge key. It fails fast and does not silently downgrade.
- `--metrics auto`: tries RAGAS once and records `metrics_backend: heuristic` and `metrics_fallback_reason` in the output.

> **Known dependency issue:** the checked-in lock (`ragas 0.3.1`) imports `PIL` and a removed `langchain_community` module. `--metrics ragas` currently exits with an actionable error. Pin a compatible `ragas` and `langchain-community` pair before publishing RAGAS numbers.

## Testing and quality gates

Tests are offline: no network, database, or provider calls. Most files are standalone scripts that exit non-zero on failure. `pytest` is also configured.

```bash
make test          # runs each tests/test_*.py script
make lint          # ruff check + ruff format --check
make typecheck     # mypy src
make test-cov      # pytest with coverage (floor 70%)
make eval-smoke    # offline retrieval regression gate
make ablation      # offline retrieval comparison
```

CI (`.github/workflows/ci.yml`):

| Job | Runs | Trigger |
|---|---|---|
| `lint-test` | ruff, mypy, pytest with coverage floor 70% | push, PR, manual |
| `security` | `detect-secrets` against `.secrets.baseline`; `pip-audit` on runtime dependencies | push, PR, nightly, manual |
| `eval-gate` | Offline retrieval smoke eval and regression gate (`hybrid_rerank` section) | push, PR, manual |
| `full-eval` | Seeds pgvector, runs the full answer eval, and applies the gate against `data/eval/baseline.json` | nightly, manual |

Dependabot (`.github/dependabot.yml`) opens weekly update PRs for uv, GitHub Actions, and Docker. `detect-secrets` also runs as a pre-commit hook.

### Verification coverage

| Layer | Coverage | Gap |
|---|---|---|
| Chunking, compression, routing, embeddings | Unit tests | No property-based or adversarial input tests |
| Dense, hybrid, and rerank retrieval | Unit tests and offline ablation | No database integration test; no load test |
| Provider fallback and tier routing | Unit tests with fake providers | No live-provider or chaos test |
| Graph node order | Test asserts `verify_citations` before `human_review_gate` | — |
| HITL interrupt and resume | End-to-end test with a real LangGraph checkpointer | Redis checkpointer not exercised in CI |
| Citation verification | Unit tests for numeric scale and unit, citation scoping, semantic judge, abstention | No labeled claim-level accuracy benchmark |
| API wiring | Tests for schemas, completed and pending flows, stream, health, readiness; 401, 429, fail-closed audit, stream timeout | Requires `fastapi` in the environment |
| Prompt-injection guard | Forged-marker neutralization, delimiter wrapping, prompt rules | Injection eval set has 2 cases and needs an LLM run |
| PII redaction and query logging | Unit tests | — |
| Audit retention | Dry-run and apply behavior against SQLite | Not tested against Postgres |
| Recommendation classifier | Metrics math; OR-ed LLM path with fake LLM | LLM accuracy not measured |
| Real RAGAS | Adapter present | Not run; dependency failure |

## Security

### Controls in place

- **Authentication:** `X-API-Key` or `Authorization: Bearer` against `api_keys` in Postgres. Keys are stored as SHA-256 digests with an optional pepper. Keys are shown once at creation. Inactive keys are rejected.
- **Authorization:** all business endpoints require a valid key. There is no per-resource authorization or tenant model.
- **Rate limiting:** per-key fixed-window limit (default 60 requests per minute, overridable per key), stored in Redis with an in-process fallback. Returns 429 with `Retry-After` and `X-RateLimit-*` headers.
- **Concurrency and timeouts:** per-key in-flight cap (default 4). Ask, stream, and review are bounded by `REQUEST_TIMEOUT_SECONDS`.
- **Input validation:** strict Pydantic models (`extra="forbid"`), query length 3–1000, `k` 1–20, ticker and year patterns.
- **Prompt injection:** retrieved passages and tool output are wrapped as `<<UNTRUSTED_…>>` data, forged delimiter markers are stripped, and the system prompt tells the model to ignore instructions inside untrusted text and not to reveal prompts or skip citations.
- **Output controls:** citation verification runs before every return path. The fixed "not investment advice" template is appended to recommendation-like output. Abstention withholds unverified answers.
- **HITL audit:** every approve or reject decision is written to `hitl_audit` (thread, reviewer key name, decision, note, draft SHA-256) before it is applied. The draft text is not stored in the audit table.
- **PII and logs:** query text is excluded from logs by default (length and hash only). Logs and traces redact emails, SSNs, and phone numbers. Tracing does not record query text.
- **Secrets:** environment-injected only. `.env` is gitignored and excluded from the Docker build context. The image contains no credentials. `.env.example` contains placeholders only. Secret scanning runs in pre-commit and CI.
- **Supply chain:** `pip-audit` on runtime dependencies in CI, Dependabot updates, and a non-root container user with a health check.
- **Source data:** SEC filings are read-only. Derived chunks and embeddings are written under `data/`.
- **SEC EDGAR access:** a contact User-Agent is required, requests are throttled, and transient failures are retried.

### Operating requirements

- Set `ENVIRONMENT=production`. This refuses to start with `API_AUTH_ENABLED=false`.
- Set `API_KEY_PEPPER` from a secrets manager. Do not commit it.
- Set `CHECKPOINTER_TYPE=redis`. HITL pauses must survive restarts.
- Replace the default Postgres password in `docker-compose.yml`, `config.py`, and CI before any deployment. These defaults are for local development only.
- Restrict `/metrics` to internal networks.
- Run `python -m src.security.retention` on a schedule. It is a dry run by default; pass `--apply` to delete rows past `AUDIT_RETENTION_DAYS`.
- Ship logs to a store that enforces `LOG_RETENTION_DAYS`. The application does not delete logs.
- Keep `--no-hitl`, `ask()` auto-approval, and the `src.llm` CLI out of production paths.

### Known security gaps

- **Prompt injection is mitigated, not solved.** Delimiters and instructions reduce risk. The injection eval set has 2 cases, and the end-to-end run is pending.
- **No tenant isolation or per-resource authorization.** Any valid key can read any thread's pending review.
- **Audit gap on partial failure.** If the graph resume fails after the audit row is written, the row records a decision that was not applied.
- **Recommendation gate is keyword-based by default.** Measured recall is 0.67 on the synthetic set. The LLM classifier is opt-in and not yet measured.
- **Pending review payloads** (query and draft) are kept in Redis for 7 days. Choose a retention period that fits your policy before use.
- **Development credentials** are committed for local use. They must be replaced for any shared environment.
- **Observability is not alerting.** There are no alerts for auth failures, 429 bursts, or verification failures yet.

Do not expose this service to untrusted users until these gaps are addressed.

## Reliability and failure modes

| Failure | Behavior |
|---|---|
| Provider timeout, 429, or 5xx | Retried with backoff, then the next provider in the chain. Repeated failures open a per-provider circuit breaker for the cooldown window. |
| All providers fail | `DegradedResponseError`; response `status="degraded"` with a reason. No unverified answer is returned. |
| Verification fails | Bounded retry to retrieval (max 2). After that, `status="abstained"` with `abstention_reason="verification_failed"`. |
| No or weak evidence | `status="abstained"` with `no_evidence` or a low-confidence reason. |
| Postgres unavailable | Retrieval cannot run; the error is surfaced. No local vector-store fallback. `/ready` returns 503. |
| Redis unavailable (`CHECKPOINTER_TYPE=redis`) | Startup fails in production. In development the checkpointer falls back to memory and logs a warning. HITL state is lost on restart in that mode. |
| Audit write fails | Review returns 503; the decision is not applied; the pending item stays pending. |
| Price fetch fails | Tool error; no price is fabricated. |
| XBRL unavailable or no diluted EPS fact | P/E degrades to a price-only answer with an explicit reason. |
| Request exceeds timeout | 504 (ask, review) or an `error` SSE event with `reason="request timed out"` (stream). |
| Partial ingestion (missing section marker) | Parser warns and continues with fewer sections. There is no completeness gate yet. |

## Observability

- **Logs:** one-line JSON with `request_id`, `thread_id`, `tier`, `provider`, `latency_ms`, and node or tool context. Clients may send `X-Request-ID`; it is echoed in the response.
- **Metrics:** `GET /metrics` exposes `finsight_requests_total{status,tier}`, `finsight_node_latency_seconds{node}`, `finsight_llm_tokens_total{provider,model,direction}`, `finsight_llm_cost_usd_total{provider,model}`, `finsight_fallback_total{from_provider,to_provider}`, `finsight_verification_failures_total{reason}`, and `finsight_hitl_total{decision}`, plus request latency and in-progress gauges.
- **Traces:** each request emits a `trace_completed` JSON event with per-node, LLM, and tool spans, latency, and token counts. Langfuse export is optional.
- **Not yet:** dashboards, alert rules, and SLO definitions. These are in Phase 8.2.

## Production readiness

| Area | Status | Notes |
|---|---|---|
| Retrieval pipeline | Implemented | Dense, BM25, RRF, rerank, compression, metadata prefilters, HNSW and composite metadata indexes |
| Ingestion | Implemented, sample scale | EDGAR client (UA, throttle, retries), idempotent upserts, embedding model and version tags, Alembic migrations |
| Query understanding | Implemented, optional LLM | Regex by default; optional structured LLM extraction; ticker comparison decomposition |
| LLM reliability | Implemented | Tier routing, fallback chain, timeouts, transient retries, circuit breakers, degraded responses |
| Citation verification | Implemented | Citation-scoped numeric grounding with scale and unit normalization; optional semantic judge; bounded retry |
| HITL | Implemented, default-on | LangGraph interrupt and resume by `thread_id`; Redis-backed pending state; audit table with fail-closed writes |
| Abstention | Implemented | No evidence, out-of-corpus filters, low rerank confidence, exhausted verification |
| Recommendation gate | Implemented, keyword default | Regex OR optional LLM; measured precision 0.80, recall 0.67 on the synthetic set |
| API | Implemented | `/v1/ask`, SSE stream, review endpoints, `/health`, `/ready`, `/metrics`; strict schemas |
| AuthN | Implemented | Hashed API keys in Postgres; CLI provisioning |
| Rate limiting | Implemented | Per-key fixed window (Redis or memory), concurrency cap, request timeouts |
| Data retention | Partial | Audit purge command (dry run by default); pending reviews expire after 7 days; log retention is external |
| Prompt-injection defense | Partial | Untrusted-data delimiters, marker stripping, system rules; end-to-end eval pending |
| Container | Implemented | Multi-stage Dockerfile with uv, pre-downloaded models, non-root user, health check |
| Compose stack | Implemented | `api`, `postgres` (pgvector), `redis`, optional `ingest` profile |
| CI | Implemented | Lint, types, tests with coverage, secret scan, dependency audit, offline eval gate, nightly full eval |
| Dependency updates | Configured | Dependabot for uv, GitHub Actions, Docker |
| Evaluation | Partial | Retrieval measured on 28 questions; end-to-end n=3; RAGAS blocked; expansion planned (Phase 8.1) |
| Observability | Implemented, no alerting | JSON logs, Prometheus metrics, traces, optional Langfuse |
| Deployment | Local and Compose | No cloud deployment, IaC, or Kubernetes manifests (out of scope until Phase 2 of the plan) |
| UI | Not started | Phase 8.5 |

## Known limitations

- **Corpus:** 2 Apple 10-Ks; Items 1A, 7, and 8. No 10-Q, 8-K, proxies, transcripts, peers, or IFRS filings.
- **Parsing:** Item 8 tables are table-aware and kept intact; other sections use text extraction. Complex rowspan and colspan tables are serialized best-effort.
- **Chunking:** section-scoped text chunks at 1000/150. Parent-child retrieval is not implemented, so E1 measures section and table awareness only.
- **Query understanding:** the regex path is the default. The LLM parser has no accuracy benchmark. Comparison decomposition covers tickers only.
- **Embeddings:** rows carry `embedding_model` and `embedding_version`, and dense search filters on the active version. There is no side-by-side multi-model table; changing dimensions recreates the table.
- **Experiments:** E2 and E4 were not adopted (see `docs/EXPERIMENTS.md`). E4 has no hard-tier labels; HyDE was not run. E3 has no p95 latency measurement.
- **P/E:** live price (yfinance) plus SEC XBRL TTM diluted EPS. Negative or zero EPS returns "not meaningful". The price is current at request time, not at the filing date.
- **Verification:** numeric and scale checks plus optional semantic judging, not calibrated entailment. Treat `verification.passed` as a floor. The abstention threshold is an initial value pending tuning.
- **Evaluation:** retrieval is measured; end-to-end is a 3-question smoke run; the abstention and injection cases are wired but not run end-to-end; RAGAS is not measured.
- **Scale:** one API process per container; no horizontal scaling, queueing, or load testing.
- **Not legal, financial, or investment advice.** Recommendation-like output is withheld pending human review and always carries a disclaimer.

## Roadmap

Priority is risk reduction, not feature count. Full checklists live in [`docs/PRD.md`](docs/PRD.md).

**P0: correctness and evidence**

1. Passage-level citation labels; expand negative and unanswerable cases.
2. Pin a working RAGAS stack; run full RAGAS; freeze judge model and prompts.
3. Calibrate the semantic judge; add NLI-based claim verification; distinguish basic and diluted EPS.
4. Extend table-aware ingestion to 10-Q, 8-K, and Item 8 notes.
5. Measure the LLM recommendation classifier and decide whether to enable it by default.

**P1: production controls**

1. Tenant isolation and per-resource authorization.
2. Tool allowlist and output sanitization for prompt-injection defense; run the injection set end-to-end.
3. Alerts and SLOs for auth failures, 429 bursts, provider errors, verification failures, and HITL volume.
4. Load test (Phase 8.2) and cost guards (Phase 8.3).

**P2: product surface**

1. Streamlit or Next.js UI: chat, streaming, clickable citations, reviewer page (Phase 8.5).
2. Peer and multi-filing comparison; temporal queries.
3. Deployment hardening and on-call runbooks.

## Components

| Layer | Package | Use |
|---|---|---|
| API | `fastapi`, `uvicorn`, `starlette` | HTTP service, SSE streaming |
| Orchestration | `langgraph`, `langchain` | Graph, state, interrupts, tools |
| LLM providers | `langchain-openai`, `langchain-groq`, `langchain-google-genai`, `openai` | Providers and fallback |
| Embeddings and rerank | `sentence-transformers` | Local 384d embeddings and cross-encoder |
| Vector store | `pgvector`, `sqlalchemy`, `psycopg2-binary`, `alembic` | Postgres with pgvector; migrations |
| Lexical search | `rank-bm25` | BM25 |
| Parsing | `beautifulsoup4`, `lxml`, `html5lib` | EDGAR HTML |
| Tokens | `tiktoken` | Budget counting |
| Cache and state | `redis`, `langgraph-checkpoint-redis` | Embedding cache, checkpoints, pending reviews, rate-limit counters |
| Metrics | `prometheus-client` | `/metrics` |
| Finance | `yfinance` | Live prices |
| Validation and config | `pydantic`, `pydantic-settings` | Schemas and settings |
| Eval | `ragas` (blocked), deterministic heuristics | Metrics |
| Resilience | `tenacity` | Retries |
| Tooling | `uv`, `ruff`, `mypy`, `pytest`, `pytest-cov`, `detect-secrets`, `pre-commit` | Dependency management and quality gates |

## State contract

`GraphState` (TypedDict) flows through the graph:

| Field | Type | Purpose |
|---|---|---|
| `query` | `str` | User question |
| `k` | `int` | Top-k to retrieve |
| `ticker`, `tickers`, `fiscal_year`, `form_type`, `intent`, `sub_queries` | various | Parsed query intent and filters |
| `tier` | `simple\|normal\|complex` | From the complexity classifier |
| `chunks` | `list[dict]` | Retrieved and compressed chunks |
| `draft_answer` / `answer` | `str` | Draft (pre-review) and final answer |
| `provider`, `model` | `str` | Provider and model that served the answer |
| `tokens_in`, `tokens_out` | `int` | Token counts |
| `sources` | `list[str]` | Distinct chunk sources |
| `tool_results` | `list[dict]` | Tool outputs |
| `verification` | `dict` | `passed`, `failed_claims`, `verified_claims`, `reasons` |
| `abstention_reason` | `str\|None` | Why the answer was withheld |
| `verification_retries` | `int` | Bounded retry count (max 2) |
| `is_recommendation` | `bool` | HITL trigger |
| `hitl_approved`, `hitl_feedback` | `bool\|None`, `str\|None` | Human decision and note |
| `degraded_reason` / `error` | `str\|None` | Degraded-mode explanation |

## Tools

- `get_stock_price(ticker)`: live price from yfinance, validated by `TickerPriceResponse`.
- `calculate_pe_ratio(price, eps, period_end=..., eps_basis=..., source=..., accession=..., as_of=...)`: `price / diluted TTM EPS`.
- `calculate_gross_margin(revenue, gross_profit)`: `gross_profit / revenue`.
- `calculate_growth_rate(current, previous)`: `(current - previous) / previous`.

All return `RatioResult` (Pydantic) with `value`, `numerator`, `denominator`, `formula`, and optional provenance. Negative or zero EPS returns `value=None` and `display="not meaningful"`. The tool node handles P/E queries automatically: it fetches the live price, pulls SEC XBRL TTM diluted EPS, computes the ratio, and appends a provenance-tagged `[Tool-augmented: …]` note. Tool output is wrapped as untrusted before it reaches the model.

## Caching and checkpoints

- **Embedding cache:** `src/retrieval/cache.py` caches query and chunk embeddings by `model::text` hash in Redis (TTL 1 hour), with an in-memory fallback.
- **Checkpoints:** `src/graph/checkpoint.py` uses `RedisSaver` when `CHECKPOINTER_TYPE=redis` and Redis is reachable. Otherwise it falls back to `MemorySaver` and logs the downgrade.
- **Pending reviews:** `src/hitl/store.py` keeps review payloads in Redis (7-day TTL) with an in-memory fallback for development.
- **Semantic answer cache:** not implemented (Phase 8.3).

## Project structure

```text
src/
  config.py            # Settings (env + .env), validators, tier and fallback maps
  routing.py           # Complexity classifier
  tokens.py            # Token counting
  api/                 # FastAPI app factory, routes, schemas, API-key auth, rate limits
  compliance/          # Fixed "not investment advice" template
  eval/                # Dataset loaders, heuristic and RAGAS metrics, judge, harness,
                       # ablation, gate, prompt-injection eval, recommendation-classifier eval
  graph/               # State, nodes, checkpointer, graph build, ask() helper
  hitl/                # Recommendation classifier, pending-review store, audit log
  ingestion/           # HTML parser (Items 1A, 7, 8), table-aware chunker
  llm/                 # Prompts, provider clients, fallback, tool calling, answer generation
  observability/       # JSON logging, request context, Prometheus metrics, tracing, pricing
  query/               # Query understanding and metadata filters
  resilience/          # Retry and circuit breaker
  retrieval/           # Embeddings, BM25, hybrid, rerank, compression, cache, storage, CLI
  sec/                 # EDGAR client and XBRL companyfacts client
  security/            # Prompt guard, PII redaction, audit retention
  tools/               # Price, ratio calculators, tool schemas
  verification/        # Numeric and semantic verification, abstention
tests/                 # Offline unit and integration-style tests
migrations/            # Alembic (0001 chunks, 0002 api_keys and hitl_audit)
data/
  *.html               # Sample SEC 10-K filings (Apple)
  filings.json         # Accession, CIK, and filed-at metadata for the samples
  parsed/, chunks/     # Derived artifacts
  eval/                # Eval, injection, recommendation datasets; baselines; ablation output
docs/
  architecture.svg, demo.svg
  EXPERIMENTS.md       # Retrieval experiments and threshold calibration
  PRD.md               # Production-readiness plan
.github/
  workflows/ci.yml     # Lint, types, tests, security, evals
  dependabot.yml       # Dependency update config
Dockerfile             # Multi-stage uv build; non-root; health check
docker-compose.yml     # api, postgres (pgvector), redis, ingest profile
Makefile               # demo, test, lint, typecheck, eval, ablation, migrate, seed
alembic.ini
pyproject.toml
uv.lock
.env.example           # Placeholders only
.secrets.baseline      # detect-secrets baseline
```

## Contributing

- Work in small changes with tests. Every new function that calls an external service needs a test or a clear TODO.
- Add type hints to new Python code. `mypy src` must pass.
- Do not change the eval datasets to improve metrics. Flag labeling problems separately.
- Keep the pipeline stages (parse, chunk, retrieve, rerank, compress, route, fallback, verify, HITL) intact. Do not bypass verification or HITL.
- Do not commit `.env`, keys, or credentials. Run `make lint typecheck test` before opening a PR.
- Migrations: add a new Alembic revision. Do not edit applied revisions.

This project does not yet have a `LICENSE` file or a `CONTRIBUTING.md`. Add them before a public release.
