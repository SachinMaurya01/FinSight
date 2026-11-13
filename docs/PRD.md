# FinSight: Production Readiness Implementation Plan

> Based on a review of the repo README (source files not inspected). File paths below are suggestions that follow your existing `src/` layout. Adjust to match your code.

## How to use this plan

- Work in phases. Each phase ends in something shippable and demo-able.
- Every task has a **Definition of Done (DoD)**. Do not tick a box until the DoD is met.
- Open one PR per task and let CI (Phase 2) guard everything after it.

| Phase | Theme | Est. effort | Outcome |
|---|---|---|---|
| 1 | Service layer (API + Docker + HITL) | 3-4 days | App runs as a service |
| 2 | CI/CD + eval regression gate | 2 days | Quality can't silently regress |
| 3 | Observability + resilience | 3 days | You can see and survive failures |
| 4 | Financial accuracy (XBRL, tables, P/E) | 5-6 days | Numbers are trustworthy |
| 5 | Retrieval + query understanding | 4-5 days | Right filing, right year, right section |
| 6 | Verification + abstention | 3 days | Wrong numbers get caught |
| 7 | Security + compliance | 3 days | Safe to expose |
| 8 | Eval expansion + UI + polish | 4 days | Demo-ready, measurable |

---

## Target architecture

```
Client (UI / curl)
   |
   v
FastAPI  --(auth, rate limit, request-id)-->  LangGraph app
   |                                             |
   |   /ask  /ask/stream  /review/{id}           |-- classify -> retrieve -> compress
   |   /health  /ready  /metrics                 |-- call_llm (+tools) -> verify
   |                                             |-- human_review_gate (interrupt)
   v                                             v
Postgres (pgvector + metadata)   Redis (cache + checkpoints)   LLM providers
        ^
        |
Ingestion worker (EDGAR fetcher + XBRL facts)
```

Proposed new layout:

```
src/
  api/
    app.py            # FastAPI factory, lifespan, middleware
    routes.py         # /ask, /ask/stream, /review, /health, /ready
    schemas.py        # request/response Pydantic models
    auth.py           # API key / JWT dependency
    ratelimit.py
  observability/
    logging.py        # structured JSON logs + request_id contextvar
    tracing.py        # Langfuse / OTel setup
    metrics.py        # Prometheus counters/histograms
  resilience/
    retry.py          # tenacity wrappers
    breaker.py        # per-provider circuit breaker
  sec/
    edgar_client.py   # rate-limited EDGAR fetcher
    xbrl.py           # companyfacts API client
  verification/
    numeric.py        # numeric claim checker
Dockerfile
Makefile
.github/workflows/ci.yml
```

---

## Phase 1: Service layer

### 1.1 FastAPI app

- [ ] Create `src/api/app.py` with an app factory and a `lifespan` that builds the graph once (not per request).
- [ ] Endpoints:
  - `POST /v1/ask` returns `{answer, sources, tier, provider, verification, thread_id, status}`
  - `GET /v1/ask/stream` streams node progress and tokens via SSE
  - `GET /v1/reviews/pending` and `POST /v1/reviews/{thread_id}` for HITL
  - `GET /health` (liveness) and `GET /ready` (checks Postgres, Redis, at least one LLM provider key)
- [ ] Define strict request models: `query` max length (e.g. 1000 chars), `k` bounded 1-20.

```python
# src/api/schemas.py
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z.\-]{1,10}$")
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    thread_id: str | None = None

class AskResponse(BaseModel):
    thread_id: str
    status: str            # "completed" | "pending_review" | "degraded"
    answer: str | None
    sources: list[str]
    provider: str | None
    tier: str | None
    verification: dict | None
```

**DoD:** `curl` against `/v1/ask` returns a cited answer; `/ready` returns 503 when Postgres is down.

### 1.2 HITL as a real workflow

- [ ] Replace the CLI approval with LangGraph `interrupt` so the graph pauses and resumes by `thread_id`.
- [ ] Persist pending state in the Redis checkpointer (already in your design). Fail loudly if Redis is configured but unreachable in production mode, instead of silently falling back to `MemorySaver`.

```python
# in human_review_gate node
from langgraph.types import interrupt

def human_review_gate(state: GraphState) -> dict:
    if not state["is_recommendation"]:
        return {"hitl_approved": True}
    decision = interrupt({
        "draft_answer": state["draft_answer"],
        "reason": "investment_recommendation",
    })
    return {"hitl_approved": bool(decision.get("approved")),
            "reviewer_note": decision.get("note")}

# in POST /v1/reviews/{thread_id}
from langgraph.types import Command
graph.invoke(Command(resume={"approved": body.approved, "note": body.note}),
             config={"configurable": {"thread_id": thread_id}})
```

**DoD:** Ask a recommendation question, restart the API container, then approve via `/reviews`; the answer completes.

### 1.3 Docker

- [ ] Multi-stage `Dockerfile` using `uv` (you already have `uv.lock`), non-root user, `HEALTHCHECK`.
- [ ] Extend `docker-compose.yml` with `api`, `postgres` (pgvector image), `redis`, and an optional `ingest` profile.
- [ ] Pre-download sentence-transformers models at build time so cold start doesn't hit the network.

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER app
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**DoD:** `docker compose up` brings up the full stack; `make seed` loads data.

---

## Phase 2: CI/CD and eval regression gate

### 2.1 CI workflow

- [ ] Add `ruff`, `mypy`, `pytest --cov` to `pyproject.toml` dev dependencies.
- [ ] Add pre-commit hooks (ruff, ruff-format, detect-secrets).

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: finsight }
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready --health-interval 5s --health-retries 10
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov=src --cov-fail-under=70
```

### 2.2 Eval regression gate

- [ ] Write `src/eval/gate.py` that compares a fresh run against `data/eval/baseline.json` and exits non-zero if any metric drops by more than a tolerance.
- [ ] Run a small smoke subset on every PR (cheap) and the full eval nightly (scheduled workflow).

```python
# src/eval/gate.py
import json, sys

TOLERANCE = {"faithfulness": 0.03, "context_recall": 0.03,
             "numeric_accuracy": 0.02, "answer_relevance": 0.05}

def main(baseline_path: str, new_path: str) -> int:
    base = json.load(open(baseline_path))["aggregate"]
    new = json.load(open(new_path))["aggregate"]
    failures = [
        f"{m}: {new[m]:.3f} < {base[m] - tol:.3f}"
        for m, tol in TOLERANCE.items()
        if new[m] < base[m] - tol
    ]
    if failures:
        print("Eval regression:\n" + "\n".join(failures))
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
```

**DoD:** A PR that deliberately breaks retrieval fails CI on the gate.

---

## Phase 3: Observability and resilience

### 3.1 Structured logging and request IDs

- [ ] JSON logs (`structlog` or `python-json-logger`).
- [ ] Middleware sets `request_id` in a `contextvar`; include `thread_id`, `tier`, `provider`, `latency_ms` on every log line.

### 3.2 Tracing and metrics

- [ ] Add Langfuse (or LangSmith / OTel) callback handler to the graph invocation so each node is a span.
- [ ] Expose Prometheus metrics at `/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `finsight_requests_total` | counter | `status`, `tier` |
| `finsight_node_latency_seconds` | histogram | `node` |
| `finsight_llm_tokens_total` | counter | `provider`, `model`, `direction` |
| `finsight_llm_cost_usd_total` | counter | `provider`, `model` |
| `finsight_fallback_total` | counter | `from_provider`, `to_provider` |
| `finsight_verification_failures_total` | counter | `reason` |
| `finsight_hitl_total` | counter | `decision` |

**DoD:** One query produces a trace showing every node with latency and token counts.

### 3.3 Retries, timeouts, circuit breakers

- [ ] Wrap each provider call with a timeout, retries on transient errors only (429, 5xx, timeouts) and exponential backoff with jitter.
- [ ] Add a per-provider circuit breaker so a dead provider is skipped for a cool-down window instead of adding its timeout to every request.

```python
# src/resilience/retry.py
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

TRANSIENT = (TimeoutError, RateLimitError, ServiceUnavailableError)  # map to your SDK errors

def with_retry(fn):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(TRANSIENT),
        reraise=True,
    )(fn)
```

```python
# src/resilience/breaker.py
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=30):
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0   # half-open: try again
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
```

- [ ] Verify the tier-to-model mapping exists for **every** provider in the fallback chain, so a fallback never serves a much weaker model.
- [ ] Return a structured degraded response (`status="degraded"`, `reason`) instead of a bare `error` string.

**DoD:** Kill the primary provider key; requests still succeed through fallback within the latency budget, and metrics show the fallback.

---

## Phase 4: Financial accuracy

### 4.1 Pull structured numbers from SEC XBRL

- [ ] Build `src/sec/xbrl.py` using the SEC `companyfacts` endpoint (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`). SEC requires a descriptive `User-Agent` header and fair-use rate limiting (stay well under 10 requests/second).
- [ ] Cache responses in Redis (facts change only when new filings land).
- [ ] Normalize facts to `{concept, value, unit, period_start, period_end, fiscal_year, fiscal_period, form, accession}`.

```python
# src/sec/xbrl.py (sketch)
import httpx

HEADERS = {"User-Agent": "FinSight your-email@example.com"}

async def company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()

def latest_annual(facts: dict, concept: str, unit: str):
    entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    return max(annual, key=lambda e: e["end"], default=None)
```

### 4.2 Rebuild the P/E tool on structured data

- [ ] Replace "extract EPS from chunks" with TTM **diluted** EPS from XBRL (sum of the last four quarters, or the latest FY value when a 10-K is the newest filing).
- [ ] Extend `RatioResult` with `period_end`, `eps_basis` (`"diluted_ttm"`), `source` (`"xbrl"`), `accession`, and `as_of` for the price.
- [ ] Add unit and period guards: refuse to divide a quarterly EPS by an annual price multiple without flagging it.
- [ ] Handle negative or zero EPS: return `"not meaningful"` instead of a number.

**DoD:** P/E for AAPL matches a trusted source within tolerance, and the answer states the EPS period used.

### 4.3 Table-aware ingestion

- [ ] Extend `parse_edgar_file` to cover **Item 8** (financial statements) in addition to Items 1A and 7.
- [ ] Parse HTML `<table>` elements into structured rows. Serialize each table as a chunk that keeps header context, e.g. `"Net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)"`.
- [ ] Never split a table across chunks. Store `chunk_type` (`text` | `table`), `section`, `unit_scale` in metadata.

**DoD:** "What was Apple's total net sales in fiscal 2023" is answered from a table chunk and the cited source is that table.

---

## Phase 5: Retrieval and query understanding

### 5.1 Metadata schema and filters

- [ ] Add columns: `ticker`, `cik`, `form_type`, `fiscal_year`, `fiscal_period`, `section`, `chunk_type`, `accession`, `filed_at`.
- [ ] Add composite indexes on `(ticker, fiscal_year, form_type)` and an **HNSW** index on the embedding column.

```sql
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_meta_idx
  ON chunks (ticker, fiscal_year, form_type);
```

- [ ] Apply the filters as SQL pre-filters in the vector query **and** in the BM25 side of hybrid search, so both sides retrieve from the same candidate set.

### 5.2 Query understanding node

- [ ] Add a `parse_query` node before `retrieve` that extracts `ticker(s)`, `fiscal_year`, `form_type`, and `intent` (lookup, ratio, comparison, recommendation).
- [ ] For comparison questions, decompose into per-entity sub-queries, retrieve for each, then merge.
- [ ] Use an LLM with a Pydantic structured-output schema, with a regex fallback for obvious cases (cheap and fast).

### 5.3 Ingestion pipeline

- [ ] `src/sec/edgar_client.py`: fetch filing lists and documents, rate-limited, with retries.
- [ ] Idempotent upserts keyed by `(accession, chunk_index)`.
- [ ] Store `embedding_model` and `embedding_version` per row. On a model change, re-embed into a new column or table instead of mixing vector spaces.
- [ ] Introduce Alembic for schema migrations.

### 5.4 Retrieval experiments (run through the eval harness)

| Experiment | Change | Keep if |
|---|---|---|
| E1 | Section-aware + parent-child chunks vs fixed 1000/150 | Recall@k up |
| E2 | `bge-base` / `e5-base` / `text-embedding-3-small` vs MiniLM | Recall@k up, latency acceptable |
| E3 | Reranker top-N 20 vs 50 | Precision up, p95 within budget |
| E4 | Query rewriting / HyDE for vague queries | Recall up on hard tier |

- [ ] Make `compress` number-safe: it must never drop sentences containing figures that match the question's entities.

**DoD:** Each experiment has a row in an `EXPERIMENTS.md` table with before/after metrics.

---

## Phase 6: Verification and abstention

### 6.1 Numeric claim verification

- [ ] Add `src/verification/numeric.py`: extract every number from the answer (normalizing `$383.3 billion`, `383,285`, `24.5%`), then confirm each appears in a cited chunk or a tool result within tolerance and with matching units.
- [ ] Treat any ungrounded number as a failed claim so the existing retry loop (max 2) triggers.

```python
import re

NUM = re.compile(r"\$?\s?(\d[\d,]*\.?\d*)\s?(billion|million|thousand|%|bn|m|k)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}

def normalize(text: str) -> list[float]:
    out = []
    for val, unit in NUM.findall(text):
        v = float(val.replace(",", ""))
        out.append(v * SCALE.get((unit or "").lower(), 1.0))
    return out

def numbers_grounded(answer: str, evidence: str, rel_tol=0.005) -> list[float]:
    """Return numbers in the answer that are NOT supported by the evidence."""
    ev = normalize(evidence)
    return [a for a in normalize(answer)
            if not any(abs(a - e) <= rel_tol * max(abs(e), 1) for e in ev)]
```

> Note: scale handling needs care (tables often report "in millions"). Use the `unit_scale` metadata from 4.3 when comparing.

### 6.2 Semantic claim verification

- [ ] If `verify_citations` is lexical, upgrade it: split the answer into claims, then check entailment against the cited chunks with an NLI cross-encoder or an LLM judge. Use the cheap model tier for this.

### 6.3 Abstention

- [ ] Return `"I can't answer this from the available filings"` when: top rerank score is below a tuned threshold, verification still fails after retries, or the requested ticker or year is not in the corpus.
- [ ] Never ship an unverified answer silently. Return `verification.passed=false` plus a caveat in the response.

**DoD:** Unanswerable and out-of-corpus eval questions get abstentions, not guesses.

---

## Phase 7: Security and compliance

- [ ] **Auth:** API-key dependency (hashed keys in Postgres) or JWT. Per-key rate limits (`slowapi` or Redis token bucket).
- [ ] **Input limits:** max query length, max concurrent requests per key, request timeout.
- [ ] **Prompt injection:** treat retrieved text and tool output as untrusted. Wrap it in delimiters, instruct the model to ignore instructions inside it, and test with injected filings in your eval set.
- [ ] **Secrets:** no secrets in the image or repo. Add `detect-secrets` to pre-commit and CI, and use env or a secrets manager in deploys.
- [ ] **Dependencies:** `pip-audit` in CI, Dependabot enabled.
- [ ] **Compliance:** fixed "not investment advice" template on all recommendation outputs. Replace keyword-based `is_recommendation` with a classifier (LLM or fine-tuned small model) and track its precision and recall in the eval set.
- [ ] **Audit log:** table `hitl_audit(thread_id, reviewer, decision, note, draft_hash, created_at)`. Log every decision.
- [ ] **PII:** make sure logs and traces don't capture user identifiers beyond what you need. Add a retention policy.

**DoD:** Unauthenticated requests get 401, burst traffic gets 429, and an injected "ignore previous instructions" filing snippet doesn't change the answer.

---

## Phase 8: Eval expansion, UI and polish

### 8.1 Eval

- [ ] Grow the dataset from 30 to 100+ questions, balanced across tiers, with **gold chunk IDs** so you can report Recall@k, MRR and nDCG separately from generation quality.
- [ ] Add numeric exact-match accuracy (tolerance and unit-aware).
- [ ] Add negative cases: wrong year, unknown ticker, out-of-scope, prompt injection.
- [ ] Use a real LLM judge from a **different** model than the generator. Report whether the heuristic fallback was used on each run so inflated scores are visible.
- [ ] Report per-tier and per-intent breakdowns, plus cost and p50/p95 latency per run.

### 8.2 Load and latency

- [ ] `locust` or `k6` script: 20 concurrent users for 5 minutes. Record p50/p95/p99, error rate, and the fallback rate.
- [ ] Set an SLO (for example p95 < 8s for `normal` tier) and alert on it.

### 8.3 Cost controls

- [ ] Per-request max tokens and max cost guard.
- [ ] Semantic answer cache (embed the query, return a cached answer above a similarity threshold, scoped by ticker and year).
- [ ] Use provider prompt caching where supported.

### 8.4 Conversation memory

- [ ] Multi-turn support keyed by `thread_id`, with follow-up resolution ("what about 2022?") before retrieval.

### 8.5 UI and repo polish

- [ ] Streamlit or Next.js UI: chat, streaming, clickable citations that expand the source passage, a reviewer page for pending HITL items.
- [ ] README: add description and topics, an architecture diagram, a Quickstart (`docker compose up`), an API reference, an eval results table, and a **Known limitations** section.
- [ ] Add `LICENSE`, `CONTRIBUTING.md`, a `Makefile` (`make up`, `make test`, `make eval`, `make seed`), and tag a `v0.1.0` release.

---

## Master checklist

**Phase 1: Service**
- [ ] FastAPI app and schemas
- [ ] HITL via `interrupt` and `/reviews`
- [ ] Dockerfile and full compose stack

**Phase 2: CI/CD**
- [ ] Lint, type-check, test, coverage in CI
- [ ] Eval regression gate (PR smoke + nightly full)

**Phase 3: Observability and resilience**
- [ ] JSON logs with request IDs
- [ ] Tracing (Langfuse / OTel)
- [ ] Prometheus metrics
- [ ] Retries, timeouts, circuit breakers
- [ ] Degraded-mode responses

**Phase 4: Financial accuracy**
- [ ] XBRL client with caching
- [ ] P/E tool rebuilt on TTM diluted EPS
- [ ] Item 8 + table-aware chunking

**Phase 5: Retrieval**
- [ ] Metadata schema, HNSW index, SQL pre-filters
- [ ] `parse_query` node and comparison decomposition
- [ ] EDGAR ingestion with idempotent upserts and Alembic
- [ ] Retrieval experiments logged in `EXPERIMENTS.md`

**Phase 6: Verification**
- [ ] Numeric claim checker
- [ ] NLI or LLM claim verification
- [ ] Abstention logic

**Phase 7: Security**
- [ ] Auth and rate limiting
- [ ] Prompt-injection defenses and tests
- [ ] Secret scanning and dependency audit
- [ ] Compliance template and audit log

**Phase 8: Polish**
- [ ] Eval set 100+ with gold chunk IDs and negatives
- [ ] Load test and SLOs
- [ ] Cost guards and semantic cache
- [ ] Multi-turn memory
- [ ] UI
- [ ] README, LICENSE, release tag

---

## Risks and notes

| Risk | Mitigation |
|---|---|
| Free-tier LLM rate limits slow evals | Cache judge calls, run the smoke subset on PRs and the full set nightly |
| XBRL concept names vary by company | Maintain a concept alias map (e.g. `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`) and test per ticker |
| Fiscal-year definitions differ from calendar year | Store `fiscal_year` from filing metadata, never derive it from dates |
| Changing the embedding model invalidates stored vectors | Version embeddings per row and re-index into a new table or column |
| Redis fallback to memory hides HITL state loss | Fail fast in production mode, allow fallback only in dev |

## Suggested resume bullets once done

- Built a production RAG service over SEC filings with hybrid retrieval, reranking, structured XBRL grounding and numeric claim verification.
- Designed an eval-gated CI pipeline that blocks regressions in faithfulness and numeric accuracy.
- Implemented provider fallback with circuit breakers, end-to-end tracing and cost and latency metrics.
- Implemented human-in-the-loop review for investment recommendations, with an audit trail.# FinSight: Production Readiness Implementation Plan

> Based on a review of the repo README (source files not inspected). File paths below are suggestions that follow your existing `src/` layout. Adjust to match your code.

## How to use this plan

- Work in phases. Each phase ends in something shippable and demo-able.
- Every task has a **Definition of Done (DoD)**. Do not tick a box until the DoD is met.
- Open one PR per task and let CI (Phase 2) guard everything after it.

| Phase | Theme | Est. effort | Outcome |
|---|---|---|---|
| 1 | Service layer (API + Docker + HITL) | 3-4 days | App runs as a service |
| 2 | CI/CD + eval regression gate | 2 days | Quality can't silently regress |
| 3 | Observability + resilience | 3 days | You can see and survive failures |
| 4 | Financial accuracy (XBRL, tables, P/E) | 5-6 days | Numbers are trustworthy |
| 5 | Retrieval + query understanding | 4-5 days | Right filing, right year, right section |
| 6 | Verification + abstention | 3 days | Wrong numbers get caught |
| 7 | Security + compliance | 3 days | Safe to expose |
| 8 | Eval expansion + UI + polish | 4 days | Demo-ready, measurable |

---

## Target architecture

```
Client (UI / curl)
   |
   v
FastAPI  --(auth, rate limit, request-id)-->  LangGraph app
   |                                             |
   |   /ask  /ask/stream  /review/{id}           |-- classify -> retrieve -> compress
   |   /health  /ready  /metrics                 |-- call_llm (+tools) -> verify
   |                                             |-- human_review_gate (interrupt)
   v                                             v
Postgres (pgvector + metadata)   Redis (cache + checkpoints)   LLM providers
        ^
        |
Ingestion worker (EDGAR fetcher + XBRL facts)
```

Proposed new layout:

```
src/
  api/
    app.py            # FastAPI factory, lifespan, middleware
    routes.py         # /ask, /ask/stream, /review, /health, /ready
    schemas.py        # request/response Pydantic models
    auth.py           # API key / JWT dependency
    ratelimit.py
  observability/
    logging.py        # structured JSON logs + request_id contextvar
    tracing.py        # Langfuse / OTel setup
    metrics.py        # Prometheus counters/histograms
  resilience/
    retry.py          # tenacity wrappers
    breaker.py        # per-provider circuit breaker
  sec/
    edgar_client.py   # rate-limited EDGAR fetcher
    xbrl.py           # companyfacts API client
  verification/
    numeric.py        # numeric claim checker
Dockerfile
Makefile
.github/workflows/ci.yml
```

---

## Phase 1: Service layer

### 1.1 FastAPI app

- [ ] Create `src/api/app.py` with an app factory and a `lifespan` that builds the graph once (not per request).
- [ ] Endpoints:
  - `POST /v1/ask` returns `{answer, sources, tier, provider, verification, thread_id, status}`
  - `GET /v1/ask/stream` streams node progress and tokens via SSE
  - `GET /v1/reviews/pending` and `POST /v1/reviews/{thread_id}` for HITL
  - `GET /health` (liveness) and `GET /ready` (checks Postgres, Redis, at least one LLM provider key)
- [ ] Define strict request models: `query` max length (e.g. 1000 chars), `k` bounded 1-20.

```python
# src/api/schemas.py
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z.\-]{1,10}$")
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    thread_id: str | None = None

class AskResponse(BaseModel):
    thread_id: str
    status: str            # "completed" | "pending_review" | "degraded"
    answer: str | None
    sources: list[str]
    provider: str | None
    tier: str | None
    verification: dict | None
```

**DoD:** `curl` against `/v1/ask` returns a cited answer; `/ready` returns 503 when Postgres is down.

### 1.2 HITL as a real workflow

- [ ] Replace the CLI approval with LangGraph `interrupt` so the graph pauses and resumes by `thread_id`.
- [ ] Persist pending state in the Redis checkpointer (already in your design). Fail loudly if Redis is configured but unreachable in production mode, instead of silently falling back to `MemorySaver`.

```python
# in human_review_gate node
from langgraph.types import interrupt

def human_review_gate(state: GraphState) -> dict:
    if not state["is_recommendation"]:
        return {"hitl_approved": True}
    decision = interrupt({
        "draft_answer": state["draft_answer"],
        "reason": "investment_recommendation",
    })
    return {"hitl_approved": bool(decision.get("approved")),
            "reviewer_note": decision.get("note")}

# in POST /v1/reviews/{thread_id}
from langgraph.types import Command
graph.invoke(Command(resume={"approved": body.approved, "note": body.note}),
             config={"configurable": {"thread_id": thread_id}})
```

**DoD:** Ask a recommendation question, restart the API container, then approve via `/reviews`; the answer completes.

### 1.3 Docker

- [ ] Multi-stage `Dockerfile` using `uv` (you already have `uv.lock`), non-root user, `HEALTHCHECK`.
- [ ] Extend `docker-compose.yml` with `api`, `postgres` (pgvector image), `redis`, and an optional `ingest` profile.
- [ ] Pre-download sentence-transformers models at build time so cold start doesn't hit the network.

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER app
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**DoD:** `docker compose up` brings up the full stack; `make seed` loads data.

---

## Phase 2: CI/CD and eval regression gate

### 2.1 CI workflow

- [ ] Add `ruff`, `mypy`, `pytest --cov` to `pyproject.toml` dev dependencies.
- [ ] Add pre-commit hooks (ruff, ruff-format, detect-secrets).

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: finsight }
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready --health-interval 5s --health-retries 10
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov=src --cov-fail-under=70
```

### 2.2 Eval regression gate

- [ ] Write `src/eval/gate.py` that compares a fresh run against `data/eval/baseline.json` and exits non-zero if any metric drops by more than a tolerance.
- [ ] Run a small smoke subset on every PR (cheap) and the full eval nightly (scheduled workflow).

```python
# src/eval/gate.py
import json, sys

TOLERANCE = {"faithfulness": 0.03, "context_recall": 0.03,
             "numeric_accuracy": 0.02, "answer_relevance": 0.05}

def main(baseline_path: str, new_path: str) -> int:
    base = json.load(open(baseline_path))["aggregate"]
    new = json.load(open(new_path))["aggregate"]
    failures = [
        f"{m}: {new[m]:.3f} < {base[m] - tol:.3f}"
        for m, tol in TOLERANCE.items()
        if new[m] < base[m] - tol
    ]
    if failures:
        print("Eval regression:\n" + "\n".join(failures))
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
```

**DoD:** A PR that deliberately breaks retrieval fails CI on the gate.

---

## Phase 3: Observability and resilience

### 3.1 Structured logging and request IDs

- [ ] JSON logs (`structlog` or `python-json-logger`).
- [ ] Middleware sets `request_id` in a `contextvar`; include `thread_id`, `tier`, `provider`, `latency_ms` on every log line.

### 3.2 Tracing and metrics

- [ ] Add Langfuse (or LangSmith / OTel) callback handler to the graph invocation so each node is a span.
- [ ] Expose Prometheus metrics at `/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `finsight_requests_total` | counter | `status`, `tier` |
| `finsight_node_latency_seconds` | histogram | `node` |
| `finsight_llm_tokens_total` | counter | `provider`, `model`, `direction` |
| `finsight_llm_cost_usd_total` | counter | `provider`, `model` |
| `finsight_fallback_total` | counter | `from_provider`, `to_provider` |
| `finsight_verification_failures_total` | counter | `reason` |
| `finsight_hitl_total` | counter | `decision` |

**DoD:** One query produces a trace showing every node with latency and token counts.

### 3.3 Retries, timeouts, circuit breakers

- [ ] Wrap each provider call with a timeout, retries on transient errors only (429, 5xx, timeouts) and exponential backoff with jitter.
- [ ] Add a per-provider circuit breaker so a dead provider is skipped for a cool-down window instead of adding its timeout to every request.

```python
# src/resilience/retry.py
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

TRANSIENT = (TimeoutError, RateLimitError, ServiceUnavailableError)  # map to your SDK errors

def with_retry(fn):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(TRANSIENT),
        reraise=True,
    )(fn)
```

```python
# src/resilience/breaker.py
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=30):
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0   # half-open: try again
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
```

- [ ] Verify the tier-to-model mapping exists for **every** provider in the fallback chain, so a fallback never serves a much weaker model.
- [ ] Return a structured degraded response (`status="degraded"`, `reason`) instead of a bare `error` string.

**DoD:** Kill the primary provider key; requests still succeed through fallback within the latency budget, and metrics show the fallback.

---

## Phase 4: Financial accuracy

### 4.1 Pull structured numbers from SEC XBRL

- [ ] Build `src/sec/xbrl.py` using the SEC `companyfacts` endpoint (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`). SEC requires a descriptive `User-Agent` header and fair-use rate limiting (stay well under 10 requests/second).
- [ ] Cache responses in Redis (facts change only when new filings land).
- [ ] Normalize facts to `{concept, value, unit, period_start, period_end, fiscal_year, fiscal_period, form, accession}`.

```python
# src/sec/xbrl.py (sketch)
import httpx

HEADERS = {"User-Agent": "FinSight your-email@example.com"}

async def company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()

def latest_annual(facts: dict, concept: str, unit: str):
    entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    return max(annual, key=lambda e: e["end"], default=None)
```

### 4.2 Rebuild the P/E tool on structured data

- [ ] Replace "extract EPS from chunks" with TTM **diluted** EPS from XBRL (sum of the last four quarters, or the latest FY value when a 10-K is the newest filing).
- [ ] Extend `RatioResult` with `period_end`, `eps_basis` (`"diluted_ttm"`), `source` (`"xbrl"`), `accession`, and `as_of` for the price.
- [ ] Add unit and period guards: refuse to divide a quarterly EPS by an annual price multiple without flagging it.
- [ ] Handle negative or zero EPS: return `"not meaningful"` instead of a number.

**DoD:** P/E for AAPL matches a trusted source within tolerance, and the answer states the EPS period used.

### 4.3 Table-aware ingestion

- [ ] Extend `parse_edgar_file` to cover **Item 8** (financial statements) in addition to Items 1A and 7.
- [ ] Parse HTML `<table>` elements into structured rows. Serialize each table as a chunk that keeps header context, e.g. `"Net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)"`.
- [ ] Never split a table across chunks. Store `chunk_type` (`text` | `table`), `section`, `unit_scale` in metadata.

**DoD:** "What was Apple's total net sales in fiscal 2023" is answered from a table chunk and the cited source is that table.

---

## Phase 5: Retrieval and query understanding

### 5.1 Metadata schema and filters

- [ ] Add columns: `ticker`, `cik`, `form_type`, `fiscal_year`, `fiscal_period`, `section`, `chunk_type`, `accession`, `filed_at`.
- [ ] Add composite indexes on `(ticker, fiscal_year, form_type)` and an **HNSW** index on the embedding column.

```sql
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_meta_idx
  ON chunks (ticker, fiscal_year, form_type);
```

- [ ] Apply the filters as SQL pre-filters in the vector query **and** in the BM25 side of hybrid search, so both sides retrieve from the same candidate set.

### 5.2 Query understanding node

- [ ] Add a `parse_query` node before `retrieve` that extracts `ticker(s)`, `fiscal_year`, `form_type`, and `intent` (lookup, ratio, comparison, recommendation).
- [ ] For comparison questions, decompose into per-entity sub-queries, retrieve for each, then merge.
- [ ] Use an LLM with a Pydantic structured-output schema, with a regex fallback for obvious cases (cheap and fast).

### 5.3 Ingestion pipeline

- [ ] `src/sec/edgar_client.py`: fetch filing lists and documents, rate-limited, with retries.
- [ ] Idempotent upserts keyed by `(accession, chunk_index)`.
- [ ] Store `embedding_model` and `embedding_version` per row. On a model change, re-embed into a new column or table instead of mixing vector spaces.
- [ ] Introduce Alembic for schema migrations.

### 5.4 Retrieval experiments (run through the eval harness)

| Experiment | Change | Keep if |
|---|---|---|
| E1 | Section-aware + parent-child chunks vs fixed 1000/150 | Recall@k up |
| E2 | `bge-base` / `e5-base` / `text-embedding-3-small` vs MiniLM | Recall@k up, latency acceptable |
| E3 | Reranker top-N 20 vs 50 | Precision up, p95 within budget |
| E4 | Query rewriting / HyDE for vague queries | Recall up on hard tier |

- [ ] Make `compress` number-safe: it must never drop sentences containing figures that match the question's entities.

**DoD:** Each experiment has a row in an `EXPERIMENTS.md` table with before/after metrics.

---

## Phase 6: Verification and abstention

### 6.1 Numeric claim verification

- [ ] Add `src/verification/numeric.py`: extract every number from the answer (normalizing `$383.3 billion`, `383,285`, `24.5%`), then confirm each appears in a cited chunk or a tool result within tolerance and with matching units.
- [ ] Treat any ungrounded number as a failed claim so the existing retry loop (max 2) triggers.

```python
import re

NUM = re.compile(r"\$?\s?(\d[\d,]*\.?\d*)\s?(billion|million|thousand|%|bn|m|k)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}

def normalize(text: str) -> list[float]:
    out = []
    for val, unit in NUM.findall(text):
        v = float(val.replace(",", ""))
        out.append(v * SCALE.get((unit or "").lower(), 1.0))
    return out

def numbers_grounded(answer: str, evidence: str, rel_tol=0.005) -> list[float]:
    """Return numbers in the answer that are NOT supported by the evidence."""
    ev = normalize(evidence)
    return [a for a in normalize(answer)
            if not any(abs(a - e) <= rel_tol * max(abs(e), 1) for e in ev)]
```

> Note: scale handling needs care (tables often report "in millions"). Use the `unit_scale` metadata from 4.3 when comparing.

### 6.2 Semantic claim verification

- [ ] If `verify_citations` is lexical, upgrade it: split the answer into claims, then check entailment against the cited chunks with an NLI cross-encoder or an LLM judge. Use the cheap model tier for this.

### 6.3 Abstention

- [ ] Return `"I can't answer this from the available filings"` when: top rerank score is below a tuned threshold, verification still fails after retries, or the requested ticker or year is not in the corpus.
- [ ] Never ship an unverified answer silently. Return `verification.passed=false` plus a caveat in the response.

**DoD:** Unanswerable and out-of-corpus eval questions get abstentions, not guesses.

---

## Phase 7: Security and compliance

- [ ] **Auth:** API-key dependency (hashed keys in Postgres) or JWT. Per-key rate limits (`slowapi` or Redis token bucket).
- [ ] **Input limits:** max query length, max concurrent requests per key, request timeout.
- [ ] **Prompt injection:** treat retrieved text and tool output as untrusted. Wrap it in delimiters, instruct the model to ignore instructions inside it, and test with injected filings in your eval set.
- [ ] **Secrets:** no secrets in the image or repo. Add `detect-secrets` to pre-commit and CI, and use env or a secrets manager in deploys.
- [ ] **Dependencies:** `pip-audit` in CI, Dependabot enabled.
- [ ] **Compliance:** fixed "not investment advice" template on all recommendation outputs. Replace keyword-based `is_recommendation` with a classifier (LLM or fine-tuned small model) and track its precision and recall in the eval set.
- [ ] **Audit log:** table `hitl_audit(thread_id, reviewer, decision, note, draft_hash, created_at)`. Log every decision.
- [ ] **PII:** make sure logs and traces don't capture user identifiers beyond what you need. Add a retention policy.

**DoD:** Unauthenticated requests get 401, burst traffic gets 429, and an injected "ignore previous instructions" filing snippet doesn't change the answer.

---

## Phase 8: Eval expansion, UI and polish

### 8.1 Eval

- [ ] Grow the dataset from 30 to 100+ questions, balanced across tiers, with **gold chunk IDs** so you can report Recall@k, MRR and nDCG separately from generation quality.
- [ ] Add numeric exact-match accuracy (tolerance and unit-aware).
- [ ] Add negative cases: wrong year, unknown ticker, out-of-scope, prompt injection.
- [ ] Use a real LLM judge from a **different** model than the generator. Report whether the heuristic fallback was used on each run so inflated scores are visible.
- [ ] Report per-tier and per-intent breakdowns, plus cost and p50/p95 latency per run.

### 8.2 Load and latency

- [ ] `locust` or `k6` script: 20 concurrent users for 5 minutes. Record p50/p95/p99, error rate, and the fallback rate.
- [ ] Set an SLO (for example p95 < 8s for `normal` tier) and alert on it.

### 8.3 Cost controls

- [ ] Per-request max tokens and max cost guard.
- [ ] Semantic answer cache (embed the query, return a cached answer above a similarity threshold, scoped by ticker and year).
- [ ] Use provider prompt caching where supported.

### 8.4 Conversation memory

- [ ] Multi-turn support keyed by `thread_id`, with follow-up resolution ("what about 2022?") before retrieval.

### 8.5 UI and repo polish

- [ ] Streamlit or Next.js UI: chat, streaming, clickable citations that expand the source passage, a reviewer page for pending HITL items.
- [ ] README: add description and topics, an architecture diagram, a Quickstart (`docker compose up`), an API reference, an eval results table, and a **Known limitations** section.
- [ ] Add `LICENSE`, `CONTRIBUTING.md`, a `Makefile` (`make up`, `make test`, `make eval`, `make seed`), and tag a `v0.1.0` release.

---

## Master checklist

**Phase 1: Service**
- [ ] FastAPI app and schemas
- [ ] HITL via `interrupt` and `/reviews`
- [ ] Dockerfile and full compose stack

**Phase 2: CI/CD**
- [ ] Lint, type-check, test, coverage in CI
- [ ] Eval regression gate (PR smoke + nightly full)

**Phase 3: Observability and resilience**
- [ ] JSON logs with request IDs
- [ ] Tracing (Langfuse / OTel)
- [ ] Prometheus metrics
- [ ] Retries, timeouts, circuit breakers
- [ ] Degraded-mode responses

**Phase 4: Financial accuracy**
- [ ] XBRL client with caching
- [ ] P/E tool rebuilt on TTM diluted EPS
- [ ] Item 8 + table-aware chunking

**Phase 5: Retrieval**
- [ ] Metadata schema, HNSW index, SQL pre-filters
- [ ] `parse_query` node and comparison decomposition
- [ ] EDGAR ingestion with idempotent upserts and Alembic
- [ ] Retrieval experiments logged in `EXPERIMENTS.md`

**Phase 6: Verification**
- [ ] Numeric claim checker
- [ ] NLI or LLM claim verification
- [ ] Abstention logic

**Phase 7: Security**
- [ ] Auth and rate limiting
- [ ] Prompt-injection defenses and tests
- [ ] Secret scanning and dependency audit
- [ ] Compliance template and audit log

**Phase 8: Polish**
- [ ] Eval set 100+ with gold chunk IDs and negatives
- [ ] Load test and SLOs
- [ ] Cost guards and semantic cache
- [ ] Multi-turn memory
- [ ] UI# FinSight: Production Readiness Implementation Plan

> Based on a review of the repo README (source files not inspected). File paths below are suggestions that follow your existing `src/` layout. Adjust to match your code.

## How to use this plan

- Work in phases. Each phase ends in something shippable and demo-able.
- Every task has a **Definition of Done (DoD)**. Do not tick a box until the DoD is met.
- Open one PR per task and let CI (Phase 2) guard everything after it.

| Phase | Theme | Est. effort | Outcome |
|---|---|---|---|
| 1 | Service layer (API + Docker + HITL) | 3-4 days | App runs as a service |
| 2 | CI/CD + eval regression gate | 2 days | Quality can't silently regress |
| 3 | Observability + resilience | 3 days | You can see and survive failures |
| 4 | Financial accuracy (XBRL, tables, P/E) | 5-6 days | Numbers are trustworthy |
| 5 | Retrieval + query understanding | 4-5 days | Right filing, right year, right section |
| 6 | Verification + abstention | 3 days | Wrong numbers get caught |
| 7 | Security + compliance | 3 days | Safe to expose |
| 8 | Eval expansion + UI + polish | 4 days | Demo-ready, measurable |

---

## Target architecture

```
Client (UI / curl)
   |
   v
FastAPI  --(auth, rate limit, request-id)-->  LangGraph app
   |                                             |
   |   /ask  /ask/stream  /review/{id}           |-- classify -> retrieve -> compress
   |   /health  /ready  /metrics                 |-- call_llm (+tools) -> verify
   |                                             |-- human_review_gate (interrupt)
   v                                             v
Postgres (pgvector + metadata)   Redis (cache + checkpoints)   LLM providers
        ^
        |
Ingestion worker (EDGAR fetcher + XBRL facts)
```

Proposed new layout:

```
src/
  api/
    app.py            # FastAPI factory, lifespan, middleware
    routes.py         # /ask, /ask/stream, /review, /health, /ready
    schemas.py        # request/response Pydantic models
    auth.py           # API key / JWT dependency
    ratelimit.py
  observability/
    logging.py        # structured JSON logs + request_id contextvar
    tracing.py        # Langfuse / OTel setup
    metrics.py        # Prometheus counters/histograms
  resilience/
    retry.py          # tenacity wrappers
    breaker.py        # per-provider circuit breaker
  sec/
    edgar_client.py   # rate-limited EDGAR fetcher
    xbrl.py           # companyfacts API client
  verification/
    numeric.py        # numeric claim checker
Dockerfile
Makefile
.github/workflows/ci.yml
```

---

## Phase 1: Service layer

### 1.1 FastAPI app

- [ ] Create `src/api/app.py` with an app factory and a `lifespan` that builds the graph once (not per request).
- [ ] Endpoints:
  - `POST /v1/ask` returns `{answer, sources, tier, provider, verification, thread_id, status}`
  - `GET /v1/ask/stream` streams node progress and tokens via SSE
  - `GET /v1/reviews/pending` and `POST /v1/reviews/{thread_id}` for HITL
  - `GET /health` (liveness) and `GET /ready` (checks Postgres, Redis, at least one LLM provider key)
- [ ] Define strict request models: `query` max length (e.g. 1000 chars), `k` bounded 1-20.

```python
# src/api/schemas.py
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z.\-]{1,10}$")
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    thread_id: str | None = None

class AskResponse(BaseModel):
    thread_id: str
    status: str            # "completed" | "pending_review" | "degraded"
    answer: str | None
    sources: list[str]
    provider: str | None
    tier: str | None
    verification: dict | None
```

**DoD:** `curl` against `/v1/ask` returns a cited answer; `/ready` returns 503 when Postgres is down.

### 1.2 HITL as a real workflow

- [ ] Replace the CLI approval with LangGraph `interrupt` so the graph pauses and resumes by `thread_id`.
- [ ] Persist pending state in the Redis checkpointer (already in your design). Fail loudly if Redis is configured but unreachable in production mode, instead of silently falling back to `MemorySaver`.

```python
# in human_review_gate node
from langgraph.types import interrupt

def human_review_gate(state: GraphState) -> dict:
    if not state["is_recommendation"]:
        return {"hitl_approved": True}
    decision = interrupt({
        "draft_answer": state["draft_answer"],
        "reason": "investment_recommendation",
    })
    return {"hitl_approved": bool(decision.get("approved")),
            "reviewer_note": decision.get("note")}

# in POST /v1/reviews/{thread_id}
from langgraph.types import Command
graph.invoke(Command(resume={"approved": body.approved, "note": body.note}),
             config={"configurable": {"thread_id": thread_id}})
```

**DoD:** Ask a recommendation question, restart the API container, then approve via `/reviews`; the answer completes.

### 1.3 Docker

- [ ] Multi-stage `Dockerfile` using `uv` (you already have `uv.lock`), non-root user, `HEALTHCHECK`.
- [ ] Extend `docker-compose.yml` with `api`, `postgres` (pgvector image), `redis`, and an optional `ingest` profile.
- [ ] Pre-download sentence-transformers models at build time so cold start doesn't hit the network.

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER app
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**DoD:** `docker compose up` brings up the full stack; `make seed` loads data.

---

## Phase 2: CI/CD and eval regression gate

### 2.1 CI workflow

- [ ] Add `ruff`, `mypy`, `pytest --cov` to `pyproject.toml` dev dependencies.
- [ ] Add pre-commit hooks (ruff, ruff-format, detect-secrets).

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: finsight }
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready --health-interval 5s --health-retries 10
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov=src --cov-fail-under=70
```

### 2.2 Eval regression gate

- [ ] Write `src/eval/gate.py` that compares a fresh run against `data/eval/baseline.json` and exits non-zero if any metric drops by more than a tolerance.
- [ ] Run a small smoke subset on every PR (cheap) and the full eval nightly (scheduled workflow).

```python
# src/eval/gate.py
import json, sys

TOLERANCE = {"faithfulness": 0.03, "context_recall": 0.03,
             "numeric_accuracy": 0.02, "answer_relevance": 0.05}

def main(baseline_path: str, new_path: str) -> int:
    base = json.load(open(baseline_path))["aggregate"]
    new = json.load(open(new_path))["aggregate"]
    failures = [
        f"{m}: {new[m]:.3f} < {base[m] - tol:.3f}"
        for m, tol in TOLERANCE.items()
        if new[m] < base[m] - tol
    ]
    if failures:
        print("Eval regression:\n" + "\n".join(failures))
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
```

**DoD:** A PR that deliberately breaks retrieval fails CI on the gate.

---

## Phase 3: Observability and resilience

### 3.1 Structured logging and request IDs

- [ ] JSON logs (`structlog` or `python-json-logger`).
- [ ] Middleware sets `request_id` in a `contextvar`; include `thread_id`, `tier`, `provider`, `latency_ms` on every log line.

### 3.2 Tracing and metrics

- [ ] Add Langfuse (or LangSmith / OTel) callback handler to the graph invocation so each node is a span.
- [ ] Expose Prometheus metrics at `/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `finsight_requests_total` | counter | `status`, `tier` |
| `finsight_node_latency_seconds` | histogram | `node` |
| `finsight_llm_tokens_total` | counter | `provider`, `model`, `direction` |
| `finsight_llm_cost_usd_total` | counter | `provider`, `model` |
| `finsight_fallback_total` | counter | `from_provider`, `to_provider` |
| `finsight_verification_failures_total` | counter | `reason` |
| `finsight_hitl_total` | counter | `decision` |

**DoD:** One query produces a trace showing every node with latency and token counts.

### 3.3 Retries, timeouts, circuit breakers

- [ ] Wrap each provider call with a timeout, retries on transient errors only (429, 5xx, timeouts) and exponential backoff with jitter.
- [ ] Add a per-provider circuit breaker so a dead provider is skipped for a cool-down window instead of adding its timeout to every request.

```python
# src/resilience/retry.py
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

TRANSIENT = (TimeoutError, RateLimitError, ServiceUnavailableError)  # map to your SDK errors

def with_retry(fn):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(TRANSIENT),
        reraise=True,
    )(fn)
```

```python
# src/resilience/breaker.py
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=30):
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0   # half-open: try again
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
```

- [ ] Verify the tier-to-model mapping exists for **every** provider in the fallback chain, so a fallback never serves a much weaker model.
- [ ] Return a structured degraded response (`status="degraded"`, `reason`) instead of a bare `error` string.

**DoD:** Kill the primary provider key; requests still succeed through fallback within the latency budget, and metrics show the fallback.

---

## Phase 4: Financial accuracy

### 4.1 Pull structured numbers from SEC XBRL

- [ ] Build `src/sec/xbrl.py` using the SEC `companyfacts` endpoint (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`). SEC requires a descriptive `User-Agent` header and fair-use rate limiting (stay well under 10 requests/second).
- [ ] Cache responses in Redis (facts change only when new filings land).
- [ ] Normalize facts to `{concept, value, unit, period_start, period_end, fiscal_year, fiscal_period, form, accession}`.

```python
# src/sec/xbrl.py (sketch)
import httpx

HEADERS = {"User-Agent": "FinSight your-email@example.com"}

async def company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()

def latest_annual(facts: dict, concept: str, unit: str):
    entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    return max(annual, key=lambda e: e["end"], default=None)
```

### 4.2 Rebuild the P/E tool on structured data

- [ ] Replace "extract EPS from chunks" with TTM **diluted** EPS from XBRL (sum of the last four quarters, or the latest FY value when a 10-K is the newest filing).
- [ ] Extend `RatioResult` with `period_end`, `eps_basis` (`"diluted_ttm"`), `source` (`"xbrl"`), `accession`, and `as_of` for the price.
- [ ] Add unit and period guards: refuse to divide a quarterly EPS by an annual price multiple without flagging it.
- [ ] Handle negative or zero EPS: return `"not meaningful"` instead of a number.

**DoD:** P/E for AAPL matches a trusted source within tolerance, and the answer states the EPS period used.

### 4.3 Table-aware ingestion

- [ ] Extend `parse_edgar_file` to cover **Item 8** (financial statements) in addition to Items 1A and 7.
- [ ] Parse HTML `<table>` elements into structured rows. Serialize each table as a chunk that keeps header context, e.g. `"Net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)"`.
- [ ] Never split a table across chunks. Store `chunk_type` (`text` | `table`), `section`, `unit_scale` in metadata.

**DoD:** "What was Apple's total net sales in fiscal 2023" is answered from a table chunk and the cited source is that table.

---

## Phase 5: Retrieval and query understanding

### 5.1 Metadata schema and filters

- [ ] Add columns: `ticker`, `cik`, `form_type`, `fiscal_year`, `fiscal_period`, `section`, `chunk_type`, `accession`, `filed_at`.
- [ ] Add composite indexes on `(ticker, fiscal_year, form_type)` and an **HNSW** index on the embedding column.

```sql
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_meta_idx
  ON chunks (ticker, fiscal_year, form_type);
```

- [ ] Apply the filters as SQL pre-filters in the vector query **and** in the BM25 side of hybrid search, so both sides retrieve from the same candidate set.

### 5.2 Query understanding node

- [ ] Add a `parse_query` node before `retrieve` that extracts `ticker(s)`, `fiscal_year`, `form_type`, and `intent` (lookup, ratio, comparison, recommendation).
- [ ] For comparison questions, decompose into per-entity sub-queries, retrieve for each, then merge.
- [ ] Use an LLM with a Pydantic structured-output schema, with a regex fallback for obvious cases (cheap and fast).

### 5.3 Ingestion pipeline

- [ ] `src/sec/edgar_client.py`: fetch filing lists and documents, rate-limited, with retries.
- [ ] Idempotent upserts keyed by `(accession, chunk_index)`.
- [ ] Store `embedding_model` and `embedding_version` per row. On a model change, re-embed into a new column or table instead of mixing vector spaces.
- [ ] Introduce Alembic for schema migrations.

### 5.4 Retrieval experiments (run through the eval harness)

| Experiment | Change | Keep if |
|---|---|---|
| E1 | Section-aware + parent-child chunks vs fixed 1000/150 | Recall@k up |
| E2 | `bge-base` / `e5-base` / `text-embedding-3-small` vs MiniLM | Recall@k up, latency acceptable |
| E3 | Reranker top-N 20 vs 50 | Precision up, p95 within budget |
| E4 | Query rewriting / HyDE for vague queries | Recall up on hard tier |

- [ ] Make `compress` number-safe: it must never drop sentences containing figures that match the question's entities.

**DoD:** Each experiment has a row in an `EXPERIMENTS.md` table with before/after metrics.

---

## Phase 6: Verification and abstention

### 6.1 Numeric claim verification

- [ ] Add `src/verification/numeric.py`: extract every number from the answer (normalizing `$383.3 billion`, `383,285`, `24.5%`), then confirm each appears in a cited chunk or a tool result within tolerance and with matching units.
- [ ] Treat any ungrounded number as a failed claim so the existing retry loop (max 2) triggers.

```python
import re

NUM = re.compile(r"\$?\s?(\d[\d,]*\.?\d*)\s?(billion|million|thousand|%|bn|m|k)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}

def normalize(text: str) -> list[float]:
    out = []
    for val, unit in NUM.findall(text):
        v = float(val.replace(",", ""))
        out.append(v * SCALE.get((unit or "").lower(), 1.0))
    return out

def numbers_grounded(answer: str, evidence: str, rel_tol=0.005) -> list[float]:
    """Return numbers in the answer that are NOT supported by the evidence."""
    ev = normalize(evidence)
    return [a for a in normalize(answer)
            if not any(abs(a - e) <= rel_tol * max(abs(e), 1) for e in ev)]
```

> Note: scale handling needs care (tables often report "in millions"). Use the `unit_scale` metadata from 4.3 when comparing.

### 6.2 Semantic claim verification

- [ ] If `verify_citations` is lexical, upgrade it: split the answer into claims, then check entailment against the cited chunks with an NLI cross-encoder or an LLM judge. Use the cheap model tier for this.

### 6.3 Abstention

- [ ] Return `"I can't answer this from the available filings"` when: top rerank score is below a tuned threshold, verification still fails after retries, or the requested ticker or year is not in the corpus.
- [ ] Never ship an unverified answer silently. Return `verification.passed=false` plus a caveat in the response.

**DoD:** Unanswerable and out-of-corpus eval questions get abstentions, not guesses.

---

## Phase 7: Security and compliance

- [ ] **Auth:** API-key dependency (hashed keys in Postgres) or JWT. Per-key rate limits (`slowapi` or Redis token bucket).
- [ ] **Input limits:** max query length, max concurrent requests per key, request timeout.
- [ ] **Prompt injection:** treat retrieved text and tool output as untrusted. Wrap it in delimiters, instruct the model to ignore instructions inside it, and test with injected filings in your eval set.
- [ ] **Secrets:** no secrets in the image or repo. Add `detect-secrets` to pre-commit and CI, and use env or a secrets manager in deploys.
- [ ] **Dependencies:** `pip-audit` in CI, Dependabot enabled.
- [ ] **Compliance:** fixed "not investment advice" template on all recommendation outputs. Replace keyword-based `is_recommendation` with a classifier (LLM or fine-tuned small model) and track its precision and recall in the eval set.
- [ ] **Audit log:** table `hitl_audit(thread_id, reviewer, decision, note, draft_hash, created_at)`. Log every decision.
- [ ] **PII:** make sure logs and traces don't capture user identifiers beyond what you need. Add a retention policy.

**DoD:** Unauthenticated requests get 401, burst traffic gets 429, and an injected "ignore previous instructions" filing snippet doesn't change the answer.

---

## Phase 8: Eval expansion, UI and polish

### 8.1 Eval

- [ ] Grow the dataset from 30 to 100+ questions, balanced across tiers, with **gold chunk IDs** so you can report Recall@k, MRR and nDCG separately from generation quality.
- [ ] Add numeric exact-match accuracy (tolerance and unit-aware).
- [ ] Add negative cases: wrong year, unknown ticker, out-of-scope, prompt injection.
- [ ] Use a real LLM judge from a **different** model than the generator. Report whether the heuristic fallback was used on each run so inflated scores are visible.
- [ ] Report per-tier and per-intent breakdowns, plus cost and p50/p95 latency per run.

### 8.2 Load and latency

- [ ] `locust` or `k6` script: 20 concurrent users for 5 minutes. Record p50/p95/p99, error rate, and the fallback rate.
- [ ] Set an SLO (for example p95 < 8s for `normal` tier) and alert on it.

### 8.3 Cost controls

- [ ] Per-request max tokens and max cost guard.
- [ ] Semantic answer cache (embed the query, return a cached answer above a similarity threshold, scoped by ticker and year).
- [ ] Use provider prompt caching where supported.

### 8.4 Conversation memory

- [ ] Multi-turn support keyed by `thread_id`, with follow-up resolution ("what about 2022?") before retrieval.

### 8.5 UI and repo polish

- [ ] Streamlit or Next.js UI: chat, streaming, clickable citations that expand the source passage, a reviewer page for pending HITL items.
- [ ] README: add description and topics, an architecture diagram, a Quickstart (`docker compose up`), an API reference, an eval results table, and a **Known limitations** section.
- [ ] Add `LICENSE`, `CONTRIBUTING.md`, a `Makefile` (`make up`, `make test`, `make eval`, `make seed`), and tag a `v0.1.0` release.

---

## Master checklist

**Phase 1: Service**
- [ ] FastAPI app and schemas
- [ ] HITL via `interrupt` and `/reviews`
- [ ] Dockerfile and full compose stack

**Phase 2: CI/CD**
- [ ] Lint, type-check, test, coverage in CI
- [ ] Eval regression gate (PR smoke + nightly full)

**Phase 3: Observability and resilience**
- [ ] JSON logs with request IDs
- [ ] Tracing (Langfuse / OTel)
- [ ] Prometheus metrics
- [ ] Retries, timeouts, circuit breakers
- [ ] Degraded-mode responses

**Phase 4: Financial accuracy**
- [ ] XBRL client with caching
- [ ] P/E tool rebuilt on TTM diluted EPS
- [ ] Item 8 + table-aware chunking

**Phase 5: Retrieval**
- [ ] Metadata schema, HNSW index, SQL pre-filters
- [ ] `parse_query` node and comparison decomposition
- [ ] EDGAR ingestion with idempotent upserts and Alembic
- [ ] Retrieval experiments logged in `EXPERIMENTS.md`

**Phase 6: Verification**
- [ ] Numeric claim checker
- [ ] NLI or LLM claim verification
- [ ] Abstention logic

**Phase 7: Security**
- [ ] Auth and rate limiting
- [ ] Prompt-injection defenses and tests
- [ ] Secret scanning and dependency audit
- [ ] Compliance template and audit log
# FinSight: Production Readiness Implementation Plan

> Based on a review of the repo README (source files not inspected). File paths below are suggestions that follow your existing `src/` layout. Adjust to match your code.

## How to use this plan

- Work in phases. Each phase ends in something shippable and demo-able.
- Every task has a **Definition of Done (DoD)**. Do not tick a box until the DoD is met.
- Open one PR per task and let CI (Phase 2) guard everything after it.

| Phase | Theme | Est. effort | Outcome |
|---|---|---|---|
| 1 | Service layer (API + Docker + HITL) | 3-4 days | App runs as a service |
| 2 | CI/CD + eval regression gate | 2 days | Quality can't silently regress |
| 3 | Observability + resilience | 3 days | You can see and survive failures |
| 4 | Financial accuracy (XBRL, tables, P/E) | 5-6 days | Numbers are trustworthy |
| 5 | Retrieval + query understanding | 4-5 days | Right filing, right year, right section |
| 6 | Verification + abstention | 3 days | Wrong numbers get caught |
| 7 | Security + compliance | 3 days | Safe to expose |
| 8 | Eval expansion + UI + polish | 4 days | Demo-ready, measurable |

---

## Target architecture

```
Client (UI / curl)
   |
   v
FastAPI  --(auth, rate limit, request-id)-->  LangGraph app
   |                                             |
   |   /ask  /ask/stream  /review/{id}           |-- classify -> retrieve -> compress
   |   /health  /ready  /metrics                 |-- call_llm (+tools) -> verify
   |                                             |-- human_review_gate (interrupt)
   v                                             v
Postgres (pgvector + metadata)   Redis (cache + checkpoints)   LLM providers
        ^
        |
Ingestion worker (EDGAR fetcher + XBRL facts)
```

Proposed new layout:

```
src/
  api/
    app.py            # FastAPI factory, lifespan, middleware
    routes.py         # /ask, /ask/stream, /review, /health, /ready
    schemas.py        # request/response Pydantic models
    auth.py           # API key / JWT dependency
    ratelimit.py
  observability/
    logging.py        # structured JSON logs + request_id contextvar
    tracing.py        # Langfuse / OTel setup
    metrics.py        # Prometheus counters/histograms
  resilience/
    retry.py          # tenacity wrappers
    breaker.py        # per-provider circuit breaker
  sec/
    edgar_client.py   # rate-limited EDGAR fetcher
    xbrl.py           # companyfacts API client
  verification/
    numeric.py        # numeric claim checker
Dockerfile
Makefile
.github/workflows/ci.yml
```

---

## Phase 1: Service layer

### 1.1 FastAPI app

- [ ] Create `src/api/app.py` with an app factory and a `lifespan` that builds the graph once (not per request).
- [ ] Endpoints:
  - `POST /v1/ask` returns `{answer, sources, tier, provider, verification, thread_id, status}`
  - `GET /v1/ask/stream` streams node progress and tokens via SSE
  - `GET /v1/reviews/pending` and `POST /v1/reviews/{thread_id}` for HITL
  - `GET /health` (liveness) and `GET /ready` (checks Postgres, Redis, at least one LLM provider key)
- [ ] Define strict request models: `query` max length (e.g. 1000 chars), `k` bounded 1-20.

```python
# src/api/schemas.py
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z.\-]{1,10}$")
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    thread_id: str | None = None

class AskResponse(BaseModel):
    thread_id: str
    status: str            # "completed" | "pending_review" | "degraded"
    answer: str | None
    sources: list[str]
    provider: str | None
    tier: str | None
    verification: dict | None
```

**DoD:** `curl` against `/v1/ask` returns a cited answer; `/ready` returns 503 when Postgres is down.

### 1.2 HITL as a real workflow

- [ ] Replace the CLI approval with LangGraph `interrupt` so the graph pauses and resumes by `thread_id`.
- [ ] Persist pending state in the Redis checkpointer (already in your design). Fail loudly if Redis is configured but unreachable in production mode, instead of silently falling back to `MemorySaver`.

```python
# in human_review_gate node
from langgraph.types import interrupt

def human_review_gate(state: GraphState) -> dict:
    if not state["is_recommendation"]:
        return {"hitl_approved": True}
    decision = interrupt({
        "draft_answer": state["draft_answer"],
        "reason": "investment_recommendation",
    })
    return {"hitl_approved": bool(decision.get("approved")),
            "reviewer_note": decision.get("note")}

# in POST /v1/reviews/{thread_id}
from langgraph.types import Command
graph.invoke(Command(resume={"approved": body.approved, "note": body.note}),
             config={"configurable": {"thread_id": thread_id}})
```

**DoD:** Ask a recommendation question, restart the API container, then approve via `/reviews`; the answer completes.

### 1.3 Docker

- [ ] Multi-stage `Dockerfile` using `uv` (you already have `uv.lock`), non-root user, `HEALTHCHECK`.
- [ ] Extend `docker-compose.yml` with `api`, `postgres` (pgvector image), `redis`, and an optional `ingest` profile.
- [ ] Pre-download sentence-transformers models at build time so cold start doesn't hit the network.

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER app
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**DoD:** `docker compose up` brings up the full stack; `make seed` loads data.

---

## Phase 2: CI/CD and eval regression gate

### 2.1 CI workflow

- [ ] Add `ruff`, `mypy`, `pytest --cov` to `pyproject.toml` dev dependencies.
- [ ] Add pre-commit hooks (ruff, ruff-format, detect-secrets).

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: finsight }
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready --health-interval 5s --health-retries 10
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov=src --cov-fail-under=70
```

### 2.2 Eval regression gate

- [ ] Write `src/eval/gate.py` that compares a fresh run against `data/eval/baseline.json` and exits non-zero if any metric drops by more than a tolerance.
- [ ] Run a small smoke subset on every PR (cheap) and the full eval nightly (scheduled workflow).

```python
# src/eval/gate.py
import json, sys

TOLERANCE = {"faithfulness": 0.03, "context_recall": 0.03,
             "numeric_accuracy": 0.02, "answer_relevance": 0.05}

def main(baseline_path: str, new_path: str) -> int:
    base = json.load(open(baseline_path))["aggregate"]
    new = json.load(open(new_path))["aggregate"]
    failures = [
        f"{m}: {new[m]:.3f} < {base[m] - tol:.3f}"
        for m, tol in TOLERANCE.items()
        if new[m] < base[m] - tol
    ]
    if failures:
        print("Eval regression:\n" + "\n".join(failures))
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
```

**DoD:** A PR that deliberately breaks retrieval fails CI on the gate.

---

## Phase 3: Observability and resilience

### 3.1 Structured logging and request IDs

- [ ] JSON logs (`structlog` or `python-json-logger`).
- [ ] Middleware sets `request_id` in a `contextvar`; include `thread_id`, `tier`, `provider`, `latency_ms` on every log line.

### 3.2 Tracing and metrics

- [ ] Add Langfuse (or LangSmith / OTel) callback handler to the graph invocation so each node is a span.
- [ ] Expose Prometheus metrics at `/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `finsight_requests_total` | counter | `status`, `tier` |
| `finsight_node_latency_seconds` | histogram | `node` |
| `finsight_llm_tokens_total` | counter | `provider`, `model`, `direction` |
| `finsight_llm_cost_usd_total` | counter | `provider`, `model` |
| `finsight_fallback_total` | counter | `from_provider`, `to_provider` |
| `finsight_verification_failures_total` | counter | `reason` |
| `finsight_hitl_total` | counter | `decision` |

**DoD:** One query produces a trace showing every node with latency and token counts.

### 3.3 Retries, timeouts, circuit breakers

- [ ] Wrap each provider call with a timeout, retries on transient errors only (429, 5xx, timeouts) and exponential backoff with jitter.
- [ ] Add a per-provider circuit breaker so a dead provider is skipped for a cool-down window instead of adding its timeout to every request.

```python
# src/resilience/retry.py
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

TRANSIENT = (TimeoutError, RateLimitError, ServiceUnavailableError)  # map to your SDK errors

def with_retry(fn):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(TRANSIENT),
        reraise=True,
    )(fn)
```

```python
# src/resilience/breaker.py
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=30):
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0   # half-open: try again
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
```

- [ ] Verify the tier-to-model mapping exists for **every** provider in the fallback chain, so a fallback never serves a much weaker model.
- [ ] Return a structured degraded response (`status="degraded"`, `reason`) instead of a bare `error` string.

**DoD:** Kill the primary provider key; requests still succeed through fallback within the latency budget, and metrics show the fallback.

---

## Phase 4: Financial accuracy

### 4.1 Pull structured numbers from SEC XBRL

- [ ] Build `src/sec/xbrl.py` using the SEC `companyfacts` endpoint (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`). SEC requires a descriptive `User-Agent` header and fair-use rate limiting (stay well under 10 requests/second).
- [ ] Cache responses in Redis (facts change only when new filings land).
- [ ] Normalize facts to `{concept, value, unit, period_start, period_end, fiscal_year, fiscal_period, form, accession}`.

```python
# src/sec/xbrl.py (sketch)
import httpx

HEADERS = {"User-Agent": "FinSight your-email@example.com"}

async def company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()

def latest_annual(facts: dict, concept: str, unit: str):
    entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    return max(annual, key=lambda e: e["end"], default=None)
```

### 4.2 Rebuild the P/E tool on structured data

- [ ] Replace "extract EPS from chunks" with TTM **diluted** EPS from XBRL (sum of the last four quarters, or the latest FY value when a 10-K is the newest filing).
- [ ] Extend `RatioResult` with `period_end`, `eps_basis` (`"diluted_ttm"`), `source` (`"xbrl"`), `accession`, and `as_of` for the price.
- [ ] Add unit and period guards: refuse to divide a quarterly EPS by an annual price multiple without flagging it.
- [ ] Handle negative or zero EPS: return `"not meaningful"` instead of a number.

**DoD:** P/E for AAPL matches a trusted source within tolerance, and the answer states the EPS period used.

### 4.3 Table-aware ingestion

- [ ] Extend `parse_edgar_file` to cover **Item 8** (financial statements) in addition to Items 1A and 7.
- [ ] Parse HTML `<table>` elements into structured rows. Serialize each table as a chunk that keeps header context, e.g. `"Net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)"`.
- [ ] Never split a table across chunks. Store `chunk_type` (`text` | `table`), `section`, `unit_scale` in metadata.

**DoD:** "What was Apple's total net sales in fiscal 2023" is answered from a table chunk and the cited source is that table.

---

## Phase 5: Retrieval and query understanding

### 5.1 Metadata schema and filters

- [ ] Add columns: `ticker`, `cik`, `form_type`, `fiscal_year`, `fiscal_period`, `section`, `chunk_type`, `accession`, `filed_at`.
- [ ] Add composite indexes on `(ticker, fiscal_year, form_type)` and an **HNSW** index on the embedding column.

```sql
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_meta_idx
  ON chunks (ticker, fiscal_year, form_type);
```

- [ ] Apply the filters as SQL pre-filters in the vector query **and** in the BM25 side of hybrid search, so both sides retrieve from the same candidate set.

### 5.2 Query understanding node

- [ ] Add a `parse_query` node before `retrieve` that extracts `ticker(s)`, `fiscal_year`, `form_type`, and `intent` (lookup, ratio, comparison, recommendation).
- [ ] For comparison questions, decompose into per-entity sub-queries, retrieve for each, then merge.
- [ ] Use an LLM with a Pydantic structured-output schema, with a regex fallback for obvious cases (cheap and fast).

### 5.3 Ingestion pipeline

- [ ] `src/sec/edgar_client.py`: fetch filing lists and documents, rate-limited, with retries.
- [ ] Idempotent upserts keyed by `(accession, chunk_index)`.
- [ ] Store `embedding_model` and `embedding_version` per row. On a model change, re-embed into a new column or table instead of mixing vector spaces.
- [ ] Introduce Alembic for schema migrations.

### 5.4 Retrieval experiments (run through the eval harness)

| Experiment | Change | Keep if |
|---|---|---|
| E1 | Section-aware + parent-child chunks vs fixed 1000/150 | Recall@k up |
| E2 | `bge-base` / `e5-base` / `text-embedding-3-small` vs MiniLM | Recall@k up, latency acceptable |
| E3 | Reranker top-N 20 vs 50 | Precision up, p95 within budget |
| E4 | Query rewriting / HyDE for vague queries | Recall up on hard tier |

- [ ] Make `compress` number-safe: it must never drop sentences containing figures that match the question's entities.

**DoD:** Each experiment has a row in an `EXPERIMENTS.md` table with before/after metrics.

---

## Phase 6: Verification and abstention

### 6.1 Numeric claim verification

- [ ] Add `src/verification/numeric.py`: extract every number from the answer (normalizing `$383.3 billion`, `383,285`, `24.5%`), then confirm each appears in a cited chunk or a tool result within tolerance and with matching units.
- [ ] Treat any ungrounded number as a failed claim so the existing retry loop (max 2) triggers.

```python
import re

NUM = re.compile(r"\$?\s?(\d[\d,]*\.?\d*)\s?(billion|million|thousand|%|bn|m|k)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}

def normalize(text: str) -> list[float]:
    out = []
    for val, unit in NUM.findall(text):
        v = float(val.replace(",", ""))
        out.append(v * SCALE.get((unit or "").lower(), 1.0))
    return out

def numbers_grounded(answer: str, evidence: str, rel_tol=0.005) -> list[float]:
    """Return numbers in the answer that are NOT supported by the evidence."""
    ev = normalize(evidence)
    return [a for a in normalize(answer)
            if not any(abs(a - e) <= rel_tol * max(abs(e), 1) for e in ev)]
```

> Note: scale handling needs care (tables often report "in millions"). Use the `unit_scale` metadata from 4.3 when comparing.

### 6.2 Semantic claim verification

- [ ] If `verify_citations` is lexical, upgrade it: split the answer into claims, then check entailment against the cited chunks with an NLI cross-encoder or an LLM judge. Use the cheap model tier for this.

### 6.3 Abstention

- [ ] Return `"I can't answer this from the available filings"` when: top rerank score is below a tuned threshold, verification still fails after retries, or the requested ticker or year is not in the corpus.
- [ ] Never ship an unverified answer silently. Return `verification.passed=false` plus a caveat in the response.

**DoD:** Unanswerable and out-of-corpus eval questions get abstentions, not guesses.

---

## Phase 7: Security and compliance

- [ ] **Auth:** API-key dependency (hashed keys in Postgres) or JWT. Per-key rate limits (`slowapi` or Redis token bucket).
- [ ] **Input limits:** max query length, max concurrent requests per key, request timeout.
- [ ] **Prompt injection:** treat retrieved text and tool output as untrusted. Wrap it in delimiters, instruct the model to ignore instructions inside it, and test with injected filings in your eval set.
- [ ] **Secrets:** no secrets in the image or repo. Add `detect-secrets` to pre-commit and CI, and use env or a secrets manager in deploys.
- [ ] **Dependencies:** `pip-audit` in CI, Dependabot enabled.
- [ ] **Compliance:** fixed "not investment advice" template on all recommendation outputs. Replace keyword-based `is_recommendation` with a classifier (LLM or fine-tuned small model) and track its precision and recall in the eval set.
- [ ] **Audit log:** table `hitl_audit(thread_id, reviewer, decision, note, draft_hash, created_at)`. Log every decision.
- [ ] **PII:** make sure logs and traces don't capture user identifiers beyond what you need. Add a retention policy.

**DoD:** Unauthenticated requests get 401, burst traffic gets 429, and an injected "ignore previous instructions" filing snippet doesn't change the answer.

---

## Phase 8: Eval expansion, UI and polish

### 8.1 Eval

- [ ] Grow the dataset from 30 to 100+ questions, balanced across tiers, with **gold chunk IDs** so you can report Recall@k, MRR and nDCG separately from generation quality.
- [ ] Add numeric exact-match accuracy (tolerance and unit-aware).
- [ ] Add negative cases: wrong year, unknown ticker, out-of-scope, prompt injection.
- [ ] Use a real LLM judge from a **different** model than the generator. Report whether the heuristic fallback was used on each run so inflated scores are visible.
- [ ] Report per-tier and per-intent breakdowns, plus cost and p50/p95 latency per run.

### 8.2 Load and latency

- [ ] `locust` or `k6` script: 20 concurrent users for 5 minutes. Record p50/p95/p99, error rate, and the fallback rate.
- [ ] Set an SLO (for example p95 < 8s for `normal` tier) and alert on it.

### 8.3 Cost controls

- [ ] Per-request max tokens and max cost guard.
- [ ] Semantic answer cache (embed the query, return a cached answer above a similarity threshold, scoped by ticker and year).
- [ ] Use provider prompt caching where supported.

### 8.4 Conversation memory

- [ ] Multi-turn support keyed by `thread_id`, with follow-up resolution ("what about 2022?") before retrieval.

### 8.5 UI and repo polish

- [ ] Streamlit or Next.js UI: chat, streaming, clickable citations that expand the source passage, a reviewer page for pending HITL items.
- [ ] README: add description and topics, an architecture diagram, a Quickstart (`docker compose up`), an API reference, an eval results table, and a **Known limitations** section.
- [ ] Add `LICENSE`, `CONTRIBUTING.md`, a `Makefile` (`make up`, `make test`, `make eval`, `make seed`), and tag a `v0.1.0` release.

---

## Master checklist

**Phase 1: Service**
- [ ] FastAPI app and schemas
- [ ] HITL via `interrupt` and `/reviews`
- [ ] Dockerfile and full compose stack

**Phase 2: CI/CD**
- [ ] Lint, type-check, test, coverage in CI
- [ ] Eval regression gate (PR smoke + nightly full)

**Phase 3: Observability and resilience**
- [ ] JSON logs with request IDs
- [ ] Tracing (Langfuse / OTel)
- [ ] Prometheus metrics
- [ ] Retries, timeouts, circuit breakers
- [ ] Degraded-mode responses

**Phase 4: Financial accuracy**
- [ ] XBRL client with caching
- [ ] P/E tool rebuilt on TTM diluted EPS
- [ ] Item 8 + table-aware chunking

**Phase 5: Retrieval**
- [ ] Metadata schema, HNSW index, SQL pre-filters
- [ ] `parse_query` node and comparison decomposition
- [ ] EDGAR ingestion with idempotent upserts and Alembic
- [ ] Retrieval experiments logged in `EXPERIMENTS.md`

**Phase 6: Verification**
- [ ] Numeric claim checker
- [ ] NLI or LLM claim verification
- [ ] Abstention logic

**Phase 7: Security**
- [ ] Auth and rate limiting
- [ ] Prompt-injection defenses and tests
- [ ] Secret scanning and dependency audit
- [ ] Compliance template and audit log

**Phase 8: Polish**
- [ ] Eval set 100+ with gold chunk IDs and negatives
- [ ] Load test and SLOs
- [ ] Cost guards and semantic cache
- [ ] Multi-turn memory
- [ ] UI
- [ ] README, LICENSE, release tag

---

## Risks and notes

| Risk | Mitigation |
|---|---|
| Free-tier LLM rate limits slow evals | Cache judge calls, run the smoke subset on PRs and the full set nightly |
| XBRL concept names vary by company | Maintain a concept alias map (e.g. `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`) and test per ticker |
| Fiscal-year definitions differ from calendar year | Store `fiscal_year` from filing metadata, never derive it from dates |
| Changing the embedding model invalidates stored vectors | Version embeddings per row and re-index into a new table or column |
| Redis fallback to memory hides HITL state loss | Fail fast in production mode, allow fallback only in dev |

## Suggested resume bullets once done

- Built a production RAG service over SEC filings with hybrid retrieval, reranking, structured XBRL grounding and numeric claim verification.
- Designed an eval-gated CI pipeline that blocks regressions in faithfulness and numeric accuracy.
- Implemented provider fallback with circuit breakers, end-to-end tracing and cost and latency metrics.
- Implemented human-in-the-loop review for investment recommendations, with an audit trail.
**Phase 8: Polish**
- [ ] Eval set 100+ with gold chunk IDs and negatives
- [ ] Load test and SLOs
- [ ] Cost guards and semantic cache
- [ ] Multi-turn memory
- [ ] UI
- [ ] README, LICENSE, release tag

---

## Risks and notes

| Risk | Mitigation |
|---|---|
| Free-tier LLM rate limits slow evals | Cache judge calls, run the smoke subset on PRs and the full set nightly |
| XBRL concept names vary by company | Maintain a concept alias map (e.g. `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`) and test per ticker |
| Fiscal-year definitions differ from calendar year | Store `fiscal_year` from filing metadata, never derive it from dates |
| Changing the embedding model invalidates stored vectors | Version embeddings per row and re-index into a new table or column |
| Redis fallback to memory hides HITL state loss | Fail fast in production mode, allow fallback only in dev |

## Suggested resume bullets once done

- Built a production RAG service over SEC filings with hybrid retrieval, reranking, structured XBRL grounding and numeric claim verification.
- Designed an eval-gated CI pipeline that blocks regressions in faithfulness and numeric accuracy.
- Implemented provider fallback with circuit breakers, end-to-end tracing and cost and latency metrics.
- Implemented human-in-the-loop review for investment recommendations, with an audit trail.
- [ ] README, LICENSE, release tag

---

## Risks and notes

| Risk | Mitigation |
|---|---|
| Free-tier LLM rate limits slow evals | Cache judge calls, run the smoke subset on PRs and the full set nightly |
| XBRL concept names vary by company | Maintain a concept alias map (e.g. `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`) and test per ticker |
| Fiscal-year definitions differ from calendar year | Store `fiscal_year` from filing metadata, never derive it from dates |
| Changing the embedding model invalidates stored vectors | Version embeddings per row and re-index into a new table or column |
| Redis fallback to memory hides HITL state loss | Fail fast in production mode, allow fallback only in dev |

## Suggested resume bullets once done

- Built a production RAG service over SEC filings with hybrid retrieval, reranking, structured XBRL grounding and numeric claim verification.
- Designed an eval-gated CI pipeline that blocks regressions in faithfulness and numeric accuracy.
- Implemented provider fallback with circuit breakers, end-to-end tracing and cost and latency metrics.
- Implemented human-in-the-loop review for investment recommendations, with an audit trail.