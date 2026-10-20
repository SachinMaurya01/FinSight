# FinSight

Grounded RAG over SEC 10-K filings. The system retrieves evidence from filing text, answers with inline citations, falls back across LLM providers, verifies claims before returning them, and gates investment-advice-like output behind human review.


## Doc Refernce:

- Item 1   Business-
- Item 1A  Risk Factors
- Item 7   Management's Discussion and Analysis
- Item 8   Financial Statements
- Item 7A  Market Risk

**Reference implementation, not a production deployment.** The pipeline and controls are real; the corpus, evaluation coverage, and operational hardening are not yet production-grade. This README separates what is measured from what is assumed.

**Links:** 
- [Architecture](docs/architecture.svg)
- [CLI demo](docs/demo.svg)
- [Results](#evaluation-results)
- [Failure analysis](#failure-analysis)

**Contents:** 
- [Scope](#scope)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Production readiness](#production-readiness)
- [Evaluation](#evaluation-results)
- [Failure analysis](#failure-analysis)
- [Security](#security-and-compliance)
- [Operations](#operations)
- [Roadmap](#roadmap)

## Executive summary

- **Problem:** 10-K answers are buried in long, table-heavy filings; generic chat-with-PDF tools lose citation traceability and numeric precision.
- **Approach:** section-scoped ingestion, hybrid retrieval (dense + BM25 with RRF), cross-encoder reranking, tiered model routing, provider fallback, citation verification, and HITL for recommendation-like output.
- **Current evidence:** offline retrieval ablation on 28 citation-labeled questions. Hybrid + rerank reaches `MRR@8 = 0.9464` vs `0.8929` for dense-only; context precision is `0.7589` vs `0.7321`.
- **Current gap:** full 30-question RAGAS is not measured. The checked-in end-to-end baseline is a 3-question heuristic smoke run. RAGAS is wired behind `--metrics ragas` but the current dependency lock cannot import it.
- **Production status:** local reference stack only. No API, auth, rate limiting, observability, or CI deployment gate yet.

## Scope

| In scope | Out of scope (today) |
|---|---|
| 10-K HTML ingestion for Item 1A and Item 7 | Full 10-K coverage, including Item 8 financial statements |
| Text chunks with ticker/period/section metadata | XBRL-native numeric extraction and table-structure preservation |
| Dense + BM25 hybrid retrieval + reranking | Peer comparison, 10-Q/8-K/proxies/transcripts |
| Provider routing and fallback | Production auth, multi-tenant isolation, rate limiting |
| Citation verification + bounded retry | NLI-grade entailment and numeric normalization |
| HITL gate for recommendation-like output | Regulatory advice, portfolio execution, or investment recommendations |
| Offline retrieval ablation + eval harness | Hosted API/UI, SLO dashboards, on-call runbooks |

## Architecture

![FinSight architecture](docs/architecture.svg)

**Offline:** `data/*.html` → `parse_edgar_file` (Item 1A, Item 7) → `chunk_document` (1000/150, metadata) → `embed_chunks` (sentence-transformers, 384d) → pgvector.

**Online:** `ask()` → classify complexity → retrieve (dense top-20 + BM25 top-20 → RRF → rerank top-8) → compress → LLM (tier routing + fallback + tools) → verify citations (retry on failure) → human review gate → final answer + sources.

**Request invariants**

1. The `src.graph` orchestration path always executes `verify_citations` before `finalize`. The lower-level `src.llm answer` CLI is a debugging path and bypasses verification and HITL.
2. Recommendation-like output pauses at the HITL gate by default. `--no-hitl` and the non-interactive `ask()` helper auto-approve; both are local/eval escape hatches, not production-safe controls.
3. All graph LLM calls go through the provider fallback chain; failures are logged with provider and reason.
4. Retrieved evidence carries `ticker`, `filing_type`, `fiscal_period`, `section`, and `chunk_index` for citation and filtering.

## Design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Hybrid retrieval with weighted RRF | Dense handles paraphrases; BM25 handles exact figures, tickers, and rare terms | Lexical matches can crowd out relevant narrative chunks (`eval_013`) |
| Cross-encoder rerank on fused candidates | Improves ordering once candidates are in the pool | Adds latency and cannot recover chunks missing from the candidate set |
| Complexity-based model routing | Controls cost/latency on simple queries | Routing accuracy is not yet measured against labeled queries |
| Provider fallback chain | Availability under rate limits/timeouts/auth failures | A fallback provider can change answer style/quality; no cross-provider consistency test |
| Citation verification before return | Blocks ungrounded claims from reaching users | Current check is keyword overlap, not entailment |
| HITL gate for recommendation-like output | Compliance-relevant control, not a model feature | Non-interactive `ask()` auto-approves for evaluation; production must not use that path |
| Redis cache + checkpointer with memory fallback | Lower latency and HITL resume across restarts | Memory fallback silently loses state across process restarts (logged) |
| Local embeddings by default | Offline reproducibility and no embedding API cost | 384d MiniLM is a quality ceiling; not benchmarked against stronger embedders |

## Production readiness

| Area | Status | Notes |
|---|---|---|
| Retrieval pipeline | Implemented | Dense, BM25, RRF, rerank, compression |
| LLM reliability | Implemented | Tier routing + fallback chain + degraded-mode error |
| Citation verification | Implemented, heuristic | Keyword overlap; NLI/numeric checks are future work |
| HITL control | Implemented, default-on | CLI gate + Redis checkpointer; `--no-hitl` and `ask()` auto-approval must be guarded in production |
| Evaluation | Partial | Retrieval measured on 28 questions; end-to-end n=3; RAGAS blocked |
| API / UI | Not included | CLI only |
| AuthN / AuthZ | Not included | No tenant model or access control |
| Rate limiting / quotas | Not included | Required before exposing an API |
| Observability | Deferred | Phase 2 LGTM stack; currently logs + token counts only |
| CI regression gate | Not configured | `make test` and `make ablation` exist, but no workflow |
| Deployment | Local only | `docker-compose.yml` for Postgres/pgvector + Redis |
| Secrets | Env-only | `.env` is gitignored; `.env.example` uses placeholders |
| Data retention / deletion | Not implemented | Sample public filings only |
| Prompt-injection defense | Not implemented | Filing text is untrusted input to the LLM; see [Security](#security-and-compliance) |

## Evaluation results

### Dataset and protocol

- `data/eval/eval_dataset.json`: 30 questions, 3 complexity tiers, 4 recommendation-framed questions.
- 28 questions have section-level citation labels; 2 are pure tool questions and are excluded from retrieval scoring.
- Retrieval ablation is offline and deterministic: local embeddings, local reranker, `k=8`, checked-in chunks.
- End-to-end baseline is a 3-question run through the full graph using Groq and the heuristic metrics backend.
- Raw artifacts: [`data/eval/retrieval_ablation.json`](data/eval/retrieval_ablation.json), [`data/eval/baseline.json`](data/eval/baseline.json).

### Retrieval ablation: dense vs hybrid vs hybrid + rerank

Reproduce: `uv run python -m src.eval --ablation`

| Configuration | Context precision | Context recall | MRR | Δ MRR vs dense |
|---|---:|---:|---:|---:|
| Dense only | 0.7321 | 1.0000 | 0.8929 | — |
| Hybrid (dense + BM25, RRF) | 0.7366 | 1.0000 | 0.9196 | +0.0267 |
| Hybrid + cross-encoder rerank | 0.7589 | 1.0000 | 0.9464 | +0.0535 |

`MRR` is the mean reciprocal rank of the first relevant chunk. Precision/recall are computed against section-level labels.

Interpretation:

- Reranking improves **ordering** (`MRR +0.0535`) but not coverage; precision moves only `+0.0268`.
- Hybrid is not uniformly better. It recovers exact financial terms (`eval_010` precision `0.250 → 0.750`) but can regress on lexical noise (`eval_013` `0.375 → 0.125`).
- Recall@8 is saturated at `1.0` because labels are section-level. Precision and MRR are the useful signals.
- These are **retrieval-only** numbers. They are not RAGAS, faithfulness, or answer-correctness scores.

### End-to-end smoke baseline (n=3, heuristic)

| Run | n | Context precision | Context recall | Faithfulness | Answer relevance | Numeric judge |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic end-to-end (Groq) | 3 | 0.9167 | 1.0000 | 1.0000 | 0.9107 | 1.0000 |

This is a smoke test, not a quality claim. All three examples passed citation verification; with `n=3`, that is not evidence of robustness.

### Acceptance criteria

PRD targets are used where defined; retrieval MRR@8 is an engineering gate added for this ablation. “Current” is measured only where evidence exists.

| Gate | Target | Current | Status |
|---|---:|---:|---|
| Retrieval MRR@8 | ≥ 0.90 | 0.9464 | Met on 28 questions |
| Context precision@8 | ≥ 0.80 | 0.7589 | Not met on coarse labels |
| Context recall@8 | ≥ 0.80 | 1.0000 | Saturated; labels are section-level |
| RAGAS faithfulness | ≥ 0.85 | Not measured | Blocked by RAGAS import/dependency issue |
| Citation judge | ≥ 90% grounded | 3/3 examples (14/14 claims) | Not statistically meaningful |
| Fallback success under provider failure | 100% | Unit-tested | No end-to-end chaos test |
| Routing accuracy (simple → cheap) | ≥ 90% | Not measured | No labeled routing set |
| HITL interception | 100% of recommendation-like output | Not measured | 4 recommendation examples exist |

### Metrics backends

- `--metrics heuristic` (default): deterministic, offline proxies. Fast, but **not RAGAS**.
- `--metrics ragas`: real RAGAS (`faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`). Requires a working RAGAS install plus an LLM judge key. It fails fast; it does not silently downgrade.
- `--metrics auto`: tries RAGAS once and records `metrics_backend: heuristic` plus `metrics_fallback_reason` in the output JSON.

> **Known dependency issue:** the checked-in lock (`ragas 0.3.1`) imports `PIL` and the removed `langchain_community.chat_models.vertexai` module. `--metrics ragas` currently exits with an actionable error. Pin a compatible `ragas`/`langchain-community` pair before publishing RAGAS numbers.

## Failure analysis

| Symptom | Example | Root cause | Next action |
|---|---|---|---|
| Low precision on table-heavy questions | `eval_004`, `eval_010`, `eval_014`, `eval_027` | HTML parser flattens Item 7 tables; chunks contain the right section but not the right passage | XBRL-first + table-aware parsing; passage-level labels |
| Hybrid regression | `eval_013` precision `0.375 → 0.125` | Lexical matches from table-adjacent text crowd out the narrative passage | Query-aware fusion, lexical score thresholds, or table-aware candidate filters |
| Rerank helps rank, not recall | `eval_004` MRR `0.50 → 1.00`, precision flat at `0.375` | The relevant chunk was already in the candidate set | Improve candidate recall separately from reranking |
| Recall appears perfect | All configurations | Citation labels are section-level, so “right section” is easy | Add passage/chunk-level labels and negative examples |
| P/E answer is fragile | `eval_008` | Live price via yfinance + EPS regex from text; filing EPS ≠ trailing EPS; basic vs diluted | XBRL companyfacts, point-in-time price, explicit EPS definition |
| Verification can pass weak claims | 3/3 smoke run passed | Keyword overlap, not entailment | NLI claim verification + numeric normalization |
| RAGAS not available | `--metrics ragas` | Dependency/lock incompatibility | Pin a working RAGAS stack; add integration test |
| Narrow corpus | 2 Apple 10-Ks | No Item 8, 10-Q/8-K, transcripts, peers, or international filings | Expand corpus before claiming broad financial QA |

## Reliability and failure modes

- **Provider failure:** fallback hops across OpenAI → Groq → Gemini → vLLM. If every provider fails, the pipeline raises `DegradedResponseError` instead of returning an unverified answer.
- **Verification failure:** bounded retry to retrieval (max 2). After exhaustion, the answer is returned with a visible verification flag, not silently passed. The verifier tolerates zero unmatched financial numeric claims and at most one non-numeric weak claim.
- **Postgres unavailable:** retrieval cannot proceed; the error is surfaced. There is no local vector-store fallback.
- **Redis unavailable:** embedding cache and Redis checkpointer fall back to in-memory behavior; HITL state is lost on restart.
- **Tool failure:** price fetch failure returns a tool error and does not fabricate a price. Missing EPS degrades to price-only with an explanation.
- **Partial ingestion:** parser warns when a section marker is missing; the document may still be chunked with fewer sections. There is no ingestion-level completeness gate yet.

## Security and compliance

Implemented controls:

- Secrets are injected via environment variables; `.env` is gitignored; no credentials are hardcoded.
- Source filings are treated as read-only; derived chunks and embeddings live under `data/`.
- SEC User-Agent and throttle settings exist in config; automated EDGAR crawling is not implemented.
- HITL is default-on for recommendation-like output. `--no-hitl` and `ask()` auto-approval are explicit local/eval escape hatches and must not be exposed in production.
- Pydantic validates config and tool outputs; citations use a typed `VerificationResult` dataclass.

Explicit gaps:

- **Prompt injection:** filing text is untrusted input. There is no instruction/data separation hardening, content sanitization, or tool-use allowlist beyond the current tool set.
- **Auth and tenancy:** no authentication, authorization, or tenant isolation.
- **Rate limiting and quotas:** no per-user or per-key limits.
- **Audit trail:** HITL decisions are not persisted to an append-only audit log.
- **PII / retention:** no data-retention policy or deletion workflow.
- **Regulatory disclaimer:** no investment-advice disclaimer is enforced in output.
- **Observability:** no traces, metrics, or alerting on provider errors, verification failures, or HITL decisions.

Do not expose this service to untrusted users until the gaps above are addressed.

## Operations

### Local quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env   # add provider keys for LLM runs; retrieval demo works without them
make demo
```

`make demo` runs `docker compose up -d`, `uv sync`, parses the sample filings, seeds pgvector, and runs a retrieval query. Manual equivalent:

```bash
docker compose up -d
uv sync
uv run python -m src.ingestion
uv run python -m src.retrieval seed
uv run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 3
```

Full pipeline:

```bash
uv run python -m src.graph ask "What were Apple's total net sales in fiscal 2023?" --k 5
uv run python -m src.graph ask "What's the current P/E ratio for AAPL given the latest filing?" --k 5
uv run python -m src.graph ask "Build a bull and bear case for AAPL, should I invest?" --k 5
```

### Common commands

```bash
# Tests (offline; no network/DB)
make test

# Offline retrieval comparison
make ablation

# End-to-end eval (requires provider keys + pgvector)
uv run python -m src.eval --limit 5
uv run python -m src.eval --metrics ragas --limit 5

# Targeted retrieval checks
uv run python -m src.retrieval search "what are the main risks" --k 5
uv run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 5
uv run python -m src.retrieval rerank "liquidity 148.3 billion" --k 5
```

### Verification coverage

| Layer | Coverage | Gap |
|---|---|---|
| Chunking, compression, routing, embeddings | Unit tests | No property-based or adversarial input tests |
| Dense/hybrid/rerank retrieval | Unit tests + offline ablation | No DB integration test; no load test |
| Provider fallback + tier routing | Unit tests with fake providers | No live-provider integration or chaos test |
| Graph node order | Unit test asserts `verify_citations` before `human_review_gate` | No end-to-end HITL resume test |
| Citation verification | Unit tests + 3-example smoke run | Heuristic overlap; no NLI/numeric regression suite |
| Real RAGAS | Adapter present | Not run; dependency/lock failure |
| API, auth, rate limits | Not implemented | — |

### Configuration

Key environment variables (see `.env.example` for the full set):

| Variable | Purpose | Default |
|---|---|---|
| `POSTGRES_DATABASE_URL` | pgvector connection | `postgresql://rag_user:rag_password@localhost:5432/rag_db` |
| `REDIS_URL` | cache/checkpoint connection | `redis://localhost:6379/0` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | embedding backend | `sentence_transformers` / `all-MiniLM-L6-v2` |
| `RERANKER_MODEL` | cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `DENSE_TOP_K` / `BM25_TOP_K` / `FUSION_TOP_K` / `RERANK_TOP_K` | retrieval depths | `20 / 20 / 20 / 8` |
| `FALLBACK_CHAIN` | provider order | `["openai","groq","gemini","vllm"]` |
| `HITL_ENABLED` | recommendation gate | `true` |
| `VERIFICATION_MAX_RETRIES` | bounded verification retries | `2` |
| `CHECKPOINTER_TYPE` | `redis` or `in_memory` | `in_memory` |

### Observability today

The pipeline logs provider/model/tier, verification failures, retrieval counts, rerank counts, cache behavior, and HITL decisions. `tokens_in` is recorded per LLM call. There is no metrics endpoint, trace export, alerting, or measured p95 latency/cost per query; those are deferred to Phase 2. Cost is bounded by tier routing, token budget, retrieval depth, and caching, but not enforced by a hard per-request budget.

## Limits and non-goals

- **Corpus:** 2 Apple 10-Ks; Items 1A and 7 only. No Item 8, 10-Q, 8-K, proxies, transcripts, peers, or IFRS filings.
- **Parsing:** HTML text extraction, not XBRL-first. Tables are flattened, so row/column relationships are lost.
- **P/E:** live price plus EPS extracted from text. Filing EPS and trailing EPS can differ; the tool does not reconcile them.
- **Evaluation:** retrieval ablation is measured; end-to-end is a 3-question smoke run; real RAGAS is not yet measured.
- **Verification:** keyword overlap, not entailment. Treat `verification.passed` as a floor.
- **Scale:** single-process CLI; no horizontal scaling, queueing, or concurrency controls.
- **Not legal, financial, or investment advice.** Recommendation-like output is withheld pending human review.

## Roadmap

Priority is risk reduction, not feature count.

**P0 — correctness and evidence**

1. Table-aware/XBRL ingestion for Item 8 and financial-statement tables.
2. Passage-level citation labels; add negative/abstention cases to the eval set.
3. Unblock and run full 30-question RAGAS; freeze judge model and prompts.
4. NLI-based claim verification + numeric normalization (`$M` vs `$B`, basic vs diluted EPS).
5. P/E via companyfacts/XBRL with explicit EPS definition and point-in-time price.

**P1 — production controls**

1. API with auth, tenant isolation, rate limiting, and request budgets.
2. Prompt-injection defenses: instruction/data separation, tool allowlist, output sanitization.
3. Structured audit logging for HITL approvals/rejections and verification failures.
4. CI gate: tests + retrieval ablation + eval regression thresholds.
5. Observability: traces/metrics for retrieval latency, provider errors, verification failure rate, HITL volume.

**P2 — product surface**

1. FastAPI + citation-highlighting UI with source preview.
2. Peer/multi-filing comparison and temporal queries.
3. Deployment hardening and on-call runbooks.

## Components

| Layer | Package | Use |
|---|---|---|
| Orchestration | `langgraph`, `langchain` | Graph, state, tools |
| LLM | `langchain-groq`, `langchain-openai`, `langchain-google-genai`, `openai` | Providers + fallback |
| Embeddings | `sentence-transformers` | Local 384d embeddings |
| Rerank | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | `pgvector`, `sqlalchemy`, `psycopg2-binary` | Postgres pgvector |
| Retrieval | `rank-bm25` | Lexical search |
| Parsing | `beautifulsoup4`, `lxml`, `html5lib` | EDGAR HTML |
| Tokens | `tiktoken` | Budget counting |
| Cache/State | `redis` | Embedding cache, checkpoint (fallback to memory) |
| Finance | `yfinance` | Live prices |
| Validation | `pydantic`, `pydantic-settings` | Config & schemas |
| Eval | `ragas` (real backend), deterministic heuristics | Metrics |

## State contract

`GraphState` (TypedDict) flows through nodes:

| Field | Type | Purpose |
|---|---|---|
| `query` | `str` | User question |
| `k` | `int` | Top-k to retrieve |
| `tier` | `simple\|normal\|complex` | From classifier |
| `chunks` | `list[dict]` | Retrieved + compressed chunks |
| `answer` / `draft_answer` | `str` | Final / draft answer |
| `provider` | `str` | LLM provider that served |
| `tokens_in` | `int` | Prompt tokens |
| `sources` | `list[str]` | Distinct chunk sources |
| `tool_results` | `list[dict]` | Tool outputs |
| `verification` | `dict` | `passed`, `failed_claims`, `verified_claims` |
| `verification_retries` | `int` | Bounded retry count (max 2) |
| `is_recommendation` | `bool` | HITL trigger |
| `hitl_approved` | `bool\|None` | Human decision |
| `error` | `str\|None` | Degraded mode |

## Tools

- `get_stock_price(ticker)` — yfinance live price, validated by `TickerPriceResponse`.
- `calculate_pe_ratio(price, eps)` — `price / eps`.
- `calculate_gross_margin(revenue, gross_profit)` — `gross_profit / revenue`.
- `calculate_growth_rate(current, previous)` — `(current-previous)/previous`.

All return `RatioResult` (Pydantic) with `value`, `numerator`, `denominator`, and `formula`. The `tool` node auto-handles P/E queries: fetch live price, extract EPS from chunks, compute the ratio, and append `[Tool-augmented: ...]`. See [Limits](#limits-and-non-goals) for why this remains fragile.

## Caching and checkpoints

`src/retrieval/cache.py` caches query/chunk embeddings by `model::text` hash in Redis (TTL 1h) with an in-memory fallback. `src/graph/checkpoint.py` uses `RedisSaver` when `checkpointer_type=redis` and Redis is reachable; otherwise it falls back to `MemorySaver` and logs the downgrade.

## Project structure

```text
src/
  config.py          # Settings
  routing.py         # Complexity classifier
  tokens.py          # Token counting
  ingestion/         # parse (Item 1A + 7) + chunk
  retrieval/         # embed, bm25, hybrid, rerank, compress, cache, storage
  llm/               # prompt, client, fallback, tools, answer
  tools/             # price, ratios, schemas
  verification/      # citation grounding
  hitl/              # recommendation gate
  graph/             # state, nodes, checkpoint, graph
  eval/              # dataset, metrics (heuristic + RAGAS), judge, harness, ablation
tests/               # unit tests (no network/DB required)
docs/                # architecture + CLI demo SVGs
data/
  *.html             # vendored SEC sample filings
  chunks/            # derived chunk JSON
  eval/              # eval_dataset.json, baseline.json, retrieval_ablation.json
Makefile             # one-command local demo and eval targets
docker-compose.yml   # Postgres/pgvector + Redis
```
