# FinSight — Financial Filings RAG

RAG assistant for SEC 10-K filings. Answers are grounded in retrieved passages, with inline citations, tool-augmented live data, provider fallback, and a human-in-the-loop gate for investment-advice-like output.

**Links:** [Architecture](docs/architecture.svg) · [CLI demo](docs/demo.svg) · [Results](#results) · [Failure analysis](#failure-analysis) · [Limits / interview notes](#limits--interview-notes)

**Topics:** `sec-edgar` · `rag` · `langgraph` · `pgvector` · `bm25` · `reranking` · `ragas` · `python`

> **Status:** sample corpus only (2 Apple 10-Ks, 247 chunks). The retrieval ablation below is measured and reproducible. The checked-in end-to-end baseline is a **3-question heuristic smoke run**, not a full RAGAS result. The 30-question RAGAS run is still blocked by the current `ragas`/`langchain-community` import combination; no provider keys were used for the checked-in run. See [Results](#results) and [Failure analysis](#failure-analysis) for exactly what is and is not proven.

## What it does

- Parses SEC 10-K HTML and extracts **Item 1A (Risk Factors)** and **Item 7 (MD&A)**.
- Chunks with metadata (`ticker`, `filing_type`, `fiscal_period`, `section`, `chunk_index`).
- Retrieves with **dense (pgvector) + BM25 + weighted RRF**, then **cross-encoder reranking**, then context compression.
- Routes queries by complexity (`simple` → cheap, `normal` → medium, `complex` → expensive) with an `OpenAI → Groq → Gemini → local vLLM` fallback chain.
- Supports tools for live price, P/E, gross margin, and growth rate.
- Verifies every claim against retrieved chunks (bounded retry), then sends recommendation-like output through a human review gate.
- Ships an offline evaluation harness plus a retrieval ablation that runs without Postgres or API keys.

## Architecture

![FinSight architecture](docs/architecture.svg)

**Offline (ingestion):** `data/*.html` → `parse_edgar_file` (Item 1A, Item 7) → `chunk_document` (1000/150, metadata) → `embed_chunks` (sentence-transformers, 384d) → pgvector.

**Online (graph):** `ask()` builds state → classify → retrieve (hybrid + rerank) → compress → LLM (routed + fallback + tools) → verify citations (retry on failure) → human review gate → finalize with sources.

## At a glance

| Item | Value |
|---|---:|
| Sample filings | 2 Apple 10-Ks (FY2023, FY2025) |
| Parsed sections | Item 1A Risk Factors, Item 7 MD&A |
| Indexed chunks | 247 |
| Eval questions | 30 (28 with citation labels, 2 pure tool questions) |
| End-to-end smoke baseline | 3 questions (heuristic metrics) |
| Default retrieval depth | 8 chunks |

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Docker (for Postgres + pgvector + Redis).

```bash
cp .env.example .env   # add provider keys for LLM runs; retrieval demo works without them
make demo
```

`make demo` is the one-command local smoke path. It runs `docker compose up -d`, `uv sync`, parses the sample filings, seeds pgvector, and runs a retrieval query.

If you prefer not to use `make`:

```bash
docker compose up -d          # Postgres/pgvector + Redis
uv sync
uv run python -m src.ingestion
uv run python -m src.retrieval seed
uv run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 3
```

Then run the full graph:

```bash
uv run python -m src.graph ask "What were Apple's total net sales in fiscal 2023?" --k 5
uv run python -m src.graph ask "What's the current P/E ratio for AAPL given the latest filing?" --k 5
uv run python -m src.graph ask "Build a bull and bear case for AAPL, should I invest?" --k 5
```

Why `uv`, not `pip install -e .`: the repo is uv-managed (`pyproject.toml` + `uv.lock`), and `uv run` keeps the local environment reproducible. Why `docker compose`: it is the supported way to get pgvector and Redis on the expected ports.

## Usage

```bash
# Retrieval
uv run python -m src.retrieval search "what are the main risks" --k 5
uv run python -m src.retrieval hybrid "total net sales 383.3 billion" --k 5
uv run python -m src.retrieval rerank "liquidity 148.3 billion" --k 5

# Single LLM call
uv run python -m src.llm answer "what are the main risks" --k 5

# Full pipeline (routing + fallback + tools + verification + HITL)
uv run python -m src.graph ask "what was Apple's total net sales in fiscal 2023" --k 5

# Offline retrieval ablation (no LLM, no DB)
uv run python -m src.eval --ablation

# End-to-end eval (requires provider keys + pgvector)
uv run python -m src.eval --limit 5
uv run python -m src.eval --metrics ragas --limit 5   # fails loudly if RAGAS is unavailable
```

Routing: `simple` → cheap, `normal` → medium, `complex` → expensive. Override with `MODEL_ROUTING_<TIER>_MODEL_NAME`.

Fallback: `openai → groq → gemini → vllm`, logged as `Served by provider=... model=... tier=...`.

## Evaluation

The eval set is `data/eval/eval_dataset.json` (30 Q/A, 3 complexity tiers, 4 recommendation-framed questions). There are two evaluation modes:

| Mode | Command | Needs LLM/DB? | What it measures |
|---|---|---|---|
| Retrieval ablation | `uv run python -m src.eval --ablation` | No | Recall/precision/MRR of dense vs hybrid vs hybrid+rerank |
| End-to-end eval | `uv run python -m src.eval --limit 5` | Yes | Answer metrics + numeric judge through the graph |

### Metrics backends

- `--metrics heuristic` (default): deterministic, offline proxies for context precision/recall, faithfulness, and answer relevance. Fast, but **not RAGAS**.
- `--metrics ragas`: real RAGAS (`faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`). Requires a working `ragas` install plus an LLM judge key. It **fails fast** instead of silently downgrading.
- `--metrics auto`: tries RAGAS once. If unavailable, records `metrics_backend: heuristic` and `metrics_fallback_reason` in the output JSON. No silent fallback is hidden.

The checked-in `data/eval/baseline.json` was produced with the heuristic backend, so it is labeled as such.

> Known dependency issue: the checked-in lock (`ragas 0.3.1`) imports `PIL` and the removed `langchain_community.chat_models.vertexai` module. `--metrics ragas` therefore exits with an actionable error in this environment. Pin a compatible `ragas`/`langchain-community` pair before claiming RAGAS numbers.

## Results

### 1. Retrieval ablation — dense only vs hybrid vs hybrid + rerank

Measured on the 28 citation-labeled questions, `k=8`, using the checked-in chunks, local `all-MiniLM-L6-v2` embeddings, and `cross-encoder/ms-marco-MiniLM-L-6-v2`. Reproduce with `uv run python -m src.eval --ablation`; raw output is in [`data/eval/retrieval_ablation.json`](data/eval/retrieval_ablation.json).

| Configuration | Context precision | Context recall | MRR | Δ MRR vs dense |
|---|---:|---:|---:|---:|
| Dense only | 0.7321 | 1.0000 | 0.8929 | — |
| Hybrid (dense + BM25, RRF) | 0.7366 | 1.0000 | 0.9196 | +0.0267 |
| Hybrid + cross-encoder rerank | 0.7589 | 1.0000 | 0.9464 | +0.0535 |

How to read this honestly:

- **Reranking helps ranking, not coverage.** MRR rises from `0.8929` to `0.9464`, but precision only rises from `0.7321` to `0.7589`.
- **Hybrid is mixed.** It recovers exact financial terms (e.g. `eval_010` precision `0.250 → 0.750`), but lexical matches can also flood the candidate set (`eval_013` precision `0.375 → 0.125`).
- **Recall is saturated at `1.0`** because the citation labels are section-level (`ticker` + `fiscal_period` + `section`). A top-8 list almost always contains the right section. Precision and MRR are the discriminating signals here.
- **These are retrieval-only numbers.** They do not measure answer correctness, faithfulness, or RAGAS quality.

### 2. End-to-end smoke baseline (3 questions, heuristic metrics)

Source: [`data/eval/baseline.json`](data/eval/baseline.json). This is the checked-in run; it is a smoke test, not a quality claim.

| Run | n | Context precision | Context recall | Faithfulness | Answer relevance | Numeric judge |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic end-to-end (Groq) | 3 | 0.9167 | 1.0000 | 1.0000 | 0.9107 | 1.0000 |

The three questions were `eval_001` (net sales), `eval_002` (competition risk), and `eval_003` (Item 1A summary). All three passed citation verification. With `n=3`, the perfect faithfulness/judge scores are not evidence of robustness.

### 3. Reproduce

```bash
# Retrieval-only, no keys, no DB
uv run python -m src.eval --ablation

# End-to-end smoke (set GROQ_API_KEY or OPENAI_API_KEY and start pgvector first)
uv run python -m src.eval --limit 3

# Full 30-question RAGAS run (after installing a working ragas stack)
uv run python -m src.eval --metrics ragas
```

## Failure analysis

### Retrieval failures

1. **Table-heavy questions are the weak spot.** `eval_004` (gross-margin trend), `eval_010` (material cash requirements), `eval_014` (liquidity/cash), and `eval_027` (share repurchases/dividends) all have precision between `0.25` and `0.50` under at least one configuration. The parser extracts plain text from Item 7, so table rows are flattened; many returned chunks are from the right section but not the right passage.
2. **Hybrid fusion can hurt as well as help.** `eval_013` ("manufacturing purchase obligations") drops from `0.375` dense precision to `0.125` hybrid/rerank precision. The phrase matches many table-adjacent chunks, and candidate fusion lets lexical matches crowd out the relevant narrative passage.
3. **Reranking fixes order, not recall.** `eval_004` goes from MRR `0.50` (dense) to `1.00` (reranked) while precision stays `0.375`. That is the expected behavior: the right chunk was already in the candidate set, but not near the top.
4. **Section-level labels are coarse.** Recall@8 is `1.0` everywhere because "right section" is easy. Precision is a conservative proxy: a chunk can be from the right section and still be the wrong passage.
5. **Two filings only.** There is no peer comparison, no 10-Q/8-K, and no earnings-call transcript. `eval_005` asks to compare "the two Apple filings"; the corpus can support that, but it cannot support peer or multi-period comparisons beyond FY2023/FY2025.

### Answer / tool failures to be ready for

1. **P/E is fragile by design.** The tool path fetches a live price and extracts EPS from retrieved text with regex. It can pick basic vs diluted EPS, annual vs quarterly EPS, or filing EPS vs trailing EPS. If no EPS is found, it correctly degrades to a live price plus an explanation instead of inventing a ratio. A reviewer should ask why the answer is not using XBRL/companyfacts.
2. **10-K answers often live in Item 8 financial statements.** The parser does not extract Item 8, and it does not preserve table structure. Questions about segment tables, cash-flow tables, or footnote-level figures can fail even when retrieval returns Item 7.
3. **Verification is keyword-overlap based.** It is a useful guardrail, but not NLI. A claim with the right entities and the wrong number can pass. The 3-question smoke run passed everything; that is not proof of correctness.
4. **HITL is a compliance control, not a model guarantee.** The CLI pauses for recommendation-like queries, but the non-interactive `ask()` helper auto-approves so evaluation can run. Production use should not rely on the helper path.
5. **The RAGAS gap is real.** The adapter is implemented, but the current environment cannot import `ragas` (missing `PIL` and a removed `langchain_community.chat_models.vertexai` module). The checked-in numbers are therefore heuristic/retrieval-only. This is documented rather than hidden.

## Limits / interview notes

- **Corpus:** 2 Apple 10-Ks; Items 1A and 7 only; no 10-Q, 8-K, proxies, transcripts, or peer filings. Not a broad financial QA benchmark.
- **Parsing:** HTML text extraction, not XBRL-first. Tables are flattened, so table-row/column relationships are lost. This is the main reason Item 8-style questions are out of scope.
- **P/E:** `get_stock_price` (yfinance) + EPS regex from chunks. Filing EPS and trailing EPS can differ; the tool does not reconcile them. If EPS cannot be extracted, it returns live price only.
- **Retrieval:** hybrid + rerank is stronger than dense-only on MRR, but not uniformly better on precision. The ablation shows both wins and regressions.
- **Evaluation:** retrieval ablation is offline and reproducible. End-to-end baseline is 3 questions. Real RAGAS is implemented behind `--metrics ragas` but not yet run in this environment.
- **Verification:** keyword overlap, not entailment. Treat `verification.passed` as a floor, not a guarantee.
- **What I would do next:** table-aware/XBRL parsing for Item 8, stronger numeric verification, full 30-question RAGAS run with a frozen judge model, and peer filings for comparison questions.

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

## State

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

LangChain tools bound to the LLM (also callable directly):

- `get_stock_price(ticker)` — yfinance live price, returns `{ticker, price, currency, as_of}` validated by `TickerPriceResponse`
- `calculate_pe_ratio(price, eps)` — `price / eps`
- `calculate_gross_margin(revenue, gross_profit)` — `gross_profit / revenue`
- `calculate_growth_rate(current, previous)` — `(current-previous)/previous`

All return `RatioResult` (Pydantic) with `value`, `numerator`, `denominator`, `formula`. The `tool` node auto-handles P/E queries: fetches live price, extracts EPS from chunks, computes the ratio, and appends `[Tool-augmented: ...]`. See [Limits](#limits--interview-notes) for why this is fragile.

## Caching

`src/retrieval/cache.py` caches query/chunk embeddings by `model::text` hash in Redis (TTL 1h) with an in-memory fallback. Repeated queries hit cache → faster.

Checkpointer: `src/graph/checkpoint.py` tries `RedisSaver` when `checkpointer_type=redis` and Redis is reachable, else `MemorySaver` (logged fallback). Enables HITL state to survive restarts when Redis is available.

## Future scope

1. **Data & ingestion:** XBRL-first, table-aware parsing for Item 8; automated EDGAR ingestion with User-Agent + throttling; 10-Q/8-K/proxy/transcripts; structure-aware chunking.
2. **Retrieval:** metadata pre-filters, query rewriting/HyDE, multi-hop and parent-document retrieval, benchmarked rerankers, separate graph nodes for dense/BM25/fusion/rerank.
3. **Live data & tools:** SEC companyfacts lookup, historical prices, DCF/ratio suite, XBRL numeric extraction with Pydantic schemas.
4. **Trust & safety:** NLI-based claim verification, numeric normalization (`$M` vs `$B`), audit logs for HITL, advice disclaimer, refusal path.
5. **Evaluation:** full 30-question RAGAS run + frozen judge, 100+ question set with peer/temporal queries, CI regression gate, cost/latency tracking.
6. **Robustness:** circuit breakers, specific timeout/429/auth handling, complete fallback chain with logged reasons.
7. **Deployment/UX:** FastAPI + chat UI with citation highlighting; keep Dockerized Postgres/Redis; AWS/MLOps and LGTM observability deferred to Phase 2.

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

## Repository metadata (for maintainers)

Suggested GitHub description:

> Grounded RAG over SEC 10-K filings with hybrid retrieval, reranking, model fallback, citation verification, and human-in-the-loop review.

Suggested topics: `sec-edgar`, `rag`, `langgraph`, `pgvector`, `bm25`, `reranking`, `ragas`, `financial-analysis`, `python`.

Set them with GitHub CLI (requires `gh auth login`):

```bash
gh repo edit SachinMaurya01/FinSight \
  --description "Grounded RAG over SEC 10-K filings with hybrid retrieval, reranking, model fallback, citation verification, and human-in-the-loop review." \
  --add-topic sec-edgar --add-topic rag --add-topic langgraph --add-topic pgvector \
  --add-topic bm25 --add-topic reranking --add-topic ragas --add-topic financial-analysis \
  --add-topic python
```
