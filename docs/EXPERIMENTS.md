# Retrieval experiments (Phase 5)

Protocol: offline retrieval ablation over the checked-in 28 citation-labeled questions, `k=8`, local embeddings + local cross-encoder reranker, and the same section-level labels used in `README.md`. Unless noted, before/after values are `context_precision / context_recall / MRR`. Source artifacts are in `/tmp` for one-off runs; the committed baseline is `data/eval/retrieval_ablation.json`.

| ID | Change | Before | After | Decision |
|---|---|---|---|---|
| E1 | Section-aware + table chunks vs fixed 1000/150 | Fixed 1000/150: `0.6027 / 0.8571 / 0.6414` (360 chunks) | Section-aware + unsplit table chunks: `0.7768 / 0.9821 / 0.9000` (433 chunks) | **Keep current.** Recall and MRR both up. The delta includes table-aware Item 8 chunks, not only parent-child chunking. |
| E2 | `BAAI/bge-base-en-v1.5` (768d) vs MiniLM (384d) | MiniLM: `0.7768 / 0.9821 / 0.9000`, 54.1s full run | bge-base: `0.7500 / 0.9821 / 0.8940`, 228.1s full run | **Do not switch.** No recall gain; precision/MRR lower and CPU runtime ~4.2x. |
| E3 | Reranker top-N 50 vs 20 | Top-20 candidates: `0.7768 / 0.9821 / 0.9000` | Top-50 candidates: `0.7545 / 0.9821 / 0.9167` | **Keep top-20.** MRR improves slightly but precision drops below the top-20 setting; no p95 budget data yet. |
| E4 | Query rewriting for vague/complex queries | Complex tier baseline (hybrid+rerank): `0.7083 / 0.9167 / 0.7833` | Complex tier + acronym expansion: `0.7083 / 0.8333 / 0.7833` | **Do not adopt.** Recall falls; the dataset has no explicit `hard` tier, so the complex tier is a proxy. HyDE was not run. |

## Notes

- E1 baseline used fixed `RecursiveCharacterTextSplitter(1000, 150)` chunks over each whole document; it did not serialize Item 8 tables as separate chunks. Current chunks are section-scoped and keep each table intact.
- E2 used the same chunks and labels; only `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` changed. Runtime is CPU end-to-end ablation time (corpus embedding + 28 queries), not API p95.
- E3 changed `DENSE_TOP_K=BM25_TOP_K=FUSION_TOP_K=50`; reranking still returned top-8.
- E4 expanded only `complex` examples with rule-based acronym/vague-term additions (`P/E → price earnings ratio`, etc.). It did not change labels or retrieval depth.
- These are retrieval-only metrics. They are not RAGAS faithfulness or answer-correctness scores.
