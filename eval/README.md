# Evaluation

A small, reproducible harness for measuring the assistant instead of eyeballing it.
Run `run_eval.py` against the gold set in `eval_set.json` (facts grounded in the
documents under `documents/`).

## What it measures

| Metric | Question it answers |
|---|---|
| **Retrieval recall@k** | When we search the KB, do the top-k chunks actually contain the fact the answer needs? |
| **Tool-routing accuracy** | Does the agent answer from docs when it should, and open a support ticket when it should? |
| **Faithfulness rate** | Is the final answer grounded in the retrieved context, or invented? (LLM-as-judge, `verify_answer`) |

## How to run

```bash
python eval/run_eval.py            # retrieval only (fast, no LLM)
python eval/run_eval.py --k 5      # retrieval recall at top-5
python eval/run_eval.py --full     # + routing and faithfulness (needs Ollama)
```

## Results (2026-08-15, 11 answerable questions)

| top-k | fixed 500-char chunks | structure-aware chunks |
|---|---|---|
| @1 | 0.64 | **0.82** |
| @3 | 0.73 | **0.91** |
| @5 | 0.91 | 0.91 |

**Finding.** With fixed 500-char chunking, source hit-rate was 100% at k=3 but fact
recall only 0.73 — facts were indexed yet ranked 4th–5th because the blind cut split
Q&A pairs across chunk boundaries. A chunking problem, not an embedding or ranking one.

**Fix.** Chunk on paragraph/heading boundaries (`ingest.py: chunk_markdown`) so each
Q&A / list stays whole, with its section heading prepended for context. Recall@3 rose
0.73 → 0.91, and answers now surface in the top 1–3 instead of needing top-5.

**Known limitation.** One query (`cards-prepaid`) still misses at every k. Rare keywords
and negations ("prepaid", "not supported") are blurred by dense embeddings and out-ranked
by the many card/fee chunks. Closing it needs hybrid retrieval (BM25 + semantic) or a
reranker — deferred as future work.
