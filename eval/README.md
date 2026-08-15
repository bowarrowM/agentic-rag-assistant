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

## Baseline (2026-08-15, 11 answerable questions)

| top-k | fact recall |
|---|---|
| @1 | 0.64 |
| @3 | 0.73 |
| @5 | 0.91 |

**Finding.** Source hit-rate is 100% at k=3, but fact recall is only 0.73 — and it
climbs to 0.91 at k=5. The facts are in the index; they just rank 4th–5th instead of
top-3. Three questions miss at k=3 (`cards-prepaid`, `withdraw-bank-time`, `uptime-target`).

**Diagnosis.** Ingestion chunks on a fixed 500-character window, which splits Q&A
pairs across chunk boundaries, so a chunk only partially matches the query. This is a
chunking problem, not an embedding or ranking one.

**Next step.** Chunk on paragraph/heading boundaries, re-ingest, and re-measure recall@3.
