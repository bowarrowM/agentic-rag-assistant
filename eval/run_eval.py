"""
Evaluation harness for the Kitty Pay agentic RAG assistant.

Run modes:
  python eval/run_eval.py              # retrieval only  (fast, no LLM needed)
  python eval/run_eval.py --k 5        # retrieval recall at top-5
  python eval/run_eval.py --full       # end-to-end: routing + faithfulness too
                                        # (needs Ollama running, like the app)

Run from the repo root so the chroma_db path resolves.
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb

# Make the repo root importable + make relative paths (chroma_db) resolve no matter which directory we launch from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHROMA_PATH = str(REPO_ROOT / "chroma_db")
COLLECTION = "kitty_docs"
EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"


def load_eval_set():
    data = json.loads(EVAL_SET.read_text())
    return data["items"]


def contains_any(haystack, needles):
    """True if any needle substring appears in haystack (case-insensitive)."""
    h = haystack.lower()
    return any(n.lower() in h for n in needles)


def contains_all(haystack, needles):
    """True if every needle substring appears in haystack (case-insensitive)."""
    h = haystack.lower()
    return all(n.lower() in h for n in needles)


# RETRIEVAL EVAL  
def evaluate_retrieval(items, k):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)

    # Only questions that are meant to be answered from the docs get scored on retrieval. Ticket / refuse items don't have a gold fact to retrieve.
    scored = [it for it in items if it["expected_behavior"] == "answer"]

    source_hits = 0
    content_recalls = []          # per-question recall = found_facts / expected_facts
    perfect = 0
    rows = []

    for it in scored:
        res = collection.query(query_texts=[it["question"]], n_results=k)
        chunks = res["documents"][0]
        sources = [m["source"] for m in res["metadatas"][0]]
        joined = "\n".join(chunks)

        source_ok = it["expected_source"] in sources
        source_hits += int(source_ok)

        # must_retrieve holds the accepted spellings of the ONE fact the answer
        # needs (e.g. "1-3 business days" / "1–3 business days"). The fact counts
        # as retrieved if any spelling appears in the top-k chunks.
        needed = it["must_retrieve"]
        recall = 1.0 if contains_any(joined, needed) else 0.0
        content_recalls.append(recall)
        if recall == 1.0:
            perfect += 1

        rows.append((it["id"], source_ok, recall))

    n = len(scored)
    print(f"\n{'='*60}\nRETRIEVAL  (top-k = {k}, {n} answerable questions)\n{'='*60}")
    print(f"{'question id':<26}{'source hit':<12}{'fact recall'}")
    print("-" * 60)
    for qid, s_ok, rec in rows:
        print(f"{qid:<26}{('yes' if s_ok else 'NO '):<12}{rec:.2f}")
    print("-" * 60)
    print(f"Source hit-rate@{k}:   {source_hits}/{n}  ({source_hits/n:.0%})")
    print(f"Mean fact recall@{k}:  {sum(content_recalls)/n:.2f}")
    print(f"Perfect recall@{k}:    {perfect}/{n}  ({perfect/n:.0%})")
    return sum(content_recalls) / n


# END-TO-END EVAL  
def evaluate_end_to_end(items):
    from agent_core import answer  # imported here so retrieval-only mode needs no LLM

    routing_ok = 0
    faithful = 0
    faithful_total = 0
    answer_ok = 0
    answer_total = 0
    rows = []

    for it in items:
        final, sources, trace = answer(it["question"])
        tools_used = [t["tool"] for t in trace if t.get("type") == "tool_call"]
        opened_ticket = "create_support_ticket" in tools_used
        verdicts = [t["verdict"] for t in trace if t.get("type") == "verify"]
        verdict = verdicts[-1] if verdicts else "N/A"

        beh = it["expected_behavior"]
        if beh == "ticket":
            route_ok = opened_ticket
        elif beh == "refuse":
            # Correct = did NOT confidently answer from nothing: either opened a ticket or said it doesn't know
            route_ok = opened_ticket or contains_any(
                final, ["don't know", "do not know", "couldn't", "cannot help", "not sure"]
            )
        else:  # answer
            route_ok = not opened_ticket
        routing_ok += int(route_ok)

        # Faithfulness 
        if verdict in ("SUPPORTED", "UNSUPPORTED"):
            faithful_total += 1
            faithful += int(verdict == "SUPPORTED")

        # Answer-correctness 
        if beh == "answer" and it["answer_must_contain"]:
            answer_total += 1
            ans_ok = contains_all(final, it["answer_must_contain"])
            answer_ok += int(ans_ok)
        else:
            ans_ok = None

        rows.append((it["id"], beh, route_ok, verdict, ans_ok))

    n = len(items)
    print(f"\n{'='*72}\nEND-TO-END  ({n} questions)\n{'='*72}")
    print(f"{'question id':<26}{'expected':<10}{'routed ok':<11}{'faithful':<11}{'answer ok'}")
    print("-" * 72)
    for qid, beh, r_ok, verd, a_ok in rows:
        a = "-" if a_ok is None else ("yes" if a_ok else "NO")
        print(f"{qid:<26}{beh:<10}{('yes' if r_ok else 'NO '):<11}{verd:<11}{a}")
    print("-" * 72)
    print(f"Tool-routing accuracy: {routing_ok}/{n}  ({routing_ok/n:.0%})")
    if faithful_total:
        print(f"Faithfulness rate:     {faithful}/{faithful_total}  ({faithful/faithful_total:.0%})  (of answers grounded in context)")
    if answer_total:
        print(f"Answer correctness:    {answer_ok}/{answer_total}  ({answer_ok/answer_total:.0%})")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Kitty Pay RAG assistant.")
    ap.add_argument("--k", type=int, default=3, help="top-k chunks for retrieval eval (default 3)")
    ap.add_argument("--full", action="store_true", help="also run end-to-end (needs Ollama)")
    args = ap.parse_args()

    items = load_eval_set()
    evaluate_retrieval(items, args.k)
    if args.full:
        evaluate_end_to_end(items)
    print()


if __name__ == "__main__":
    main()
