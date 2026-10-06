"""Offline RAG evaluation on a golden question set.

Metrics
-------
* retrieval hit@k   - expected source appears in the cited/retrieved passages
* MRR               - reciprocal rank of the expected source
* answer keyword recall - expected facts (numbers, names) appear in the answer
* refusal accuracy  - questions whose answer is not accessible are refused, not hallucinated
* latency p50 / p95

    python -m scripts.evaluate                       # local report
    python -m scripts.evaluate --langsmith           # also run as a LangSmith experiment
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.pipeline.container import build_container
from src.pipeline.rag_pipeline import QueryRequest
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logger import setup_logging
from src.utils.tracing import configure_tracing

DEFAULT_SET = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"


def load_examples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(example: dict[str, Any], response: dict[str, Any]) -> dict[str, float]:
    answer = response["answer"].lower()
    sources = [c["source"] for c in response["citations"]]
    expected = example.get("expected_source")
    if expected is None:  # should be refused
        refused = "couldn't find" in answer or "could not find" in answer
        return {
            "hit": float(refused),
            "rr": float(refused),
            "keyword_recall": float(refused),
            "refusal": float(refused),
        }
    rank = sources.index(expected) + 1 if expected in sources else 0
    kws = example.get("expected_keywords", [])
    recall = sum(k.lower() in answer for k in kws) / len(kws) if kws else 1.0
    return {"hit": float(rank > 0), "rr": 1 / rank if rank else 0.0, "keyword_recall": recall}


def run_local(examples: list[dict[str, Any]]) -> None:
    container = build_container(get_settings())
    rows, latencies = [], []
    for ex in examples:
        res = container.rag.answer(QueryRequest(ex["question"], role=ex.get("role", "employee"), use_cache=False))
        s = score(ex, res)
        rows.append(s)
        latencies.append(res["timings_ms"]["total"])
        flag = "OK " if s["hit"] and s["keyword_recall"] >= 0.99 else "XX "
        print(f"{flag} {ex['question'][:70]:<70} hit={s['hit']:.0f} kw={s['keyword_recall']:.2f}")

    n = len(rows)
    lat = sorted(latencies)
    print("\n=== Summary ===")
    print(f"examples            : {n}")
    print(f"hit@k               : {sum(r['hit'] for r in rows) / n:.2%}")
    print(f"MRR                 : {sum(r['rr'] for r in rows) / n:.3f}")
    print(f"answer keyword recall: {sum(r['keyword_recall'] for r in rows) / n:.2%}")
    print(f"latency p50 / p95   : {statistics.median(lat):.0f} ms / {lat[int(0.95 * (n - 1))]:.0f} ms")


def run_langsmith(examples: list[dict[str, Any]], dataset_name: str) -> None:
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        ds = client.create_dataset(dataset_name, description="HR assistant golden QA")
        client.create_examples(
            dataset_id=ds.id,
            examples=[
                {
                    "inputs": {"question": e["question"], "role": e.get("role", "employee")},
                    "outputs": {k: e.get(k) for k in ("expected_source", "expected_keywords")},
                }
                for e in examples
            ],
        )
    container = build_container(get_settings())

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return container.rag.answer(QueryRequest(inputs["question"], role=inputs["role"], use_cache=False))

    def retrieval_hit(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": "retrieval_hit", "score": score(reference_outputs, outputs)["hit"]}

    def keyword_recall(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": "keyword_recall", "score": score(reference_outputs, outputs)["keyword_recall"]}

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[retrieval_hit, keyword_recall],
        experiment_prefix=f"hr-rag-{get_settings().llm.model}",
        metadata={"llm": get_settings().llm.model, "reranker": get_settings().reranker.provider},
        max_concurrency=4,
    )
    print(f"LangSmith experiment: {results.experiment_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_SET)
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--dataset", default="hr-assistant-golden-qa")
    args = parser.parse_args()

    s = get_settings()
    setup_logging("WARNING", s.resolve_path(s.app.log_dir), json_logs=False)
    configure_tracing(
        s.tracing.langsmith_enabled,
        s.langsmith_api_key.get_secret_value() if s.langsmith_api_key else None,
        s.tracing.project,
    )
    examples = load_examples(args.file)
    run_local(examples)
    if args.langsmith:
        run_langsmith(examples, args.dataset)


if __name__ == "__main__":
    main()
