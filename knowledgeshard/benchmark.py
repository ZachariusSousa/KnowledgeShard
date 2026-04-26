"""Gold-rubric benchmark scoring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .savant import Savant
from .storage import KnowledgeStore


def load_benchmark(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["questions"])


def score_answer(answer: str, citations: list[dict], item: dict) -> dict:
    text = answer.lower()
    required = [point.lower() for point in item.get("required_key_points", [])]
    forbidden = [claim.lower() for claim in item.get("forbidden_claims", [])]
    matched = [point for point in required if point in text]
    forbidden_hits = [claim for claim in forbidden if claim in text]
    citation_tags = set(item.get("expected_citation_tags", []))
    citation_text = " ".join(
        f"{citation.get('source', '')} {citation.get('excerpt', '')}".lower()
        for citation in citations
    )
    cited = bool(citations)
    tag_matched = not citation_tags or any(tag.lower() in citation_text for tag in citation_tags)
    required_score = len(matched) / max(len(required), 1)
    score = required_score
    if not cited or not tag_matched:
        score *= 0.75
    if forbidden_hits:
        score = 0.0
    threshold = float(item.get("threshold", 0.85))
    return {
        "question": item["question"],
        "score": round(score, 3),
        "passed": score >= threshold,
        "matched_key_points": matched,
        "forbidden_hits": forbidden_hits,
        "citations_present": cited,
        "citation_tag_matched": tag_matched,
    }


def run_benchmark(
    benchmark_path: str | Path = "benchmarks/mario_kart_wii_10.json",
    db_path: str | Path = "data/knowledgeshard.db",
    domain: str = "mario-kart-wii",
    top_k: int = 5,
) -> dict:
    store = KnowledgeStore(db_path)
    savant = Savant(domain=domain, store=store)
    started = perf_counter()
    results = []
    for item in load_benchmark(benchmark_path):
        response = savant.query(item["question"], num_experts=top_k)
        result = score_answer(
            response.answer,
            [asdict(citation) for citation in response.citations],
            item,
        )
        result["query_id"] = response.query_id
        result["confidence"] = response.confidence
        results.append(result)
    accuracy = sum(1 for result in results if result["passed"]) / max(len(results), 1)
    return {
        "domain": domain,
        "benchmark": str(benchmark_path),
        "accuracy": round(accuracy, 3),
        "passed": accuracy >= 0.85,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "results": results,
    }
