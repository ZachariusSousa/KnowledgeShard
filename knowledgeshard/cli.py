"""Command line interface for the local savant."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import run_benchmark
from .graph import KnowledgeGraph
from .model_runtime import train_lora
from .obsession import ObsessionLoop
from .savant import Savant
from .seed import load_seed_facts
from .storage import KnowledgeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledgeshard", description="Run a local single-domain savant.")
    parser.add_argument("--db", default="data/knowledgeshard.db", help="SQLite database path.")
    parser.add_argument("--domain", default="mario-kart-wii", help="Savant domain.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Load seed facts into the knowledge graph.")
    seed.add_argument("--file", default="data/seeds/mario_kart_wii.json", help="Seed JSON file.")

    ask = subparsers.add_parser("ask", help="Ask the local savant a question.")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)

    add_fact = subparsers.add_parser("add-fact", help="Add one manual fact.")
    add_fact.add_argument("subject")
    add_fact.add_argument("relation")
    add_fact.add_argument("object")
    add_fact.add_argument("--confidence", type=float, default=0.8)
    add_fact.add_argument("--source", default="manual")

    correct = subparsers.add_parser("correct", help="Correct a previous answer.")
    correct.add_argument("query_id")
    correct.add_argument("correction")
    correct.add_argument("--confidence", type=float, default=1.0)

    subparsers.add_parser("status", help="Show savant status.")
    subparsers.add_parser("metrics", help="Show MVP metrics.")

    graph = subparsers.add_parser("graph", help="Inspect the approved knowledge graph.")
    graph_subparsers = graph.add_subparsers(dest="graph_command", required=True)
    graph_subparsers.add_parser("stats", help="Show graph node and edge counts.")
    neighbors = graph_subparsers.add_parser("neighbors", help="Show outgoing graph facts for an entity.")
    neighbors.add_argument("entity")

    obsess = subparsers.add_parser("obsess", help="Run the curated obsession review loop.")
    obsess.add_argument("--config", default="config/mario_kart_wii.sources.json")
    obsess_subparsers = obsess.add_subparsers(dest="obsess_command", required=True)
    fetch = obsess_subparsers.add_parser("fetch", help="Fetch sources into pending facts.")
    fetch.add_argument("--limit-per-source", type=int, default=10)
    review = obsess_subparsers.add_parser("review", help="List pending facts.")
    review.add_argument("--limit", type=int, default=20)
    approve = obsess_subparsers.add_parser("approve", help="Approve one pending fact.")
    approve.add_argument("pending_id")
    reject = obsess_subparsers.add_parser("reject", help="Reject one pending fact.")
    reject.add_argument("pending_id")
    run_once = obsess_subparsers.add_parser("run-once", help="Run one fetch cycle.")
    run_once.add_argument("--limit-per-source", type=int, default=10)

    train = subparsers.add_parser("train-lora", help="Train a PEFT LoRA adapter from approved facts.")
    train.add_argument("--output-dir", default="weights/lora")
    train.add_argument("--model-id", default=None)

    benchmark = subparsers.add_parser("benchmark", help="Run the Mario Kart Wii gold-rubric benchmark.")
    benchmark.add_argument("--file", default="benchmarks/mario_kart_wii_10.json")
    benchmark.add_argument("--results-dir", default="benchmarks/results")
    benchmark.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = KnowledgeStore(args.db)
    savant = Savant(domain=args.domain, store=store)

    if args.command == "seed":
        count = load_seed_facts(Path(args.file), store, args.domain)
        print(f"Loaded {count} facts into {args.db}.")
        return 0
    if args.command == "ask":
        response = savant.query(args.question, num_experts=args.top_k)
        print(response.answer)
        print(f"query_id: {response.query_id}")
        for index, citation in enumerate(response.citations, start=1):
            print(f"[{index}] {citation.excerpt} ({citation.source}, confidence {citation.confidence:.2f})")
        return 0
    if args.command == "add-fact":
        fact = savant.add_fact(args.subject, args.relation, args.object, args.confidence, args.source)
        print(f"Added fact {fact.id}.")
        return 0
    if args.command == "correct":
        correction = savant.correct(args.query_id, args.correction, args.confidence)
        print(f"Saved correction {correction.id}.")
        return 0
    if args.command == "status":
        print(asdict_like(savant.status()))
        return 0
    if args.command == "metrics":
        print(asdict_like(savant.metrics()))
        return 0
    if args.command == "graph":
        graph = KnowledgeGraph(store, args.domain)
        if args.graph_command == "stats":
            print(json.dumps(graph.stats(), indent=2))
            return 0
        if args.graph_command == "neighbors":
            print(json.dumps(graph.neighbors(args.entity), indent=2))
            return 0
    if args.command == "obsess":
        loop = ObsessionLoop(store, args.domain, args.config)
        if args.obsess_command == "fetch":
            print(json.dumps(loop.fetch(args.limit_per_source), indent=2))
            return 0
        if args.obsess_command == "review":
            print(json.dumps(loop.review(args.limit), indent=2))
            return 0
        if args.obsess_command == "approve":
            print(json.dumps(loop.approve(args.pending_id), indent=2))
            return 0
        if args.obsess_command == "reject":
            print(json.dumps(loop.reject(args.pending_id), indent=2))
            return 0
        if args.obsess_command == "run-once":
            print(json.dumps(loop.run_once(args.limit_per_source), indent=2))
            return 0
    if args.command == "train-lora":
        print(json.dumps(train_lora(args.domain, args.db, args.output_dir, args.model_id), indent=2))
        return 0
    if args.command == "benchmark":
        result = run_benchmark(args.file, args.db, args.domain, args.top_k)
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = results_dir / f"results_{timestamp}.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Results saved to: {output_path}")
        return 0
    return 1


def asdict_like(payload: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


if __name__ == "__main__":
    raise SystemExit(main())
