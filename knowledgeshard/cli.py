"""Command line interface for the local savant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .savant import Savant
from .seed import load_seed_facts
from .storage import KnowledgeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledgeshard", description="Run a local single-domain savant.")
    parser.add_argument("--db", default="data/knowledgeshard.db", help="SQLite database path.")
    parser.add_argument("--domain", default="trains", help="Savant domain.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Load seed facts into the knowledge graph.")
    seed.add_argument("--file", default="data/seeds/trains.json", help="Seed JSON file.")

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
    return 1


def asdict_like(payload: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


if __name__ == "__main__":
    raise SystemExit(main())
