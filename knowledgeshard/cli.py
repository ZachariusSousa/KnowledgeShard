"""Command line interface for the local savant."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .model_runtime import train_lora
from .research import ResearchAgent
from .savant import Savant
from .seed import load_seed_facts
from .storage import KnowledgeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledgeshard", description="Run a local single-domain savant.")
    parser.add_argument("--db", default="data/knowledgeshard.db", help="SQLite database path.")
    parser.add_argument("--domain", default=None, help="Savant domain. Defaults to an inferred or existing domain.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Load seed facts into local storage.")
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
    subparsers.add_parser("model-status", help="Show local model runtime status.")
    subparsers.add_parser("metrics", help="Show MVP metrics.")

    research = subparsers.add_parser("research", help="Explore a topic and store interesting research notes.")
    research.add_argument("--config", default="config/mario_kart_wii.sources.json")
    research.add_argument("--topic", required=True, help="Natural-language subject to explore.")
    research_subparsers = research.add_subparsers(dest="research_command", required=True)
    research_ingest = research_subparsers.add_parser("ingest", help="Discover and store raw readable documents.")
    research_ingest.add_argument("--budget", type=int, default=8)
    research_chunk = research_subparsers.add_parser("chunk", help="Split stored documents into pending research chunks.")
    research_chunk.add_argument("--limit", type=int, default=20)
    research_chunk.add_argument("--chunk-chars", type=int, default=3200)
    research_chunk.add_argument("--overlap-chars", type=int, default=300)
    research_process = research_subparsers.add_parser("process", help="Slowly process pending chunks with the local LLM.")
    research_process.add_argument("--chunks", type=int, default=10)
    research_notes = research_subparsers.add_parser("notes", help="List extracted research notes.")
    research_notes.add_argument("--limit", type=int, default=20)
    research_synthesize = research_subparsers.add_parser("synthesize", help="Promote research notes into pending facts.")
    research_synthesize.add_argument("--limit", type=int, default=20)
    research_subparsers.add_parser("crawl-status", help="Show Crawl4AI ingestion backend status.")
    train = subparsers.add_parser("train-lora", help="Train a PEFT LoRA adapter from approved facts.")
    train.add_argument("--output-dir", default="weights/lora")
    train.add_argument("--model-id", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = KnowledgeStore(args.db)
    domain = resolve_domain(args, store)
    savant = Savant(domain=domain, store=store)

    if args.command == "seed":
        count = load_seed_facts(Path(args.file), store, domain)
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
    if args.command == "model-status":
        print(asdict_like(savant.model_status()))
        return 0
    if args.command == "metrics":
        print(asdict_like(savant.metrics()))
        return 0
    if args.command == "research":
        agent = ResearchAgent(store, domain, args.topic, config_path=args.config)
        if args.research_command == "ingest":
            print(json.dumps(agent.ingest(args.budget), indent=2))
            return 0
        if args.research_command == "chunk":
            print(json.dumps(agent.chunk(args.limit, args.chunk_chars, args.overlap_chars), indent=2))
            return 0
        if args.research_command == "process":
            print(json.dumps(agent.process(args.chunks), indent=2))
            return 0
        if args.research_command == "notes":
            print(json.dumps(agent.notes(args.limit), indent=2))
            return 0
        if args.research_command == "crawl-status":
            print(json.dumps(agent.crawl_status(), indent=2))
            return 0
        if args.research_command == "synthesize":
            print(json.dumps(agent.synthesize(args.limit), indent=2))
            return 0
    if args.command == "train-lora":
        print(json.dumps(train_lora(domain, args.db, args.output_dir, args.model_id), indent=2))
        return 0
    return 1


def asdict_like(payload: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


FOCUS_WORDS = {
    "advanced",
    "competitive",
    "facts",
    "guide",
    "guides",
    "items",
    "mechanic",
    "mechanics",
    "meta",
    "route",
    "routes",
    "shortcuts",
    "strategy",
    "tech",
    "tips",
    "vehicle",
    "vehicles",
    "wiki",
}


def resolve_domain(args: argparse.Namespace, store: KnowledgeStore) -> str:
    if args.domain:
        return args.domain
    topic = getattr(args, "topic", None)
    if topic:
        return infer_domain(topic, store.list_domains())
    domains = store.list_domains()
    if domains:
        return domains[0]
    if args.command == "seed":
        return seed_domain(Path(args.file))
    return "mario-kart-wii"


def seed_domain(path: Path) -> str:
    return slugify(path.stem) or "general"


def infer_domain(obsession: str, existing_domains: list[str]) -> str:
    obsession_terms = set(slug_terms(obsession))
    best_domain = ""
    best_score = 0.0
    for domain in existing_domains:
        domain_terms = set(slug_terms(domain))
        if not domain_terms:
            continue
        score = len(obsession_terms & domain_terms) / len(domain_terms)
        if score > best_score:
            best_domain = domain
            best_score = score
    if best_domain and best_score >= 0.6:
        return best_domain

    topic_terms: list[str] = []
    for term in slug_terms(obsession):
        if term in FOCUS_WORDS and topic_terms:
            break
        topic_terms.append(term)
    return "-".join(topic_terms[:6]) or "general"


def slug_terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", value.lower()) if term]


def slugify(value: str) -> str:
    return "-".join(slug_terms(value))


if __name__ == "__main__":
    raise SystemExit(main())
