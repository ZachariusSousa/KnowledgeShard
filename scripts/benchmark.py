"""Run the KnowledgeShard benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledgeshard.benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a gold-rubric KnowledgeShard benchmark.")
    parser.add_argument("--domain", default="mario-kart-wii")
    parser.add_argument("--db", default="data/knowledgeshard.db")
    parser.add_argument("--file", default="benchmarks/mario_kart_wii_10.json")
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = run_benchmark(args.file, args.db, args.domain, args.top_k)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = results_dir / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Results saved to: {output_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
