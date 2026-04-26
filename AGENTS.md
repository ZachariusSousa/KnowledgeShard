# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

KnowledgeShard is a local Python MVP for a single-domain "savant" knowledge system. It stores facts in SQLite, retrieves relevant facts with lightweight lexical scoring, returns cited answers, and records user corrections as new facts.

The larger product vision in `Plan.md` is a local-first distributed savant network: many specialized experts, peer-to-peer gossip, query routing to the best domain experts, reputation, privacy-aware routing, and continual learning. The codebase currently implements the single-savant foundation, so keep changes compatible with that future without prematurely building the full distributed system.

Core modules:

- `knowledgeshard/models.py`: immutable dataclasses for facts, citations, query responses, and corrections.
- `knowledgeshard/storage.py`: SQLite persistence and schema initialization.
- `knowledgeshard/retrieval.py`: stdlib TF-IDF-style lexical scoring.
- `knowledgeshard/savant.py`: main runtime orchestration for add/query/correct/status/metrics.
- `knowledgeshard/seed.py`: JSON seed loader.
- `knowledgeshard/cli.py`: command-line interface.
- `knowledgeshard/server.py`: zero-dependency HTTP server.
- `knowledgeshard/api.py`: optional FastAPI app.
- `data/seeds/trains.json`: checked-in seed facts.
- `tests/test_savant.py`: unittest coverage for seed/query/correction flows.

## Product Direction From Plan.md

- Specialization matters more than broad generality. Prefer domain-specific facts, citations, confidence, and correction flows over generic chatbot behavior.
- Local-first is a core constraint. Personal queries and model/runtime state should stay local by default.
- Future phases include richer knowledge graph entities/relations, gossip sync, CRDT-style conflict handling, query routing, response aggregation, reputation, and feedback propagation.
- Sources and confidence are not optional decoration. Answers should remain grounded in stored facts and citations, and uncertain knowledge should stay visibly uncertain.
- User corrections are part of the learning loop. Treat them as durable knowledge with provenance, not as transient UI feedback.

## Environment

This project targets Python 3.10+ syntax and intentionally keeps the core runtime on the standard library. Optional API dependencies are listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Do not assume third-party packages are available unless they are in `requirements.txt` or the task explicitly adds them.

## Common Commands

Run tests:

```powershell
python -m unittest
```

Seed the default local database:

```powershell
python -m knowledgeshard.cli seed
```

Ask a question through the CLI:

```powershell
python -m knowledgeshard.cli ask "Why do trains derail in monsoons?"
```

Run the stdlib HTTP server:

```powershell
python -m knowledgeshard.server --host 127.0.0.1 --port 8080
```

Run the FastAPI app, if optional dependencies are installed:

```powershell
python -m uvicorn knowledgeshard.api:app --host 127.0.0.1 --port 8080
```

## Coding Guidelines

- Keep core behavior dependency-light. Prefer the standard library for storage, retrieval, parsing, and tests unless a dependency is justified.
- Preserve the dataclass-based model style and immutable model instances where practical.
- Keep `KnowledgeStore` responsible for persistence details; avoid SQLite calls from higher-level modules when adding features.
- Keep `Savant` focused on orchestration: domain behavior, query flow, correction flow, and metric/status composition.
- Retrieval is deliberately transparent and deterministic. Changes to ranking or token normalization should include focused tests.
- Keep answers citation-first. Do not add behavior that fabricates details beyond retrieved facts without marking the uncertainty.
- Clamp user-provided confidence values to the `[0.0, 1.0]` range, matching existing behavior.
- Use `Path` for filesystem paths and keep default data paths under `data/`.
- Store generated runtime databases under ignored paths such as `data/*.db` or `data/test/`.
- When adding fields or tables, favor shapes that can evolve toward `Plan.md` concepts: entities, relations, source refs, version metadata, and conflict state.
- Avoid broad refactors unless they are needed for the requested behavior.

## Testing Notes

- Add or update `unittest` tests in `tests/` for behavior changes.
- Tests should use temporary or unique database files under `data/test/`; that directory is ignored by git.
- Query behavior should be asserted through public APIs such as `Savant.query`, `Savant.correct`, and `KnowledgeStore` methods rather than direct DB inspection unless schema behavior is under test.
- After changing storage, retrieval, seed loading, CLI behavior, or API response shape, run:

```powershell
python -m unittest
```

## Data And Persistence

- Seed data is JSON with a top-level `facts` list.
- Each fact should include `subject`, `relation`, and `object`; optional fields include `id`, `confidence`, `source`, `domain`, and `tags`.
- Preserve provenance when ingesting or correcting data. Prefer explicit `source` values and tags over anonymous facts.
- Plan.md's long-term graph model includes entities, relations, source references, timestamps, confidence, version vectors, and disputed/conflict states. Use that direction when designing migrations, but keep the current MVP schema simple unless the task requires expansion.
- Do not commit generated SQLite databases, server logs, virtual environments, caches, or test databases.
- Be careful when editing `data/seeds/trains.json`; retrieval tests and examples may depend on the train-domain vocabulary.

## API Notes

- `knowledgeshard/server.py` is the no-extra-dependencies HTTP API and should keep working without FastAPI installed.
- `knowledgeshard/api.py` is optional and should continue to import safely when FastAPI/Pydantic are missing.
- Keep response payloads dataclass-friendly so they can be serialized with `dataclasses.asdict`.
- Keep public API changes aligned with the planned local API shape: `/query`, `/correct`, `/status`, and `/metrics`.
- If future networked routing is added, provide a local-only path for sensitive queries before sending anything to peers.

## Security And Privacy

- Default to local storage and local inference assumptions.
- Do not introduce telemetry, remote calls, peer sync, or external ingestion without an explicit feature request and clear user control.
- Treat personal queries as sensitive. Future routing code should be privacy-aware and should support local-only behavior.
- Knowledge graph data may eventually be shared over P2P gossip; keep personal or secret data out of seed facts and shared correction payloads.
- If adding network protocols later, design for signed messages, authenticated peers, and explicit deletion/blacklist handling as described in `Plan.md`.

## Before Finishing Changes

1. Run `python -m unittest` for code changes.
2. Check `git status --short` and make sure only intentional files changed.
3. Mention any tests that could not be run and why.
