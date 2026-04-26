"""FastAPI app matching the public MVP API in Plan.md."""

from __future__ import annotations

from dataclasses import asdict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised only when API deps are absent.
    FastAPI = None
    BaseModel = object
    Field = None

from .graph import KnowledgeGraph
from .savant import Savant
from .storage import KnowledgeStore

store = KnowledgeStore()
savant = Savant(store=store)
app = FastAPI(title="KnowledgeShard Local Savant", version="0.1.0") if FastAPI else None


class QueryRequest(BaseModel):
    if Field:
        question: str = Field(min_length=1)
        num_experts: int = Field(default=3, ge=1, le=5)
        timeout_seconds: int = Field(default=30, ge=1, le=120)


class CorrectionRequest(BaseModel):
    if Field:
        query_id: str = Field(min_length=1)
        savant_id: str = Field(min_length=1)
        correction: str = Field(min_length=1)
        confidence: float = Field(default=1.0, ge=0.0, le=1.0)


if app:

    @app.post("/query")
    async def query(request: QueryRequest) -> dict:
        return asdict(
            savant.query(
                question=request.question,
                num_experts=request.num_experts,
                timeout_seconds=request.timeout_seconds,
            )
        )


    @app.post("/correct")
    async def correct(request: CorrectionRequest) -> dict:
        correction = savant.correct(
            query_id=request.query_id,
            correction=request.correction,
            confidence=request.confidence,
        )
        return {"received": True, "correction": asdict(correction)}


    @app.get("/status")
    async def status() -> dict:
        return savant.status()


    @app.get("/metrics")
    async def metrics() -> dict:
        return savant.metrics()


    @app.get("/graph/entities/{entity}")
    async def graph_entity(entity: str) -> dict:
        return {"entity": entity, "neighbors": KnowledgeGraph(store, savant.domain).neighbors(entity)}


    @app.get("/obsession/pending")
    async def pending() -> dict:
        return {"pending": [item.__dict__ for item in store.list_pending_facts(savant.domain)]}
