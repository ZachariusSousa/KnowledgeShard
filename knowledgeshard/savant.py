"""Local single-savant runtime."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from uuid import uuid4

from .models import Citation, Correction, Fact, QueryResponse
from .model_runtime import OptionalModelRuntime
from .retrieval import score_prepared, tokenize
from .storage import KnowledgeStore


class Savant:
    def __init__(
        self,
        domain: str = "mario-kart-wii",
        store: KnowledgeStore | None = None,
        savant_id: str | None = None,
        model_runtime: OptionalModelRuntime | None = None,
    ) -> None:
        self.domain = domain
        self.store = store or KnowledgeStore()
        self.savant_id = savant_id or f"local-{domain}-savant"
        self.model_runtime = model_runtime or OptionalModelRuntime()

    def add_fact(
        self,
        subject: str,
        relation: str,
        object: str,
        confidence: float = 0.8,
        source: str = "manual",
        tags: tuple[str, ...] = (),
    ) -> Fact:
        fact = Fact(
            subject=subject,
            relation=relation,
            object=object,
            confidence=max(0.0, min(confidence, 1.0)),
            source=source,
            domain=self.domain,
            tags=tags,
        )
        self.store.upsert_fact(fact)
        return fact

    def query(self, question: str, num_experts: int = 3, timeout_seconds: int = 30) -> QueryResponse:
        facts = self.store.list_facts(self.domain)
        corpus = [fact.text for fact in facts]
        corpus_tokens = [set(tokenize(item)) for item in corpus]
        ranked = sorted(
            ((score_prepared(question, fact.text, corpus_tokens), fact) for fact in facts),
            key=lambda item: (item[0], item[1].confidence),
            reverse=True,
        )
        relevant = [(match_score, fact) for match_score, fact in ranked if match_score > 0][: max(num_experts, 1)]

        query_id = uuid4().hex
        if not relevant:
            response = QueryResponse(
                query_id=query_id,
                question=question,
                answer=(
                    f"I do not have enough {self.domain} knowledge to answer that yet. "
                    "Add a fact or run an obsession ingest, then ask again."
                ),
                confidence=0.0,
                citations=(),
                savant_id=self.savant_id,
                domain=self.domain,
            )
            self.store.log_query(query_id, question, response.answer, 0.0, [])
            return response

        citations = tuple(
            Citation(
                fact_id=fact.id,
                source=fact.source,
                confidence=fact.confidence,
                excerpt=fact.text,
            )
            for match_score, fact in relevant
        )
        confidence = round(mean((fact.confidence * 0.7) + (match_score * 0.3) for match_score, fact in relevant), 3)
        fact_lines = " ".join(f"{fact.subject} {fact.relation} {fact.object}." for _, fact in relevant)
        model_answer = self._model_answer(question, citations)
        answer = model_answer or (
            f"As the {self.domain} savant, my best answer is: {fact_lines} "
            f"Overall confidence: {confidence:.2f}."
        )
        response = QueryResponse(
            query_id=query_id,
            question=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            savant_id=self.savant_id,
            domain=self.domain,
        )
        self.store.log_query(
            query_id,
            question,
            answer,
            confidence,
            [asdict(citation) for citation in citations],
        )
        return response

    def _model_answer(self, question: str, citations: tuple[Citation, ...]) -> str | None:
        if not self.model_runtime.config.enabled:
            return None
        evidence = "\n".join(f"- {citation.excerpt} (source: {citation.source})" for citation in citations)
        prompt = (
            "You are a local Mario Kart Wii savant. Answer only from the cited evidence. "
            "If evidence is insufficient, say what is uncertain.\n\n"
            f"Question: {question}\nEvidence:\n{evidence}\nAnswer:"
        )
        generated = self.model_runtime.generate(prompt, max_new_tokens=220)
        if not generated:
            return None
        return f"{generated}\n\nCitations: " + "; ".join(citation.source for citation in citations)

    def correct(self, query_id: str, correction: str, confidence: float = 1.0) -> Correction:
        saved = Correction(
            query_id=query_id,
            savant_id=self.savant_id,
            correction=correction,
            confidence=max(0.0, min(confidence, 1.0)),
        )
        self.store.add_correction(saved)
        self.add_fact(
            subject="User correction",
            relation="states",
            object=correction,
            confidence=saved.confidence,
            source=f"correction:{query_id}",
            tags=("correction",),
        )
        return saved

    def status(self) -> dict:
        return {
            "savant_id": self.savant_id,
            "domain": self.domain,
            "status": "ready",
            "fact_count": self.store.count_facts(self.domain),
            "pending_fact_count": self.store.count_pending_facts(self.domain),
            "model_enabled": self.model_runtime.config.enabled,
            "model_backend": self.model_runtime.config.backend,
            "model_loaded": self.model_runtime.backend_loaded or self.model_runtime._pipeline is not None,
            "model_error": self.model_runtime.error,
        }

    def model_status(self) -> dict:
        return self.model_runtime.status()

    def metrics(self) -> dict:
        return {
            "queries_answered": self.store.count_queries(),
            "corrections_received": self.store.count_corrections(),
            "facts_in_domain": self.store.count_facts(self.domain),
            "pending_facts": self.store.count_pending_facts(self.domain),
            "entities_in_domain": self.store.count_entities(self.domain),
            "relations_in_domain": self.store.count_relations(self.domain),
        }
