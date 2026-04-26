"""Knowledge graph projection helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .storage import KnowledgeStore

try:  # pragma: no cover - covered by fallback tests when networkx is absent.
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


class KnowledgeGraph:
    def __init__(self, store: KnowledgeStore, domain: str = "mario-kart-wii") -> None:
        self.store = store
        self.domain = domain

    def build(self) -> Any:
        relations = self.store.list_relations(self.domain)
        if nx:
            graph = nx.MultiDiGraph(domain=self.domain)
            for relation in relations:
                graph.add_node(relation.subject)
                graph.add_node(relation.object)
                graph.add_edge(
                    relation.subject,
                    relation.object,
                    predicate=relation.predicate,
                    fact_id=relation.fact_id,
                    confidence=relation.confidence,
                    source=relation.source,
                )
            return graph

        fallback: dict[str, list[dict[str, object]]] = defaultdict(list)
        for relation in relations:
            fallback[relation.subject].append(
                {
                    "target": relation.object,
                    "predicate": relation.predicate,
                    "fact_id": relation.fact_id,
                    "confidence": relation.confidence,
                    "source": relation.source,
                }
            )
        return dict(fallback)

    def stats(self) -> dict:
        graph = self.build()
        if nx:
            return {
                "domain": self.domain,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "backend": "networkx",
            }
        nodes = set(graph)
        edges = 0
        for source, outgoing in graph.items():
            nodes.add(source)
            edges += len(outgoing)
            for edge in outgoing:
                nodes.add(str(edge["target"]))
        return {"domain": self.domain, "nodes": len(nodes), "edges": edges, "backend": "stdlib"}

    def neighbors(self, entity: str) -> list[dict[str, object]]:
        graph = self.build()
        if nx:
            if entity not in graph:
                return []
            neighbors: list[dict[str, object]] = []
            for _, target, data in graph.out_edges(entity, data=True):
                neighbors.append(
                    {
                        "source": entity,
                        "predicate": data["predicate"],
                        "target": target,
                        "fact_id": data["fact_id"],
                        "confidence": data["confidence"],
                        "citation": data["source"],
                    }
                )
            return neighbors
        return [
            {
                "source": entity,
                "predicate": edge["predicate"],
                "target": edge["target"],
                "fact_id": edge["fact_id"],
                "confidence": edge["confidence"],
                "citation": edge["source"],
            }
            for edge in graph.get(entity, [])
        ]
