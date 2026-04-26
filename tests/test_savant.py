from pathlib import Path
import unittest
from uuid import uuid4

from knowledgeshard.benchmark import score_answer
from knowledgeshard.graph import KnowledgeGraph
from knowledgeshard.models import PendingFact
from knowledgeshard.obsession import ObsessionLoop
from knowledgeshard.savant import Savant
from knowledgeshard.seed import load_seed_facts
from knowledgeshard.storage import KnowledgeStore


class SavantTests(unittest.TestCase):
    def db_path(self) -> Path:
        path = Path("data/test") / f"{uuid4().hex}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_seed_and_query_returns_citations(self):
        store = KnowledgeStore(self.db_path())
        load_seed_facts("data/seeds/mario_kart_wii.json", store, "mario-kart-wii")
        savant = Savant(domain="mario-kart-wii", store=store)

        response = savant.query("What is the advantage of manual drift in Mario Kart Wii?")

        self.assertGreater(response.confidence, 0)
        self.assertTrue(response.citations)
        self.assertIn("mini-turbo", response.answer.lower())
        self.assertGreaterEqual(store.count_facts("mario-kart-wii"), 1000)

    def test_correction_is_stored_as_fact(self):
        store = KnowledgeStore(self.db_path())
        savant = Savant(domain="mario-kart-wii", store=store)

        correction = savant.correct("query-1", "Rails should be inspected after flooding.", 0.95)
        facts = store.list_facts("mario-kart-wii")

        self.assertEqual(correction.query_id, "query-1")
        self.assertEqual(store.count_corrections(), 1)
        self.assertIn("flooding", facts[0].object)

    def test_pending_fact_is_not_queried_until_approved(self):
        store = KnowledgeStore(self.db_path())
        pending = PendingFact(
            id="pending-shortcut",
            subject="DK Summit double shortcut",
            relation="requires",
            object="a Mushroom and precise alignment",
            domain="mario-kart-wii",
            tags=("strategy",),
        )
        self.assertTrue(store.add_pending_fact(pending))
        savant = Savant(domain="mario-kart-wii", store=store)

        before = savant.query("How does the DK Summit double shortcut work?")
        self.assertFalse(before.citations)

        store.approve_pending_fact("pending-shortcut")
        after = savant.query("How does the DK Summit double shortcut work?")
        self.assertTrue(after.citations)
        self.assertIn("precise alignment", after.answer)

    def test_graph_projection_uses_approved_facts(self):
        store = KnowledgeStore(self.db_path())
        savant = Savant(domain="mario-kart-wii", store=store)
        savant.add_fact("Manual drift", "enables", "mini-turbos", tags=("drift",))
        store.add_pending_fact(
            PendingFact(
                id="pending-unapproved",
                subject="Unapproved rumor",
                relation="claims",
                object="not in graph",
                domain="mario-kart-wii",
            )
        )

        graph = KnowledgeGraph(store, "mario-kart-wii")

        self.assertGreaterEqual(graph.stats()["edges"], 1)
        self.assertEqual(graph.neighbors("Unapproved rumor"), [])
        self.assertEqual(graph.neighbors("Manual drift")[0]["target"], "mini-turbos")

    def test_obsession_fetch_deduplicates_pending_facts(self):
        store = KnowledgeStore(self.db_path())
        loop = ObsessionLoop(store, "mario-kart-wii")

        def fake_fetch(source, limit):
            return [{"title": "New time trial route", "link": "https://example.test/route", "summary": "A route note."}]

        loop._fetch_source = fake_fetch
        first = loop.fetch()
        second = loop.fetch()

        self.assertEqual(first["pending_added"], 1)
        self.assertEqual(second["pending_added"], 0)
        self.assertEqual(store.count_pending_facts("mario-kart-wii"), 1)

    def test_benchmark_scoring_accepts_key_points_and_rejects_forbidden_claims(self):
        item = {
            "question": "What does Lightning do?",
            "required_key_points": ["Lightning", "shrinks opponents"],
            "forbidden_claims": ["targets first place"],
            "expected_citation_tags": ["lightning"],
            "threshold": 0.85,
        }
        good = score_answer(
            "Lightning shrinks opponents and makes them drop items.",
            [{"source": "seed:items", "excerpt": "Lightning shrinks opponents"}],
            item,
        )
        bad = score_answer(
            "Lightning targets first place.",
            [{"source": "seed:items", "excerpt": "Lightning"}],
            item,
        )

        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])


if __name__ == "__main__":
    unittest.main()
