from pathlib import Path
import unittest
from uuid import uuid4

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
        load_seed_facts("data/seeds/trains.json", store)
        savant = Savant(store=store)

        response = savant.query("Why do trains derail in monsoons?")

        self.assertGreater(response.confidence, 0)
        self.assertTrue(response.citations)
        self.assertIn("derail", response.answer.lower())

    def test_correction_is_stored_as_fact(self):
        store = KnowledgeStore(self.db_path())
        savant = Savant(store=store)

        correction = savant.correct("query-1", "Rails should be inspected after flooding.", 0.95)
        facts = store.list_facts("trains")

        self.assertEqual(correction.query_id, "query-1")
        self.assertEqual(store.count_corrections(), 1)
        self.assertIn("flooding", facts[0].object)


if __name__ == "__main__":
    unittest.main()
