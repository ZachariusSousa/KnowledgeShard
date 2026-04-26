from pathlib import Path
import unittest
from uuid import uuid4

from knowledgeshard.benchmark import score_answer
from knowledgeshard.graph import KnowledgeGraph
from knowledgeshard.model_runtime import ModelConfig, OptionalModelRuntime, load_dotenv
from knowledgeshard.models import PendingFact, SourceCandidate, SourceDocument
from knowledgeshard.obsession import DocumentFetcher, FactExtractor, ObsessionLoop, SearchResult, parse_duckduckgo_results
from knowledgeshard.savant import Savant
from knowledgeshard.seed import load_seed_facts
from knowledgeshard.storage import KnowledgeStore


class SavantTests(unittest.TestCase):
    def setUp(self):
        Path("data/test").mkdir(parents=True, exist_ok=True)
        self._db_paths: list[Path] = []

    def tearDown(self):
        for path in self._db_paths:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + "-journal").unlink(missing_ok=True)

    def db_path(self) -> Path:
        path = Path("data/test") / f"{uuid4().hex}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_paths.append(path)
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

    def test_enabled_model_runtime_writes_final_answer_from_model(self):
        class FakeRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))
                self.backend_loaded = True

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                self.last_prompt = prompt
                return "Manual drift is useful because it can charge mini-turbos from the cited evidence."

        store = KnowledgeStore(self.db_path())
        load_seed_facts("data/seeds/mario_kart_wii.json", store, "mario-kart-wii")
        runtime = FakeRuntime()
        savant = Savant(domain="mario-kart-wii", store=store, model_runtime=runtime)

        response = savant.query("What is the advantage of manual drift in Mario Kart Wii?")

        self.assertIn("Manual drift is useful", response.answer)
        self.assertIn("Citations:", response.answer)
        self.assertIn("Evidence:", runtime.last_prompt)

    def test_dotenv_loader_sets_missing_model_environment(self):
        env_path = Path("data/test") / f"{uuid4().hex}.env"
        env_path.write_text("KS_ENABLE_MODEL=1\nKS_MODEL_BACKEND=ollama\n", encoding="utf-8")
        self._db_paths.append(env_path)
        old_enable = __import__("os").environ.pop("KS_ENABLE_MODEL", None)
        old_backend = __import__("os").environ.pop("KS_MODEL_BACKEND", None)
        try:
            load_dotenv(env_path)
            config = ModelConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.backend, "ollama")
        finally:
            if old_enable is not None:
                __import__("os").environ["KS_ENABLE_MODEL"] = old_enable
            else:
                __import__("os").environ.pop("KS_ENABLE_MODEL", None)
            if old_backend is not None:
                __import__("os").environ["KS_MODEL_BACKEND"] = old_backend
            else:
                __import__("os").environ.pop("KS_MODEL_BACKEND", None)

    def test_discovery_deduplicates_and_scores_sources(self):
        class FakeSearch:
            def search(self, query, limit=5):
                return [
                    SearchResult("Mario Kart Wii Manual Drift Guide", "https://example.test/mkwii-drift"),
                    SearchResult("Mario Kart Wii Manual Drift Guide", "https://example.test/mkwii-drift"),
                    SearchResult("Unrelated Cooking Notes", "https://example.test/cooking"),
                ]

        store = KnowledgeStore(self.db_path())
        savant = Savant(domain="mario-kart-wii", store=store)
        savant.add_fact("Manual drift", "enables", "mini-turbos", tags=("drift",))
        loop = ObsessionLoop(store, "mario-kart-wii", search_provider=FakeSearch())

        result = loop.discover(query_limit=1, results_per_query=3)
        sources = loop.sources()

        self.assertEqual(result["discovered"], 1)
        self.assertEqual(len(sources), 1)
        self.assertIn("mkwii-drift", sources[0]["url"])

    def test_document_fetcher_extracts_readable_html(self):
        fetcher = DocumentFetcher()
        text = fetcher.readable_text(
            "<html><head><style>.x{}</style></head><body><h1>Mario Kart Wii</h1>"
            "<script>ignore()</script><p>Manual drift enables mini-turbos.</p></body></html>"
        )

        self.assertIn("Mario Kart Wii", text)
        self.assertIn("Manual drift enables mini-turbos.", text)
        self.assertNotIn("ignore", text)

    def test_fact_extractor_validates_model_json(self):
        class FakeRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                return """
                [
                  {
                    "subject": "Mario Kart Wii manual drift",
                    "relation": "enables",
                    "object": "mini-turbos after sustained drifting",
                    "confidence": 0.82,
                    "tags": ["drift", "mechanics"]
                  }
                ]
                """

        store = KnowledgeStore(self.db_path())
        document = SourceDocument(
            source_id="source-1",
            url="https://example.test/mkwii-drift",
            title="Mario Kart Wii Manual Drift",
            text_excerpt="Mario Kart Wii manual drift enables mini-turbos after sustained drifting.",
            content_hash="hash",
            domain="mario-kart-wii",
        )

        facts, error = FactExtractor(store, "mario-kart-wii", FakeRuntime()).extract(document)

        self.assertIsNone(error)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source, document.url)
        self.assertIn("mechanics", facts[0].tags)

    def test_run_once_discovers_fetches_and_stages_pending_facts(self):
        class FakeSearch:
            def search(self, query, limit=5):
                return [SearchResult("Mario Kart Wii Lightning Guide", "https://example.test/lightning")]

        class FakeFetcher:
            def fetch(self, source, max_chars=12000):
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title=source.title,
                    text_excerpt="Mario Kart Wii Lightning shrinks opponents and can create shock dodges.",
                    content_hash="lightning-hash",
                    domain=source.domain,
                )

        class FakeExtractor:
            def extract(self, document, limit=5):
                return [
                    PendingFact(
                        id="extracted-lightning",
                        subject="Mario Kart Wii Lightning",
                        relation="shrinks",
                        object="opponents and can create shock dodge opportunities",
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted", "lightning"),
                    )
                ], None

        store = KnowledgeStore(self.db_path())
        loop = ObsessionLoop(
            store,
            "mario-kart-wii",
            search_provider=FakeSearch(),
            fetcher=FakeFetcher(),
            extractor=FakeExtractor(),
        )

        result = loop.run_once(budget=1)

        self.assertEqual(result["pending_added"], 1)
        self.assertEqual(store.count_pending_facts("mario-kart-wii"), 1)

    def test_parse_duckduckgo_results_decodes_redirects(self):
        page = (
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fmkwii">'
            "Mario Kart Wii Guide</a>"
        )

        results = parse_duckduckgo_results(page)

        self.assertEqual(results[0].url, "https://example.test/mkwii")

    def test_discovery_fallback_creates_sources_when_search_returns_empty(self):
        class EmptySearch:
            def search(self, query, limit=5):
                return []

        store = KnowledgeStore(self.db_path())
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii", search_provider=EmptySearch())

        result = loop.discover(query_limit=1, results_per_query=2)

        self.assertGreater(result["discovered"], 0)
        self.assertTrue(loop.sources())

    def test_auto_approve_high_confidence_and_stage_low_confidence(self):
        class FakeExtractor:
            def extract(self, document, limit=5):
                return [
                    PendingFact(
                        id="auto-fact",
                        subject="Mario Kart Wii Auto Fact",
                        relation="states",
                        object="high confidence learning enters the graph",
                        confidence=0.92,
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted"),
                    ),
                    PendingFact(
                        id="pending-fact",
                        subject="Mario Kart Wii Pending Fact",
                        relation="states",
                        object="low confidence learning waits for review",
                        confidence=0.55,
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted"),
                    ),
                ], None

        store = KnowledgeStore(self.db_path())
        document = SourceDocument(
            source_id="source-1",
            url="https://example.test/mkwii-learning",
            title="Mario Kart Wii Learning",
            text_excerpt=(
                "Mario Kart Wii Auto Fact states high confidence learning enters the graph. "
                "Mario Kart Wii Pending Fact states low confidence learning waits for review."
            ),
            content_hash="learn-hash",
            domain="mario-kart-wii",
            obsession="Mario Kart Wii",
        )
        store.add_source_document(document)
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii", extractor=FakeExtractor())

        result = loop.extract(auto_approve=True, auto_confidence_threshold=0.8)

        self.assertEqual(result["auto_approved"], 1)
        self.assertEqual(result["pending_added"], 1)
        self.assertEqual(store.count_facts("mario-kart-wii"), 1)
        self.assertEqual(store.count_pending_facts("mario-kart-wii"), 1)
        self.assertIn("auto-approved", store.list_facts("mario-kart-wii")[0].tags)

    def test_run_logs_and_learned_logs_are_visible(self):
        class FakeSearch:
            def search(self, query, limit=5):
                return [SearchResult("Mario Kart Wii Source", "https://example.test/source")]

        class FakeFetcher:
            def fetch(self, source, max_chars=12000):
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title=source.title,
                    text_excerpt="Mario Kart Wii Learned Fact states run logs should show learned facts.",
                    content_hash="source-hash",
                    domain=source.domain,
                    obsession=source.obsession,
                )

        class FakeExtractor:
            def extract(self, document, limit=5):
                return [
                    PendingFact(
                        id="learned-fact",
                        subject="Mario Kart Wii Learned Fact",
                        relation="states",
                        object="run logs should show learned facts",
                        confidence=0.91,
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted"),
                    )
                ], None

        store = KnowledgeStore(self.db_path())
        loop = ObsessionLoop(
            store,
            "mario-kart-wii",
            obsession="Mario Kart Wii",
            search_provider=FakeSearch(),
            fetcher=FakeFetcher(),
            extractor=FakeExtractor(),
        )

        result = loop.run_once(budget=1, auto_approve=True)

        self.assertEqual(result["auto_approved"], 1)
        self.assertTrue(loop.learned())
        self.assertTrue(loop.documents())
        self.assertTrue(loop.runs())

    def test_daemon_can_run_one_cycle(self):
        class EmptySearch:
            def search(self, query, limit=5):
                return []

        store = KnowledgeStore(self.db_path())
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii", search_provider=EmptySearch())

        results = loop.run_daemon(budget=1, interval_minutes=0.01, max_cycles=1)

        self.assertEqual(len(results), 1)
        self.assertIn("discovered", results[0])

    def test_auto_approve_discards_search_page_and_placeholder_facts(self):
        class FakeExtractor:
            def extract(self, document, limit=5):
                return [
                    PendingFact(
                        id="bad-search",
                        subject="DuckDuckGo",
                        relation="search_engine",
                        object="none",
                        confidence=1.0,
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted"),
                    ),
                    PendingFact(
                        id="good-grounded",
                        subject="Mario Kart Wii Lightning",
                        relation="shrinks",
                        object="opponents",
                        confidence=0.93,
                        source=document.url,
                        domain=document.domain,
                        tags=("obsession", "extracted"),
                    ),
                ], None

        store = KnowledgeStore(self.db_path())
        document = SourceDocument(
            source_id="source-1",
            url="https://example.test/lightning",
            title="Mario Kart Wii Lightning",
            text_excerpt="Mario Kart Wii Lightning shrinks opponents during races.",
            content_hash="quality-hash",
            domain="mario-kart-wii",
            obsession="Mario Kart Wii",
        )
        store.add_source_document(document)
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii", extractor=FakeExtractor())

        result = loop.extract(auto_approve=True)

        self.assertEqual(result["auto_approved"], 1)
        self.assertEqual(store.count_facts("mario-kart-wii"), 1)
        self.assertIn("discarded DuckDuckGo", " ".join(result["errors"]))

    def test_fetch_skips_search_result_sources(self):
        class FakeFetcher:
            def fetch(self, source, max_chars=12000):
                raise AssertionError("search result source should not be fetched")

        store = KnowledgeStore(self.db_path())
        store.upsert_source_candidate(
            SourceCandidate(
                id="search-source",
                url="https://duckduckgo.com/html/?q=Mario+Kart+Wii",
                title="Mario Kart Wii search",
                domain="mario-kart-wii",
                obsession="Mario Kart Wii",
                discovery_query="Mario Kart Wii",
                relevance_score=1.0,
            )
        )
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii", fetcher=FakeFetcher())

        result = loop.fetch_documents(limit=10)

        self.assertIn("skipped search result page", " ".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
