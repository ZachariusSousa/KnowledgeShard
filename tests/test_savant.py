from pathlib import Path
import unittest
from uuid import uuid4

from knowledgeshard.benchmark import score_answer
from knowledgeshard.cli import build_parser, infer_domain
from knowledgeshard.graph import KnowledgeGraph
from knowledgeshard.ingest import normalize_links, source_document_from_crawl_result
from knowledgeshard.model_runtime import ModelConfig, OptionalModelRuntime, load_dotenv
from knowledgeshard.models import PendingFact, ResearchChunk, ResearchNote, SourceCandidate, SourceDocument
from knowledgeshard.obsession import (
    DocumentFetcher,
    FactExtractor,
    ObsessionLoop,
    ResearchProfile,
    SearchResult,
    SourceScorer,
    StructuredExtractor,
    fact_quality_error,
    parse_confidence,
    parse_duckduckgo_results,
    required_terms_present,
)
from knowledgeshard.research import ResearchAgent, is_research_noise, novelty_score, parse_research_note, supports_angle
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
        loop = ObsessionLoop(store, "mario-kart-wii", config_path="data/test/missing-profile.json", search_provider=FakeSearch())

        result = loop.discover(query_limit=1, results_per_query=3)
        sources = loop.sources()

        self.assertEqual(result["discovered"], 1)
        self.assertEqual(len(sources), 1)
        self.assertIn("mkwii-drift", sources[0]["url"])

    def test_crawl4ai_result_normalizes_markdown_document(self):
        class FakeCrawlResult:
            success = True
            markdown = "# Mario Kart Wii\n\nManual drift enables mini-turbos."
            metadata = {"title": "Mario Kart Wii Manual Drift", "sourceURL": "https://example.test/final"}
            links = [{"href": "/wiki/Mini-Turbo", "text": "Mini-Turbo"}]

        source = SourceCandidate(
            id="source-1",
            url="https://example.test/start",
            title="Fallback title",
            domain="mario-kart-wii",
            discovery_query="test",
        )

        document = source_document_from_crawl_result(source, FakeCrawlResult())
        links = normalize_links(FakeCrawlResult.links, document.url)

        self.assertEqual(document.url, "https://example.test/final")
        self.assertEqual(document.title, "Mario Kart Wii Manual Drift")
        self.assertIn("Manual drift enables mini-turbos.", document.full_text)
        self.assertEqual(document.text_excerpt, document.full_text)
        self.assertEqual(links[0].url, "https://example.test/wiki/Mini-Turbo")

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
        store.upsert_source_candidate(
            SourceCandidate(
                id="source-1",
                url="https://example.test/mkwii-learning",
                title="Mario Kart Wii Learning",
                domain="mario-kart-wii",
                obsession="Mario Kart Wii",
                discovery_query="test",
                trust_score=0.9,
                relevance_score=0.9,
            )
        )
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

    def test_parse_confidence_accepts_words_and_bad_values(self):
        self.assertEqual(parse_confidence("low"), 0.35)
        self.assertEqual(parse_confidence("high"), 0.85)
        self.assertEqual(parse_confidence("not numeric"), 0.55)

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
        store.upsert_source_candidate(
            SourceCandidate(
                id="source-1",
                url="https://example.test/source",
                title="Mario Kart Wii Source",
                domain="mario-kart-wii",
                obsession="Mario Kart Wii",
                discovery_query="test",
                trust_score=0.9,
                relevance_score=0.9,
            )
        )
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
                        evidence_text="Mario Kart Wii Auto Fact states high confidence learning enters the graph.",
                        extraction_method="model",
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
                        evidence_text="Mario Kart Wii Pending Fact states low confidence learning waits for review.",
                        extraction_method="model",
                        tags=("obsession", "extracted"),
                    ),
                ], None

        store = KnowledgeStore(self.db_path())
        store.upsert_source_candidate(
            SourceCandidate(
                id="source-1",
                url="https://example.test/mkwii-learning",
                title="Mario Kart Wii Learning",
                domain="mario-kart-wii",
                obsession="Mario Kart Wii",
                discovery_query="test",
                trust_score=0.9,
                relevance_score=0.9,
            )
        )
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
                        evidence_text="Mario Kart Wii Learned Fact states run logs should show learned facts.",
                        extraction_method="model",
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
                        evidence_text="DuckDuckGo is a search engine.",
                        extraction_method="model",
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
                        evidence_text="Mario Kart Wii Lightning shrinks opponents during races.",
                        extraction_method="model",
                        tags=("obsession", "extracted"),
                    ),
                    PendingFact(
                        id="bad-section-placeholder",
                        subject="Mario Kart Wii",
                        relation="has_gameplay",
                        object="Details in Gameplay section",
                        confidence=1.0,
                        source=document.url,
                        domain=document.domain,
                        evidence_text="Details in Gameplay section",
                        extraction_method="model",
                        tags=("obsession", "extracted"),
                    ),
                ], None

        store = KnowledgeStore(self.db_path())
        store.upsert_source_candidate(
            SourceCandidate(
                id="source-1",
                url="https://example.test/lightning",
                title="Mario Kart Wii Lightning",
                domain="mario-kart-wii",
                obsession="Mario Kart Wii",
                discovery_query="test",
                trust_score=0.9,
                relevance_score=0.9,
            )
        )
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
        self.assertIn("section placeholder", " ".join(result["errors"]))

    def test_infer_domain_reuses_existing_domain_from_obsession(self):
        domain = infer_domain(
            "Mario Kart Wii competitive karts",
            ["trains", "mario-kart-wii"],
        )

        self.assertEqual(domain, "mario-kart-wii")

    def test_infer_domain_stops_before_focus_words(self):
        domain = infer_domain("Mario Kart Wii competitive karts", [])

        self.assertEqual(domain, "mario-kart-wii")

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

    def test_crawl4ai_link_normalization_resolves_relative_urls(self):
        links = normalize_links(
            [{"href": "/wiki/Manual_Drift", "text": "Manual drift"}],
            "https://example.test/root",
        )

        self.assertEqual(links[0].url, "https://example.test/wiki/Manual_Drift")
        self.assertEqual(links[0].title, "Manual drift")

    def test_source_scorer_uses_profile_trust_and_denies_patterns(self):
        profile = ResearchProfile(
            domain="generic-domain",
            allowed_hosts=("trusted.test",),
            denied_url_patterns=("search",),
            source_trust={"trusted.test": 0.91},
        )
        scorer = SourceScorer("generic-domain", "trusted topic", profile)

        trusted = scorer.score(SearchResult("Trusted topic", "https://trusted.test/page"), "trusted topic")
        denied = scorer.score(SearchResult("Search", "https://trusted.test/search?q=topic"), "trusted topic")

        self.assertEqual(trusted.trust_score, 0.91)
        self.assertEqual(trusted.status, "candidate")
        self.assertEqual(denied.status, "rejected")

    def test_table_fact_with_unknown_relation_is_not_auto_approved(self):
        store = KnowledgeStore(self.db_path())
        store.upsert_source_candidate(
            SourceCandidate(
                id="source-1",
                url="https://trusted.test/file",
                title="File history",
                domain="robotics",
                discovery_query="robotics",
                trust_score=0.9,
                relevance_score=0.9,
            )
        )
        document = SourceDocument(
            source_id="source-1",
            url="https://trusted.test/file",
            title="File history",
            text_excerpt="current 16:36 ExampleUser",
            content_hash="file-hash",
            domain="robotics",
        )
        store.add_source_document(document)
        loop = ObsessionLoop(store, "robotics", config_path="data/test/missing-profile.json")
        loop.document_tables[document.content_hash] = [[["Name", "Date Time"], ["current", "16:36"]]]

        result = loop.extract(auto_approve=True)

        self.assertEqual(result["auto_approved"], 0)
        self.assertIn("discarded current has_date_time", " ".join(result["errors"]))

    def test_structured_extractor_reads_generic_table(self):
        profile = ResearchProfile(domain="robotics")
        document = SourceDocument(
            source_id="source-1",
            url="https://trusted.test/robots",
            title="Robotics glossary",
            text_excerpt="Servo torque high",
            content_hash="robots-hash",
            domain="robotics",
        )
        table = [["Name", "Property"], ["Servo", "high torque"]]

        facts = StructuredExtractor("robotics", profile).extract(document, [table])

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Servo")
        self.assertEqual(facts[0].relation, "has_property")
        self.assertEqual(facts[0].object, "high torque")
        self.assertEqual(facts[0].extraction_method, "table")

    def test_evidence_validation_rejects_unsupported_fact(self):
        document = SourceDocument(
            source_id="source-1",
            url="https://trusted.test/page",
            title="Robotics glossary",
            text_excerpt="Servo enables precise movement.",
            content_hash="evidence-hash",
            domain="robotics",
        )
        fact = PendingFact(
            id="unsupported",
            subject="Sensor",
            relation="enables",
            object="precise movement",
            source=document.url,
            domain=document.domain,
            evidence_text="Servo enables precise movement.",
        )

        self.assertEqual(fact_quality_error(fact, document, "robotics"), "subject not grounded in document")

    def test_profile_required_terms_reject_broader_series_fact(self):
        profile = ResearchProfile(domain="mario-kart-wii", required_terms=("mario", "kart", "wii"))
        document = SourceDocument(
            source_id="source-1",
            url="https://www.mariowiki.com/Mario_Kart",
            title="Mario Kart",
            text_excerpt="First installment Super Mario Kart (1992).",
            content_hash="series-hash",
            domain="mario-kart-wii",
        )
        fact = PendingFact(
            id="series-fact",
            subject="Super Mario Kart",
            relation="first_install",
            object="Mario Kart series",
            source=document.url,
            domain=document.domain,
            evidence_text="First installment Super Mario Kart (1992).",
        )

        self.assertEqual(
            fact_quality_error(fact, document, "Mario Kart Wii competitive karts", profile),
            "missing required domain terms",
        )

    def test_required_terms_present_all_terms(self):
        profile = ResearchProfile(domain="mario-kart-wii", required_terms=("mario", "kart", "wii"))

        self.assertTrue(required_terms_present("Mario Kart Wii vehicle stats", profile))
        self.assertFalse(required_terms_present("Mario Kart series", profile))

    def test_generic_autonomous_run_crawls_and_auto_approves_grounded_fact(self):
        class EmptySearch:
            def search(self, query, limit=5):
                return []

        class FakeFetcher(DocumentFetcher):
            def fetch(self, source, max_chars=12000):
                if source.url.endswith("/seed"):
                    self.last_links = [SearchResult("Robotics glossary", "https://trusted.test/glossary")]
                    self.last_tables = []
                    return SourceDocument(
                        source_id=source.id,
                        url=source.url,
                        title="Robotics seed",
                        text_excerpt="Robotics glossary links to servo facts.",
                        content_hash="seed-hash",
                        domain=source.domain,
                        obsession=source.obsession,
                    )
                self.last_links = []
                self.last_tables = [[["Name", "Property"], ["Servo", "precise movement"]]]
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title="Robotics glossary",
                    text_excerpt="Servo precise movement.",
                    content_hash="glossary-hash",
                    domain=source.domain,
                    obsession=source.obsession,
                )

        class NoModelExtractor:
            def extract(self, document, limit=5):
                return [], None

        store = KnowledgeStore(self.db_path())
        profile_path = Path("data/test") / f"{uuid4().hex}.json"
        profile_path.write_text(
            """
            {
              "research_profile": {
                "domain": "robotics",
                "seed_urls": ["https://trusted.test/seed"],
                "allowed_hosts": ["trusted.test"],
                "source_trust": {"trusted.test": 0.9},
                "default_crawl_depth": 1
              }
            }
            """,
            encoding="utf-8",
        )
        self._db_paths.append(profile_path)
        loop = ObsessionLoop(
            store,
            "robotics",
            obsession="robotics servo movement",
            config_path=profile_path,
            search_provider=EmptySearch(),
            fetcher=FakeFetcher(),
            extractor=NoModelExtractor(),
        )

        first = loop.run_once(budget=2, auto_approve=True)
        second = loop.run_once(budget=3, auto_approve=True)

        self.assertGreaterEqual(first["links_discovered"], 1)
        self.assertEqual(second["auto_approved"], 1)
        self.assertEqual(store.count_facts("robotics"), 1)

    def test_crawler_does_not_boost_irrelevant_links_with_agenda_text(self):
        store = KnowledgeStore(self.db_path())
        profile = ResearchProfile(
            domain="robotics",
            allowed_hosts=("trusted.test",),
            source_trust={"trusted.test": 0.9},
        )
        loop = ObsessionLoop(store, "robotics", obsession="robotics servo movement", config_path="data/test/missing-profile.json")
        loop.profile = profile
        loop.scorer = SourceScorer("robotics", "robotics servo movement", profile)
        source = SourceCandidate(
            id="seed",
            url="https://trusted.test/seed",
            title="Robotics seed",
            domain="robotics",
            discovery_query="robotics servo movement",
            trust_score=0.9,
            relevance_score=1.0,
        )

        discovered = loop._discover_links(
            source,
            [
                SearchResult("Unrelated release list", "https://trusted.test/nintendo_selects"),
                SearchResult("Servo movement guide", "https://trusted.test/servo-movement"),
            ],
            limit=5,
        )
        urls = [item["url"] for item in loop.sources(10)]

        self.assertEqual(discovered, 1)
        self.assertEqual(urls, ["https://trusted.test/servo-movement"])

    def test_profile_required_terms_filter_crawl_links(self):
        store = KnowledgeStore(self.db_path())
        profile = ResearchProfile(
            domain="mario-kart-wii",
            allowed_hosts=("mariowiki.com",),
            required_terms=("mario", "kart", "wii"),
            source_trust={"mariowiki.com": 0.9},
        )
        loop = ObsessionLoop(store, "mario-kart-wii", obsession="Mario Kart Wii competitive karts", config_path="data/test/missing-profile.json")
        loop.profile = profile
        loop.scorer = SourceScorer("mario-kart-wii", "Mario Kart Wii competitive karts", profile)
        source = SourceCandidate(
            id="seed",
            url="https://www.mariowiki.com/Mario_Kart_Wii",
            title="Mario Kart Wii",
            domain="mario-kart-wii",
            discovery_query="Mario Kart Wii competitive karts",
            trust_score=0.9,
            relevance_score=1.0,
        )

        discovered = loop._discover_links(
            source,
            [
                SearchResult("Mario Kart", "https://www.mariowiki.com/Mario_Kart"),
                SearchResult("Mario Kart Wii vehicles", "https://www.mariowiki.com/Mario_Kart_Wii_vehicles"),
            ],
            limit=5,
        )
        urls = [item["url"] for item in loop.sources(10)]

        self.assertEqual(discovered, 1)
        self.assertEqual(urls, ["https://www.mariowiki.com/Mario_Kart_Wii_vehicles"])

    def test_research_agent_stores_findings_and_report(self):
        class FakeSearch:
            def search(self, query, limit=5):
                return [SearchResult("Servo movement guide", "https://trusted.test/servo")]

        class FakeFetcher(DocumentFetcher):
            def fetch(self, source, max_chars=12000):
                if source.url.endswith("/seed"):
                    self.last_links = [SearchResult("Servo movement guide", "https://trusted.test/servo")]
                    self.last_tables = []
                    return SourceDocument(
                        source_id=source.id,
                        url=source.url,
                        title="Robotics seed",
                        text_excerpt="Robotics seed page links to servo movement guide.",
                        content_hash="robotics-seed",
                        domain=source.domain,
                        obsession=source.obsession,
                    )
                self.last_links = []
                self.last_tables = []
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title="Servo movement guide",
                    text_excerpt=(
                        "Servo movement is notable because precise control enables advanced robotics strategies. "
                        "Competitive robotics teams often use fast servo response for reliable mechanisms."
                    ),
                    content_hash="servo-research",
                    domain=source.domain,
                    obsession=source.obsession,
                )

        profile_path = Path("data/test") / f"{uuid4().hex}.json"
        profile_path.write_text(
            """
            {
              "research_profile": {
                "domain": "robotics",
                "seed_urls": ["https://trusted.test/seed"],
                "allowed_hosts": ["trusted.test"],
                "source_trust": {"trusted.test": 0.9}
              }
            }
            """,
            encoding="utf-8",
        )
        self._db_paths.append(profile_path)
        store = KnowledgeStore(self.db_path())
        agent = ResearchAgent(
            store,
            "robotics",
            "servo movement",
            config_path=str(profile_path),
            search_provider=FakeSearch(),
            fetcher=FakeFetcher(),
        )

        result = agent.run_once(budget=3)

        self.assertGreaterEqual(result["findings_added"], 1)
        self.assertGreaterEqual(result["links_discovered"], 1)
        self.assertTrue(agent.findings())
        self.assertTrue(agent.reports())
        self.assertIn("next_questions", result)

    def test_novelty_score_rewards_interesting_topic_overlap(self):
        score = novelty_score(
            "Competitive robotics teams use fast servo response because it improves reliable mechanisms.",
            "servo movement",
            "Servo movement guide",
        )

        self.assertGreater(score, 0.25)

    def test_research_filters_angle_mismatch_and_metadata_noise(self):
        self.assertTrue(
            supports_angle(
                "Several advanced drift techniques are retained in Mario Kart Wii.",
                "Mario Kart Wii competitive karts drift mini turbo mechanics",
            )
        )
        self.assertTrue(
            supports_angle(
                "The Flame Runner has high speed, weight, drift, and mini-turbo stats.",
                "Mario Kart Wii competitive vehicles vehicle stats",
            )
        )
        self.assertFalse(
            supports_angle(
                "Mario Kart Wii has a new vehicle type: bikes.",
                "Mario Kart Wii competitive vehicles vehicle stats",
            )
        )
        self.assertFalse(
            supports_angle(
                "The Mario Kart Channel allows tournaments.",
                "Mario Kart Wii competitive karts vehicle stats",
            )
        )
        self.assertTrue(is_research_noise("Developer Nintendo EAD Publisher Nintendo Platform Wii Release dates April 2008."))
        self.assertTrue(is_research_noise("1.1.4.1 Lightweight 1.1.4.2 Medium weight 1.1.8 Mario Kart Wii 1.1.8.1 Small karts."))

    def test_source_document_persists_full_text_and_deduplicates(self):
        store = KnowledgeStore(self.db_path())
        document = SourceDocument(
            id="doc-full",
            source_id="source-1",
            url="https://trusted.test/full",
            title="Full document",
            text_excerpt="short excerpt",
            full_text="short excerpt plus the rest of the locally stored readable document",
            content_hash="full-hash",
            domain="robotics",
            obsession="servo movement",
        )

        self.assertTrue(store.add_source_document(document))
        self.assertFalse(store.add_source_document(document))
        stored = store.list_source_documents("robotics", 1)[0]

        self.assertIn("rest of the locally stored", stored.full_text)
        self.assertEqual(stored.text_excerpt, "short excerpt")

    def test_research_chunks_notes_and_retry_state_are_persisted(self):
        store = KnowledgeStore(self.db_path())
        chunk = ResearchChunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="Servo movement enables precise robotics control.",
            char_count=48,
            token_count=6,
            topic="servo movement",
            domain="robotics",
            priority=0.9,
        )
        note = ResearchNote(
            id="note-1",
            chunk_id="chunk-1",
            document_id="doc-1",
            topic="servo movement",
            domain="robotics",
            summary="Servo movement supports precise control.",
            claims=("Servo movement enables precise robotics control.",),
            entities=("Servo",),
            relations=("Servo enables control",),
            questions=("Which servos are fastest?",),
            evidence_quotes=("Servo movement enables precise robotics control.",),
            confidence=0.82,
            source="https://trusted.test/servo",
        )

        self.assertTrue(store.add_research_chunk(chunk))
        self.assertFalse(store.add_research_chunk(chunk))
        store.update_research_chunk_status("chunk-1", "failed", error="bad json", increment_attempts=True)
        failed = store.list_research_chunks("robotics", "servo movement", "failed", 1)[0]
        self.assertEqual(failed.attempts, 1)
        self.assertEqual(failed.error, "bad json")
        self.assertTrue(store.add_research_note(note))
        self.assertFalse(store.add_research_note(note))

        stored_note = store.list_research_notes("robotics", "servo movement", 1)[0]
        self.assertEqual(stored_note.claims, note.claims)
        self.assertEqual(stored_note.evidence_quotes, note.evidence_quotes)

    def test_research_process_validates_json_and_clamps_confidence(self):
        chunk = ResearchChunk(
            id="chunk-json",
            document_id="doc-json",
            chunk_index=0,
            text="Servo movement enables precise robotics control.",
            char_count=48,
            token_count=6,
            topic="servo movement",
            domain="robotics",
        )

        note = parse_research_note(
            """
            {
              "summary": "Servo movement supports precise control.",
              "claims": ["Servo movement enables precise robotics control."],
              "entities": ["Servo"],
              "relations": ["Servo enables control"],
              "questions": ["Which servos are fastest?"],
              "evidence_quotes": ["Servo movement enables precise robotics control."],
              "confidence": 2.5
            }
            """,
            chunk,
        )

        self.assertEqual(note.confidence, 1.0)
        self.assertEqual(note.claims[0], "Servo movement enables precise robotics control.")
        with self.assertRaises(ValueError):
            parse_research_note("not json", chunk)

    def test_research_process_leaves_unavailable_model_chunks_pending(self):
        store = KnowledgeStore(self.db_path())
        store.add_research_chunk(
            ResearchChunk(
                id="chunk-pending",
                document_id="doc-1",
                chunk_index=0,
                text="Servo movement enables precise robotics control.",
                char_count=48,
                token_count=6,
                topic="servo movement",
                domain="robotics",
            )
        )
        agent = ResearchAgent(
            store,
            "robotics",
            "servo movement",
            model_runtime=OptionalModelRuntime(ModelConfig(enabled=False)),
        )

        result = agent.process(chunks=1)

        self.assertEqual(result["processed"], 0)
        self.assertIn("model runtime", result["error"])
        self.assertEqual(store.list_research_chunks("robotics", "servo movement", "pending", 1)[0].id, "chunk-pending")

    def test_research_synthesize_creates_pending_not_approved_fact(self):
        store = KnowledgeStore(self.db_path())
        store.add_research_note(
            ResearchNote(
                id="note-synth",
                chunk_id="chunk-synth",
                document_id="doc-synth",
                topic="servo movement",
                domain="robotics",
                summary="Servo movement supports precise control.",
                claims=("Servo movement enables precise robotics control.",),
                questions=("Which servo type is most reliable?",),
                evidence_quotes=("Servo movement enables precise robotics control.",),
                confidence=0.79,
                source="https://trusted.test/servo",
            )
        )
        agent = ResearchAgent(store, "robotics", "servo movement")

        result = agent.synthesize(limit=10)

        self.assertEqual(result["pending_added"], 1)
        self.assertEqual(store.count_pending_facts("robotics"), 1)
        self.assertEqual(store.count_facts("robotics"), 0)
        pending = store.list_pending_facts("robotics")[0]
        self.assertEqual(pending.relation, "claims")
        self.assertIn("precise robotics control", pending.object)

    def test_research_cli_staged_commands_parse(self):
        parser = build_parser()

        args = parser.parse_args(["--domain", "robotics", "research", "--topic", "servo movement", "ingest", "--budget", "2"])
        self.assertEqual(args.command, "research")
        self.assertEqual(args.research_command, "ingest")
        self.assertEqual(args.budget, 2)
        args = parser.parse_args(["--domain", "robotics", "research", "--topic", "servo movement", "crawl-status"])
        self.assertEqual(args.research_command, "crawl-status")

        args = parser.parse_args(["obsess", "run-once", "--budget", "1"])
        self.assertEqual(args.command, "obsess")
        self.assertEqual(args.obsess_command, "run-once")


if __name__ == "__main__":
    unittest.main()
