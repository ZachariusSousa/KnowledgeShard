from pathlib import Path
import http.client
import json
import sqlite3
import threading
import time
import unittest
from uuid import uuid4

from knowledgeshard.cli import build_parser, infer_domain
from knowledgeshard.extraction import FactExtractor, StructuredExtractor, evidence_hash, fact_quality_error, parse_confidence
from knowledgeshard.ingest import normalize_links, source_document_from_crawl_result
from knowledgeshard.model_runtime import ModelConfig, OptionalModelRuntime, load_dotenv
from knowledgeshard.models import PendingFact, ResearchChunk, SourceCandidate, SourceDocument
from knowledgeshard.research import ResearchAgent, ResearchJobManager, parse_research_note
from knowledgeshard.savant import Savant
from knowledgeshard.seed import load_seed_facts
from knowledgeshard.server import SavantRequestHandler
from knowledgeshard.sources import ResearchProfile, SearchResult, SourceScorer, parse_duckduckgo_results, required_terms_present
from knowledgeshard.storage import KnowledgeStore
from http.server import ThreadingHTTPServer


class KnowledgeShardTests(unittest.TestCase):
    def setUp(self):
        Path("data/test").mkdir(parents=True, exist_ok=True)
        self._paths: list[Path] = []

    def tearDown(self):
        for path in self._paths:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + "-journal").unlink(missing_ok=True)

    def db_path(self) -> Path:
        path = Path("data/test") / f"{uuid4().hex}.db"
        self._paths.append(path)
        return path

    def test_seed_query_and_correction_flow(self):
        store = KnowledgeStore(self.db_path())
        load_seed_facts("data/seeds/mario_kart_wii.json", store, "mario-kart-wii")
        savant = Savant(domain="mario-kart-wii", store=store)

        response = savant.query("What is the advantage of manual drift in Mario Kart Wii?")
        correction = savant.correct("query-1", "Rails should be inspected after flooding.", 0.95)

        self.assertTrue(response.citations)
        self.assertIn("mini-turbo", response.answer.lower())
        self.assertEqual(correction.confidence, 0.95)
        self.assertEqual(store.count_corrections(), 1)

    def test_fresh_schema_contains_only_active_tables(self):
        path = self.db_path()
        KnowledgeStore(path)
        db = sqlite3.connect(path)
        try:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            db.close()

        self.assertEqual(
            tables,
            {
                "corrections",
                "facts",
                "pending_facts",
                "query_log",
                "research_chunks",
                "research_notes",
                "research_synthesis_runs",
                "source_candidates",
                "source_documents",
            },
        )

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
        self.assertEqual(links[0].url, "https://example.test/wiki/Mini-Turbo")

    def test_source_profile_scoring_and_search_parsing(self):
        profile = ResearchProfile(
            domain="mario-kart-wii",
            allowed_hosts=("mariowiki.com",),
            required_terms=("mario", "kart", "wii"),
            source_trust={"mariowiki.com": 0.9},
        )
        scorer = SourceScorer("mario-kart-wii", "Mario Kart Wii vehicles", profile)

        trusted = scorer.score(SearchResult("Mario Kart Wii vehicles", "https://www.mariowiki.com/Mario_Kart_Wii_vehicles"), "vehicles")
        rejected = scorer.score(SearchResult("Mario Kart", "https://www.mariowiki.com/Mario_Kart"), "vehicles")
        page = '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fmkwii">Mario Kart Wii Guide</a>'

        self.assertEqual(trusted.status, "candidate")
        self.assertGreater(trusted.trust_score, 0.8)
        self.assertTrue(required_terms_present("Mario Kart Wii vehicle stats", profile))
        self.assertFalse(required_terms_present("Mario Kart series", profile))
        self.assertEqual(rejected.status, "candidate")
        self.assertEqual(parse_duckduckgo_results(page)[0].url, "https://example.test/mkwii")

    def test_fact_extraction_validates_model_json(self):
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
                    "evidence_text": "Mario Kart Wii manual drift enables mini-turbos after sustained drifting.",
                    "tags": ["drift", "mechanics"]
                  }
                ]
                """

        document = SourceDocument(
            source_id="source-1",
            url="https://example.test/mkwii-drift",
            title="Mario Kart Wii Manual Drift",
            text_excerpt="Mario Kart Wii manual drift enables mini-turbos after sustained drifting.",
            content_hash="hash",
            domain="mario-kart-wii",
        )

        facts, error = FactExtractor("mario-kart-wii", "manual drift", FakeRuntime()).extract(document)

        self.assertIsNone(error)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source, document.url)
        self.assertIn("mechanics", facts[0].tags)

    def test_structured_extractor_and_quality_checks(self):
        profile = ResearchProfile(domain="robotics")
        document = SourceDocument(
            source_id="source-1",
            url="https://trusted.test/robots",
            title="Robotics glossary",
            text_excerpt="Servo enables precise movement.",
            content_hash="robots-hash",
            domain="robotics",
        )
        table = [["Name", "Property"], ["Servo", "precise movement"]]
        facts = StructuredExtractor("robotics", profile).extract(document, [table])
        unsupported = PendingFact(
            id="unsupported",
            subject="Sensor",
            relation="enables",
            object="precise movement",
            source=document.url,
            domain=document.domain,
            evidence_text="Servo enables precise movement.",
        )

        self.assertEqual(facts[0].subject, "Servo")
        self.assertEqual(facts[0].relation, "has_property")
        self.assertEqual(fact_quality_error(unsupported, document, "robotics"), "subject not grounded in document")
        self.assertEqual(parse_confidence("high"), 0.85)
        self.assertEqual(parse_confidence("not numeric"), 0.55)

    def test_research_ingest_chunk_process_and_synthesize(self):
        class FakeSearch:
            def search(self, query, limit=5):
                return [SearchResult("Servo movement guide", "https://trusted.test/servo")]

        class FakeFetcher:
            last_links = []
            last_tables = []

            def fetch(self, source, max_chars=12000):
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title="Servo movement guide",
                    text_excerpt="Servo movement enables precise robotics control.",
                    full_text="Servo movement enables precise robotics control. Evidence should stay local.",
                    content_hash="servo-hash",
                    domain=source.domain,
                    obsession=source.obsession,
                )

        class FakeRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                return """
                {
                  "summary": "Servo movement supports precise control.",
                  "claims": ["Servo movement enables precise robotics control."],
                  "entities": ["Servo"],
                  "relations": ["Servo enables control"],
                  "questions": ["Which servos are fastest?"],
                  "evidence_quotes": ["Servo movement enables precise robotics control."],
                  "confidence": 0.82
                }
                """

        profile_path = Path("data/test") / f"{uuid4().hex}.json"
        profile_path.write_text(
            """
            {
              "research_profile": {
                "domain": "robotics",
                "seed_urls": ["https://trusted.test/servo"],
                "allowed_hosts": ["trusted.test"],
                "source_trust": {"trusted.test": 0.9}
              }
            }
            """,
            encoding="utf-8",
        )
        self._paths.append(profile_path)
        store = KnowledgeStore(self.db_path())
        agent = ResearchAgent(
            store,
            "robotics",
            "servo movement",
            config_path=str(profile_path),
            search_provider=FakeSearch(),
            fetcher=FakeFetcher(),
            model_runtime=FakeRuntime(),
        )

        ingest = agent.ingest(budget=1)
        chunk = agent.chunk(limit=1, chunk_chars=500, overlap_chars=0)
        process = agent.process(chunks=1)
        synthesize = agent.synthesize(limit=1)

        self.assertEqual(ingest["stored"], 1)
        self.assertEqual(chunk["chunks_created"], 1)
        self.assertEqual(process["notes_added"], 1)
        self.assertEqual(synthesize["pending_added"], 1)
        self.assertEqual(store.count_pending_facts("robotics"), 1)

    def test_research_process_repairs_missing_comma_json_without_model_retry(self):
        class RepairRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))
                self.calls = 0

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                self.calls += 1
                if self.calls == 1:
                    return '{"summary": "Servo movement supports control", "claims": ["Servo movement enables control" "Servo improves precision"]}'
                raise AssertionError("deterministic JSON repair should avoid model retry")

        store = KnowledgeStore(self.db_path())
        store.add_research_chunk(
            ResearchChunk(
                id="repair-chunk",
                document_id="doc-1",
                chunk_index=0,
                text="Servo movement enables control.",
                char_count=31,
                token_count=4,
                topic="servo movement",
                domain="robotics",
            )
        )
        runtime = RepairRuntime()

        result = ResearchAgent(store, "robotics", "servo movement", model_runtime=runtime).process(chunks=1)

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["notes_added"], 1)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(store.get_research_chunk("repair-chunk").status, "processed")

    def test_research_process_repairs_invalid_json_escape_without_model_retry(self):
        class EscapeRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))
                self.calls = 0

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                self.calls += 1
                if self.calls == 1:
                    return r"""
                    {
                      "summary": "Servo movement supports control.",
                      "claims": ["Servo movement enables control\precision."],
                      "entities": ["Servo"],
                      "relations": ["Servo enables control"],
                      "questions": [],
                      "evidence_quotes": ["Servo movement enables control\precision."],
                      "confidence": 0.86
                    }
                    """
                raise AssertionError("deterministic JSON repair should avoid model retry")

        store = KnowledgeStore(self.db_path())
        store.add_research_chunk(
            ResearchChunk(
                id="escape-chunk",
                document_id="doc-1",
                chunk_index=0,
                text="Servo movement enables control precision.",
                char_count=41,
                token_count=5,
                topic="servo movement",
                domain="robotics",
            )
        )
        runtime = EscapeRuntime()

        result = ResearchAgent(store, "robotics", "servo movement", model_runtime=runtime).process(chunks=1)

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["notes_added"], 1)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(store.get_research_chunk("escape-chunk").status, "processed")

    def test_research_process_repairs_unescaped_inner_quotes_without_model_retry(self):
        class QuoteRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))
                self.calls = 0

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                self.calls += 1
                if self.calls == 1:
                    return """
                    {
                      "summary": "The "Mission Mode" entry describes a removed mode.",
                      "claims": ["The "Mission Mode" entry describes a removed mode."],
                      "entities": ["Mission Mode"],
                      "relations": ["Mission Mode describes removed content"],
                      "questions": [],
                      "evidence_quotes": ["The "Mission Mode" entry describes a removed mode."],
                      "confidence": 0.86
                    }
                    """
                raise AssertionError("deterministic JSON repair should avoid model retry")

        store = KnowledgeStore(self.db_path())
        store.add_research_chunk(
            ResearchChunk(
                id="quote-chunk",
                document_id="doc-1",
                chunk_index=0,
                text="Mission Mode describes removed content.",
                char_count=39,
                token_count=5,
                topic="mission mode",
                domain="mario-kart-wii",
            )
        )
        runtime = QuoteRuntime()

        result = ResearchAgent(store, "mario-kart-wii", "mission mode", model_runtime=runtime).process(chunks=1)

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["notes_added"], 1)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(store.get_research_chunk("quote-chunk").status, "processed")

    def test_research_auto_approve_keeps_weak_facts_pending(self):
        store = KnowledgeStore(self.db_path())
        high = PendingFact(
            id="high",
            subject="servo movement",
            relation="claims",
            object="Servo movement enables precise robotics control.",
            confidence=0.91,
            source="chunk:1",
            domain="robotics",
            tags=("research", "synthesized", "pending-review"),
            evidence_text="Servo movement enables precise robotics control.",
            evidence_hash=evidence_hash("Servo movement enables precise robotics control."),
            extraction_method="research-note",
        )
        low = PendingFact(
            id="low",
            subject="servo movement",
            relation="claims",
            object="Servo movement may improve control.",
            confidence=0.5,
            source="chunk:2",
            domain="robotics",
            tags=("research", "synthesized", "pending-review"),
            evidence_text="Servo movement may improve control.",
            evidence_hash=evidence_hash("Servo movement may improve control."),
            extraction_method="research-note",
        )
        missing_evidence = PendingFact(
            id="missing-evidence",
            subject="servo movement",
            relation="claims",
            object="Servo movement has undocumented benefits.",
            confidence=0.95,
            source="chunk:3",
            domain="robotics",
            tags=("research", "synthesized", "pending-review"),
            extraction_method="research-note",
        )
        store.add_pending_fact(high)
        store.add_pending_fact(low)
        store.add_pending_fact(missing_evidence)

        result = ResearchAgent(store, "robotics", "servo movement").auto_approve(threshold=0.82)
        facts = store.list_facts("robotics")
        pending = store.list_pending_facts("robotics")

        self.assertEqual(result["approved"], 1)
        self.assertEqual(facts[0].id, "high")
        self.assertIn("auto-approved", facts[0].tags)
        self.assertEqual({fact.id for fact in pending}, {"low", "missing-evidence"})

    def test_research_job_manager_runs_background_cycle(self):
        profile_path = self.research_profile_path()
        store = KnowledgeStore(self.db_path())
        manager = ResearchJobManager(
            store,
            agent_factory=lambda store, domain, topic, config_path: self.fake_research_agent(store, domain, topic, config_path),
        )

        job = manager.start("servo movement", "robotics", str(profile_path), max_cycles=1)
        snapshot = self.wait_for_job(manager, job.id)

        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["phase"], "completed")
        self.assertEqual(snapshot["cycles_completed"], 1)
        self.assertGreaterEqual(snapshot["counters"]["stored"], 1)
        self.assertGreaterEqual(snapshot["counters"]["notes_added"], 1)
        self.assertEqual(store.count_facts("robotics"), 1)
        self.assertEqual(store.count_pending_facts("robotics"), 0)

    def test_server_serves_research_ui_and_job_routes(self):
        profile_path = self.research_profile_path()
        store = KnowledgeStore(self.db_path())
        old_store = SavantRequestHandler.store
        old_savant = SavantRequestHandler.savant
        old_jobs = SavantRequestHandler.research_jobs
        old_config = SavantRequestHandler.default_config
        SavantRequestHandler.store = store
        SavantRequestHandler.savant = Savant(domain="robotics", store=store)
        SavantRequestHandler.default_config = str(profile_path)
        SavantRequestHandler.research_jobs = ResearchJobManager(
            store,
            agent_factory=lambda store, domain, topic, config_path: self.fake_research_agent(store, domain, topic, config_path),
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), SavantRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/")
            response = conn.getresponse()
            html = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("KnowledgeShard Research", html)

            body = json.dumps({"topic": "servo movement", "domain": "robotics"}).encode("utf-8")
            conn.request("POST", "/research/start", body, {"content-type": "application/json"})
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 201)
            self.assertIn("job_id", payload)

            snapshot = self.wait_for_job(SavantRequestHandler.research_jobs, payload["job_id"])
            conn.request("GET", f"/research/status?job_id={payload['job_id']}")
            response = conn.getresponse()
            status_payload = json.loads(response.read().decode("utf-8"))

            self.assertEqual(snapshot["status"], "completed")
            self.assertEqual(response.status, 200)
            self.assertEqual(status_payload["job"]["status"], "completed")
            self.assertEqual(status_payload["job"]["approved_count"], 1)
            self.assertTrue(status_payload["job"]["notes"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            SavantRequestHandler.store = old_store
            SavantRequestHandler.savant = old_savant
            SavantRequestHandler.research_jobs = old_jobs
            SavantRequestHandler.default_config = old_config

    def test_research_note_parser_clamps_confidence(self):
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

    def test_dotenv_loader_and_cli_parse(self):
        env_path = Path("data/test") / f"{uuid4().hex}.env"
        env_path.write_text("KS_ENABLE_MODEL=1\nKS_MODEL_BACKEND=ollama\n", encoding="utf-8")
        self._paths.append(env_path)
        old_enable = __import__("os").environ.pop("KS_ENABLE_MODEL", None)
        old_backend = __import__("os").environ.pop("KS_MODEL_BACKEND", None)
        try:
            load_dotenv(env_path)
            config = ModelConfig.from_env()
            parser = build_parser()
            args = parser.parse_args(["--domain", "robotics", "research", "--topic", "servo movement", "ingest", "--budget", "2"])

            self.assertTrue(config.enabled)
            self.assertEqual(config.backend, "ollama")
            self.assertEqual(args.command, "research")
            self.assertEqual(args.research_command, "ingest")
        finally:
            if old_enable is not None:
                __import__("os").environ["KS_ENABLE_MODEL"] = old_enable
            else:
                __import__("os").environ.pop("KS_ENABLE_MODEL", None)
            if old_backend is not None:
                __import__("os").environ["KS_MODEL_BACKEND"] = old_backend
            else:
                __import__("os").environ.pop("KS_MODEL_BACKEND", None)

    def test_infer_domain_reuses_existing_domain(self):
        self.assertEqual(infer_domain("Mario Kart Wii competitive vehicles", ["mario-kart-wii"]), "mario-kart-wii")
        self.assertEqual(infer_domain("robotics servo movement guide", []), "robotics-servo-movement")

    def research_profile_path(self) -> Path:
        profile_path = Path("data/test") / f"{uuid4().hex}.json"
        profile_path.write_text(
            """
            {
              "research_profile": {
                "domain": "robotics",
                "seed_urls": ["https://trusted.test/servo"],
                "allowed_hosts": ["trusted.test"],
                "source_trust": {"trusted.test": 0.9}
              }
            }
            """,
            encoding="utf-8",
        )
        self._paths.append(profile_path)
        return profile_path

    def fake_research_agent(self, store, domain, topic, config_path):
        class FakeSearch:
            def search(self, query, limit=5):
                return [SearchResult("Servo movement guide", "https://trusted.test/servo")]

        class FakeFetcher:
            last_links = []
            last_tables = []

            def fetch(self, source, max_chars=12000):
                return SourceDocument(
                    source_id=source.id,
                    url=source.url,
                    title="Servo movement guide",
                    text_excerpt="Servo movement enables precise robotics control.",
                    full_text="Servo movement enables precise robotics control. Evidence should stay local.",
                    content_hash="servo-hash",
                    domain=source.domain,
                    obsession=source.obsession,
                )

        class FakeRuntime(OptionalModelRuntime):
            def __init__(self):
                super().__init__(ModelConfig(enabled=True, backend="fake"))

            @property
            def available(self):
                return True

            def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
                return """
                {
                  "summary": "Servo movement supports precise control.",
                  "claims": ["Servo movement enables precise robotics control."],
                  "entities": ["Servo"],
                  "relations": ["Servo enables control"],
                  "questions": ["Which servos are fastest?"],
                  "evidence_quotes": ["Servo movement enables precise robotics control."],
                  "confidence": 0.91
                }
                """

        return ResearchAgent(
            store,
            domain,
            topic,
            config_path=config_path,
            search_provider=FakeSearch(),
            fetcher=FakeFetcher(),
            model_runtime=FakeRuntime(),
        )

    def wait_for_job(self, manager: ResearchJobManager, job_id: str) -> dict:
        deadline = time.time() + 5
        snapshot = manager.get(job_id)
        while time.time() < deadline:
            snapshot = manager.get(job_id)
            if snapshot and snapshot["status"] in {"completed", "failed", "stopped"}:
                return snapshot
            time.sleep(0.05)
        self.fail(f"research job {job_id} did not finish; last snapshot: {snapshot}")


if __name__ == "__main__":
    unittest.main()
