"""Zero-dependency HTTP server for the MVP API."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .research import ResearchJobManager
from .savant import Savant
from .storage import KnowledgeStore


class SavantRequestHandler(BaseHTTPRequestHandler):
    store = KnowledgeStore()
    savant = Savant(store=store)
    research_jobs = ResearchJobManager(store)
    default_config = "config/mario_kart_wii.sources.json"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.write_html(RESEARCH_CONSOLE_HTML)
            return
        if parsed.path == "/status":
            self.write_json(self.savant.status())
            return
        if parsed.path == "/model-status":
            self.write_json(self.savant.model_status())
            return
        if parsed.path == "/metrics":
            self.write_json(self.savant.metrics())
            return
        if parsed.path == "/pending":
            self.write_json({"pending": [pending.__dict__ for pending in self.store.list_pending_facts(self.savant.domain)]})
            return
        if parsed.path == "/research/jobs":
            self.write_json({"jobs": [self.enrich_job(job) for job in self.research_jobs.list()]})
            return
        if parsed.path == "/research/status":
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0]
            job = self.research_jobs.get(job_id)
            if job is None:
                self.write_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self.write_json({"job": self.enrich_job(job)})
            return
        self.write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self.read_json()
        if parsed.path == "/query":
            response = self.savant.query(
                question=str(payload.get("question", "")),
                num_experts=int(payload.get("num_experts", 3)),
                timeout_seconds=int(payload.get("timeout_seconds", 30)),
            )
            self.write_json(asdict(response))
            return
        if parsed.path == "/correct":
            correction = self.savant.correct(
                query_id=str(payload.get("query_id", "")),
                correction=str(payload.get("correction", "")),
                confidence=float(payload.get("confidence", 1.0)),
            )
            self.write_json({"received": True, "correction": asdict(correction)})
            return
        if parsed.path == "/research/start":
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                self.write_json({"error": "topic is required"}, HTTPStatus.BAD_REQUEST)
                return
            domain = str(payload.get("domain") or self.resolve_domain(topic))
            job = self.research_jobs.start(
                topic=topic,
                domain=domain,
                config_path=str(payload.get("config") or self.default_config),
                max_cycles=int(payload.get("max_cycles", 1)),
            )
            self.write_json({"job_id": job.id, "job": self.enrich_job(job.snapshot())}, HTTPStatus.CREATED)
            return
        if parsed.path == "/research/stop":
            job_id = str(payload.get("job_id", ""))
            if not self.research_jobs.stop(job_id):
                self.write_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            job = self.research_jobs.get(job_id)
            self.write_json({"stopped": True, "job": self.enrich_job(job) if job else None})
            return
        self.write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def resolve_domain(self, topic: str) -> str:
        domains = self.store.list_domains()
        if domains:
            topic_terms = {term for term in topic.lower().replace("-", " ").split() if term}
            for domain in domains:
                domain_terms = {term for term in domain.lower().replace("-", " ").split() if term}
                if domain_terms and len(topic_terms & domain_terms) / len(domain_terms) >= 0.6:
                    return domain
            return domains[0]
        return self.savant.domain

    def enrich_job(self, job: dict) -> dict:
        if not job:
            return {}
        domain = str(job.get("domain", self.savant.domain))
        topic = str(job.get("topic", ""))
        pending = self.store.list_pending_facts(domain)
        approved = self.store.list_learned_facts(domain)
        notes = self.store.list_research_notes(domain, topic, 5) if topic else []
        output = dict(job)
        output.update(
            {
                "pending_count": self.store.count_pending_facts(domain),
                "approved_count": self.store.count_facts(domain),
                "model_status": self.savant.model_status(),
                "notes": [asdict(note) for note in notes],
                "pending": [pending_fact.__dict__ for pending_fact in pending[:10]],
                "approved": [asdict(fact) for fact in approved[:10]],
            }
        )
        return output

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KnowledgeShard MVP HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", default="data/knowledgeshard.db")
    parser.add_argument("--domain", default="mario-kart-wii")
    parser.add_argument("--config", default="config/mario_kart_wii.sources.json")
    args = parser.parse_args(argv)

    store = KnowledgeStore(args.db)
    SavantRequestHandler.store = store
    SavantRequestHandler.savant = Savant(domain=args.domain, store=store)
    SavantRequestHandler.research_jobs = ResearchJobManager(store)
    SavantRequestHandler.default_config = args.config

    server = ThreadingHTTPServer((args.host, args.port), SavantRequestHandler)
    print(f"KnowledgeShard API listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


RESEARCH_CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KnowledgeShard Research</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    body { margin: 0; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 28px; margin: 0 0 20px; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    .toolbar { display: grid; grid-template-columns: 1fr 220px auto auto; gap: 10px; align-items: end; margin-bottom: 18px; }
    label { display: grid; gap: 6px; font-size: 13px; color: #52606d; }
    input { padding: 10px 12px; border: 1px solid #cbd2d9; border-radius: 6px; font-size: 15px; background: white; }
    button { border: 0; border-radius: 6px; padding: 11px 14px; font-size: 14px; cursor: pointer; background: #1f6feb; color: white; }
    button.secondary { background: #6b7280; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
    .panel, .metric { background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; }
    .metric strong { display: block; font-size: 24px; margin-top: 4px; }
    .content { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .item { border-top: 1px solid #eef2f7; padding: 10px 0; }
    .item:first-child { border-top: 0; padding-top: 0; }
    .muted { color: #66788a; font-size: 13px; }
    .pill { display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e6f0ff; color: #174ea6; font-size: 12px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f8fafc; padding: 10px; border-radius: 6px; }
    @media (max-width: 800px) { .toolbar, .content, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>KnowledgeShard Research</h1>
    <section class="toolbar">
      <label>Research focus <input id="topic" placeholder="Mario Kart Wii drift mechanics"></label>
      <label>Domain <input id="domain" placeholder="auto"></label>
      <button id="start">Start</button>
      <button id="stop" class="secondary" disabled>Stop</button>
    </section>
    <section class="grid">
      <div class="metric"><span class="muted">Status</span><strong id="status">idle</strong></div>
      <div class="metric"><span class="muted">Phase</span><strong id="phase">none</strong></div>
      <div class="metric"><span class="muted">Pending</span><strong id="pendingCount">0</strong></div>
      <div class="metric"><span class="muted">Approved</span><strong id="approvedCount">0</strong></div>
    </section>
    <section class="content">
      <div class="panel"><h2>Latest Notes</h2><div id="notes" class="muted">No notes yet.</div></div>
      <div class="panel"><h2>Auto-Approved Facts</h2><div id="approved" class="muted">No auto-approved facts yet.</div></div>
      <div class="panel"><h2>Pending Review</h2><div id="pending" class="muted">No pending facts.</div></div>
      <div class="panel"><h2>Runtime</h2><pre id="runtime">{}</pre></div>
    </section>
  </main>
  <script>
    let activeJob = "";
    let poller = null;
    const $ = id => document.getElementById(id);
    async function post(url, body) {
      const response = await fetch(url, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      return payload;
    }
    async function refresh() {
      if (!activeJob) return;
      const response = await fetch(`/research/status?job_id=${encodeURIComponent(activeJob)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      render(payload.job);
      if (["completed", "failed", "stopped"].includes(payload.job.status)) {
        clearInterval(poller);
        $("start").disabled = false;
        $("stop").disabled = true;
      }
    }
    function render(job) {
      $("status").textContent = job.status;
      $("phase").textContent = job.phase;
      $("pendingCount").textContent = job.pending_count;
      $("approvedCount").textContent = job.approved_count;
      $("runtime").textContent = JSON.stringify({job: job.counters, errors: job.errors, model: job.model_status}, null, 2);
      $("notes").innerHTML = list(job.notes, note => `<div class="item"><span class="pill">${note.confidence}</span> ${escapeHtml(note.summary)}</div>`);
      $("approved").innerHTML = list(job.approved, fact => `<div class="item">${escapeHtml(fact.object)}<div class="muted">${escapeHtml(fact.source)}</div></div>`);
      $("pending").innerHTML = list(job.pending, fact => `<div class="item">${escapeHtml(fact.object)}<div class="muted">${escapeHtml(fact.source)}</div></div>`);
    }
    function list(items, fn) {
      return items && items.length ? items.map(fn).join("") : "<span class='muted'>None.</span>";
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
    }
    $("start").addEventListener("click", async () => {
      const topic = $("topic").value.trim();
      if (!topic) return;
      $("start").disabled = true;
      const payload = {topic};
      if ($("domain").value.trim()) payload.domain = $("domain").value.trim();
      const result = await post("/research/start", payload);
      activeJob = result.job_id;
      $("stop").disabled = false;
      render(result.job);
      poller = setInterval(refresh, 1500);
      refresh();
    });
    $("stop").addEventListener("click", async () => {
      if (!activeJob) return;
      await post("/research/stop", {job_id: activeJob});
      refresh();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
