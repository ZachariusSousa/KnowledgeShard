"""Zero-dependency HTTP server for the MVP API."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .savant import Savant
from .storage import KnowledgeStore


class SavantRequestHandler(BaseHTTPRequestHandler):
    store = KnowledgeStore()
    savant = Savant(store=store)

    def do_GET(self) -> None:
        if self.path == "/status":
            self.write_json(self.savant.status())
            return
        if self.path == "/metrics":
            self.write_json(self.savant.metrics())
            return
        self.write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        payload = self.read_json()
        if self.path == "/query":
            response = self.savant.query(
                question=str(payload.get("question", "")),
                num_experts=int(payload.get("num_experts", 3)),
                timeout_seconds=int(payload.get("timeout_seconds", 30)),
            )
            self.write_json(asdict(response))
            return
        if self.path == "/correct":
            correction = self.savant.correct(
                query_id=str(payload.get("query_id", "")),
                correction=str(payload.get("correction", "")),
                confidence=float(payload.get("confidence", 1.0)),
            )
            self.write_json({"received": True, "correction": asdict(correction)})
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

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KnowledgeShard MVP HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), SavantRequestHandler)
    print(f"KnowledgeShard API listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
