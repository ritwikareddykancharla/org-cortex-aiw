"""HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for OrgCortex."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

from .semantic_fabric import OrgCortexEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"

ENGINE = OrgCortexEngine()

CORS_HEADERS: Dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _normalize_team_acl(raw_acl: Any) -> List[str]:
    if isinstance(raw_acl, list):
        return [str(x).strip() for x in raw_acl if str(x).strip()]
    if isinstance(raw_acl, str):
        parts = [p.strip() for p in raw_acl.split(",") if p.strip()]
        return parts or ["core_platform", "exec_leadership"]
    return ["core_platform", "exec_leadership"]


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure API router shared by local ThreadingHTTPServer and AWS Lambda `/api/*` origin."""
    try:
        if method == "GET" and path == "/api/health":
            return 200, {
                "status": "ok",
                "service": "org-cortex-aiw",
                "bedrock_model": os.environ.get("ORGCORTEX_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "delivery": os.environ.get("ORGCORTEX_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "graph_nodes": len(ENGINE.artifacts),
                "graph_edges": len(ENGINE.edges),
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/state":
            role_team = query.get("role_team") or query.get("role") or "payments_sre"
            snap = ENGINE.get_state_snapshot(role_team=role_team)
            snap["request_id"] = request_id
            return 200, snap

        if method == "POST" and path == "/api/graph/discover":
            artifact_type = str(body.get("artifact_type") or "slack_thread").strip()
            title = str(body.get("title") or "").strip()
            content = str(body.get("content") or "").strip()
            if not title or not content:
                return 400, {"error": "Both `title` and `content` are required"}
            team_acl = _normalize_team_acl(body.get("team_acl"))
            out = ENGINE.ingest_and_discover(
                artifact_type=artifact_type,
                title=title,
                content=content,
                team_acl=team_acl,
                author=str(body.get("author") or "live-studio@orgcortex.io"),
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/meta-agent/reason":
            question = str(body.get("question") or "").strip()
            if not question:
                return 400, {"error": "`question` is required"}
            token_budget = int(body.get("token_budget") or 2048)
            role_team = str(body.get("role_team") or "payments_sre").strip()
            out = ENGINE.reason(
                question=question,
                token_budget=token_budget,
                role_team=role_team,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            ENGINE.reset()
            return 200, {
                "status": "reset_complete",
                "graph_nodes": len(ENGINE.artifacts),
                "graph_edges": len(ENGINE.edges),
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` behavior."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {
        k: str(v)
        for k, v in (event.get("queryStringParameters") or {}).items()
        if v is not None
    }
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    # Diagnostic static fallback if origin URL is inspected directly
    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalOrgCortexHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3097) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalOrgCortexHandler)
    print(f"OrgCortex studio running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3097"))
    run_local_server(port=port_num)
