#!/usr/bin/env python3
"""RAVANA agent service — stdlib HTTP server exposing the cognitive engine.

Runs inside the Docker container (port 4001). The set-and-forget loop (and you)
can drive RAVANA over HTTP:
  GET  /health        -> {"status":"ok","turns":N}
  POST /chat  {"q":"..."} -> {"reply":"...","agent_actions":[...]}
  POST /act   {"tool":"github_cli","arg":"status"} -> guarded tool result

No external web framework (stdlib http.server) to keep the image lean.
The agentic hard guards in ravana/agent/tool_registry.py apply to /act.
"""
import os
import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

os.environ.setdefault("RAVANA_OFFLINE", "1")
from ravana.chat.engine import CognitiveChatEngine
from ravana.agent.tool_registry import ToolRegistry

_engine = None
_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                              user_suffix="")
                _engine._save_path = os.path.join(PROJ, "weights",
                                                  "ravana_weights.pkl")
                try:
                    _engine.load()
                except Exception:
                    pass
                _engine.start_background_learning()
    return _engine


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            eng = get_engine()
            self._json({"status": "ok", "turns": getattr(eng, "turn_count", 0)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {}
        if self.path == "/chat":
            q = str(payload.get("q", ""))
            eng = get_engine()
            reply = eng.process_turn(q)
            actions = getattr(eng, "_agent_action_log", [])[-5:]
            self._json({"reply": reply, "agent_actions": actions})
        elif self.path == "/act":
            from ravana.agent.decision_gate import ToolCall
            reg = ToolRegistry()
            call = ToolCall(tool=str(payload.get("tool", "")),
                            arg=str(payload.get("arg", "")), reason="http")
            self._json({"result": reg.execute(call)})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    port = int(os.environ.get("RAVANA_AGENT_PORT", "4001"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[ravana-agent] listening on :{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
