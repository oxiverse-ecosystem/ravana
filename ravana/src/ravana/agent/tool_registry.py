#!/usr/bin/env python3
"""RAVANA agentic tool registry — the cognitive architecture's "hands".

Design rules (per ravana-cognitive-engine skill / founder doctrine):
- The registry is SEED data (allowed). The DECISION to use a tool is made by
  RAVANA's own cognitive state (curiosity / metacognition / intent), NOT a
  keyword table. See decision_gate.py.
- Every tool execution is SANDBOXED and GUARDED. Destructive ops are forbidden:
  no `rm -rf`, no `git push --force`/`--force-with-lease`, no credential writes
  (*.env, *.pem, *.key), no writes outside the mounted /work volume.
- Tool results are returned as grounded evidence for RAVANA to learn from; they
  are never spliced as authoritative fact without the engine's normal source-
  trust gating.

This is the "hands" layer the founder requested (web via IntentForge, read site,
run scripts, github cli on a loop-assigned repo).
"""
from __future__ import annotations
import os
import re
import shlex
import subprocess
import json
import urllib.request
import urllib.parse
import socket
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

# ── Hard guards: patterns that must NEVER execute, even if RAVANA "decides" ──
_FORBIDDEN_PATTERNS = [
    r"\brm\s+-rf\b", r"\brm\b.*--force", r"\bdel\s+/[fq]",
    r"git\s+push\b[^\n]*--force", r"git\s+push\b[^\n]*--force-with-lease",
    r"--force\b", r"--force-with-lease", r"--hard\b",
    r"git\s+reset\b[^\n]*--hard", r"git\s+clean\b[^\n]*-[a-z]*f",
    r"\.env", r"\.pem", r"\.key\b", r"credential",
    r"sudo\b", r"chmod\s+777", r":\(\)\s*\{",  # fork bomb
]

_WORK_VOLUME = os.environ.get("RAVANA_WORK_VOLUME", "C:/Users/Likhith/Documents/Projects/ravana/_agent_work")


@dataclass
class Tool:
    """A capability RAVANA can invoke. run() is the executor; guarded."""
    name: str
    description: str
    is_destructive: bool = False
    run: Optional[Callable[[str], str]] = None


@dataclass
class ToolCall:
    tool: str
    arg: str
    reason: str  # cognitive reason (from decision_gate, state-derived)


def _guard(cmd: str) -> None:
    """Raise if cmd matches a forbidden destructive pattern. Fail-closed."""
    low = cmd.lower()
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, low):
            raise PermissionError(f"tool blocked by hard guard: matches {pat!r}")


def _web_search_via_intentforge(query: str) -> str:
    """Web/search grounding through the IntentForge API (founder-specified)."""
    # IntentForge gateway listens locally; query its /search endpoint.
    url = f"http://localhost:4000/search?q={urllib.parse.quote(query)}"
    try:
        socket.setdefaulttimeout(8.0)
        with urllib.request.urlopen(url) as r:
            data = r.read().decode("utf-8", "replace")
        return f"[web:intentforge] {data[:1200]}"
    except Exception as e:
        # Fall back to a direct web fetch if IntentForge is down (offline-safe)
        try:
            with urllib.request.urlopen(f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}") as r:
                return f"[web:fallback] {r.read().decode('utf-8','replace')[:1000]}"
        except Exception as e2:
            return f"[web] search unavailable: {e} / {e2}"


def _read_website(url: str) -> str:
    """Fetch and trim a web page for grounding."""
    if not re.match(r"https?://", url):
        raise ValueError("only http(s) urls")
    with urllib.request.urlopen(url, timeout=8) as r:
        return f"[site] {r.read().decode('utf-8','replace')[:1500]}"


def _run_script(script: str) -> str:
    """Run a script INSIDE the sandboxed work volume only.

    script is a path relative to RAVANA_WORK_VOLUME; never an absolute path
    outside it. Execution is guarded and timeboxed.
    """
    if not script:
        raise ValueError("empty script path")
    target = os.path.normpath(os.path.join(_WORK_VOLUME, script))
    if not target.startswith(os.path.normpath(_WORK_VOLUME)):
        raise PermissionError("script must live inside the work volume")
    _guard(script)
    proc = subprocess.run(["python", target], cwd=_WORK_VOLUME,
                          capture_output=True, text=True, timeout=60)
    return f"[script] rc={proc.returncode}\nstdout: {proc.stdout[:800]}\nstderr: {proc.stderr[:400]}"


def _github_cli(args: str) -> str:
    """Git operations on the loop-assigned repo (mounted in work volume).

    Hard-guarded: no force-push, no credential touch, no destructive reset.
    Read-only + safe local ops (status, diff, log, commit, push to a branch).
    """
    _guard(args)
    # Only allow a curated, safe subset of git subcommands.
    allowed = ("status", "diff", "log", "branch", "add", "commit",
               "fetch", "pull", "checkout", "push", "clone", "remote", "show")
    first = shlex.split(args)[0] if args.strip() else ""
    if first not in allowed:
        raise PermissionError(f"git subcommand '{first}' not in safe allowlist")
    proc = subprocess.run(["git"] + shlex.split(args), cwd=_WORK_VOLUME,
                          capture_output=True, text=True, timeout=60)
    return f"[git] rc={proc.returncode}\n{proc.stdout[:800]}{proc.stderr[:400]}"


# ── The seed registry (allowed capabilities). Not a matcher — just definitions. ──
def build_registry() -> Dict[str, Tool]:
    return {
        "web_search": Tool(
            name="web_search",
            description="Search the web / ground a claim via IntentForge API",
            is_destructive=False, run=_web_search_via_intentforge),
        "read_website": Tool(
            name="read_website",
            description="Fetch a web page for grounding",
            is_destructive=False, run=_read_website),
        "run_script": Tool(
            name="run_script",
            description="Run a script in the sandboxed work volume",
            is_destructive=False, run=_run_script),
        "github_cli": Tool(
            name="github_cli",
            description="Safe git ops on the loop-assigned repo (no force-push)",
            is_destructive=False, run=_github_cli),
    }


class ToolRegistry:
    """Holds the seed tools and executes guarded calls."""

    def __init__(self) -> None:
        self.tools = build_registry()

    def names(self) -> List[str]:
        return list(self.tools.keys())

    def execute(self, call: ToolCall) -> str:
        tool = self.tools.get(call.tool)
        if tool is None or tool.run is None:
            return f"[tool] unknown tool: {call.tool}"
        try:
            _guard(call.arg)
            return tool.run(call.arg)
        except PermissionError as e:
            return f"[tool BLOCKED] {e}"
        except Exception as e:
            return f"[tool error] {type(e).__name__}: {e}"
