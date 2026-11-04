"""
Jarvis Voice Assistant - Built-in Tools

Core tools that ship with Jarvis. Add your own tools by creating
new modules and importing them in tools/__init__.py
"""

import ast
import operator
import subprocess
import sys
import tempfile
import os
from datetime import datetime

from jarvis.tools.registry import tool


# UTILITY: Safe arithmetic evaluation

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


# BUILT-IN TOOLS

@tool("Get the current local date and time.")
def get_current_time() -> str:
    """Return current date and time in a spoken-friendly format."""
    return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")


@tool(
    "Evaluate an arithmetic expression, e.g. '12 * (3 + 4) / 5'.",
    {"expression": {"type": "string", "description": "The arithmetic expression."}},
    ["expression"],
)
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    return str(_safe_eval(ast.parse(expression, mode="eval")))


# EXAMPLE: How to add custom tools
# @tool(
#     "Search the web for information.",
#     {"query": {"type": "string", "description": "Search query"}},
#     ["query"],
# )
# def web_search(query: str) -> str:
#     # Your implementation here
#     return f"Search results for: {query}"
#
# @tool(
#     "Control smart home devices.",
#     {
#         "device": {"type": "string", "description": "Device name"},
#         "action": {"type": "string", "description": "Action to perform"},
#     },
#     ["device", "action"],
# )
# def smart_home(device: str, action: str) -> str:
#     # Your implementation here
#     return f"Turned {action} the {device}"


# In-memory reminder store (simple list; persists for the lifetime of the process)
_REMINDERS: list[dict] = []


@tool(
    "Set a reminder for a future time.",
    {
        "message": {"type": "string", "description": "What to remind about."},
        "when": {"type": "string", "description": "When to remind (e.g. 'in 10 minutes', 'today at 5pm')."},
    },
    ["message", "when"],
)
def set_reminder(message: str, when: str) -> str:
    """Store a reminder and return a confirmation."""
    _REMINDERS.append({"message": message, "when": when})
    return f"Reminder set: '{message}' at {when} ({len(_REMINDERS)} total reminders)."


@tool(
    "List all active reminders.",
    {},
    [],
)
def list_reminders() -> str:
    """Return all stored reminders."""
    if not _REMINDERS:
        return "No reminders set."
    lines = [f"{i+1}. {r['message']} at {r['when']}" for i, r in enumerate(_REMINDERS)]
    return "Reminders:\n" + "\n".join(lines)


@tool(
    "Write and execute Python code to perform calculations or other tasks.",
    {
        "code": {"type": "string", "description": "Python source code to execute."},
    },
    ["code"],
)
def run_python(code: str) -> str:
    """Write Python code to a temp file and execute it, capturing stdout.

    WARNING: This runs arbitrary Python code on the local machine. Only use
    with trusted input.

    Args:
        code: Python source code to run.

    Returns:
        stdout output from the script, or an error message if execution failed.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(code)
        tmp.flush()
        tmp.close()

        result = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Python script timed out after 30 seconds."
    except Exception as exc:  # pragma: no cover - runtime
        return f"Error: {exc}"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@tool(
    "Search the web using AI-powered crawl (Crawl4AI).",
    {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {"type": "integer", "description": "Maximum number of results to return (default 5)."},
    },
    ["query"],
)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Crawl4AI (AI-powered browser automation).

    Uses crawl4ai to fetch and extract search results. Requires:
        pip install crawl4ai
        crawl4ai-setup  # install browser binaries

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        Formatted string with result titles and URLs.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        import asyncio
    except ImportError:
        return (
            "Error: crawl4ai not installed. Run: pip install crawl4ai\n"
            "Then run: crawl4ai-setup (to install browser binaries)"
        )

    search_url = "https://html.duckduckgo.com/html/?q=" + __import__("urllib.parse").quote(query)

    async def _fetch():
        async with AsyncWebCrawler(verbose=False) as crawler:
            return await crawler.arun(
                url=search_url,
                word_count_threshold=10,
            )

    try:
        result = asyncio.run(_fetch())
    except Exception as exc:  # pragma: no cover - network
        return f"Error: web search failed: {exc}"

    markdown = getattr(result, "markdown", "") or ""
    if not markdown.strip():
        return "No results found."

    import re
    from urllib.parse import urlparse, parse_qs

    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown)

    results = []
    for title, url in links[:max_results]:
        title = title.strip()
        url = url.strip()
        if url.startswith("//duckduckgo.com/l/?"):
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                url = qs["uddg"][0]
        results.append(f"{len(results)+1}. {title}\\n   {url}")

    if not results:
        return "No results found."

    return "\\n\\n".join(results)