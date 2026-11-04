"""
jarvis.jext - Load tools from .jext (Jarvis Extension) files.

A .jext file describes a tool in plain text. Jarvis parses it into an
OpenAI-compatible function schema and registers it so the LLM can call it.

File format (example):

    Tool: get_weather

    Description:
    Gets current weather information for a city.

    Parameters:
    city: Name of the city.
    unit: Temperature unit, Celsius or Fahrenheit.

Produces:

    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets current weather information for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string", "description": "Name of the city." },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit."
            }
          },
          "required": ["city"],
          "additionalProperties": false,
          "strict": true
        }
      }
    }

Usage:

    from jarvis.jext import load_jext_directory, register_jext_tools

    # Load all .jext files from a folder
    tools = load_jext_directory("my_tools")

    # Register them with Jarvis (they need a Python callable to execute)
    register_jext_tools(tools, jarvis)

Tools loaded from .jext files without a matching Python callable execute a
default handler that returns a "tool not implemented" message. To give a tool
real behavior, define a Python function with the same name and register it
via @tool or jarvis.tools before loading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


def _infer_type(param_name: str, desc: str) -> str:
    """Guess JSON schema type from parameter name/description."""
    d = (param_name + " " + desc).lower()
    if any(k in d for k in ("count", "number of", "how many", "age", "size", "limit")):
        return "integer"
    if any(k in d for k in ("price", "temp", "temperature", "amount", "rating", "score")):
        return "number"
    if any(k in d for k in ("is ", "enabled", "active", "on ", "valid", "correct")):
        return "boolean"
    return "string"


def _enum_from_desc(desc: str) -> Optional[List[str]]:
    """Extract an enum list from a description like 'Celsius or Fahrenheit'."""
    m = re.findall(r"\b([A-Za-z][A-Za-z]+)\b", desc)
    # Keep short, capitalizable words that look like enum values
    candidates = [w.lower() for w in m if len(w) > 2]
    common = {
        "celsius", "fahrenheit", "kelvin",
        "heat", "cool", "auto",
        "on", "off",
        "true", "false",
        "yes", "no",
        "read", "write", "delete", "create",
        "meters", "feet", "miles", "kilometers",
        "json", "text", "csv", "xml",
    }
    found = [c for c in candidates if c in common]
    # De-duplicate preserving order
    seen = set()
    out = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out if len(out) >= 2 else None


def parse_jext(text: str) -> Dict[str, Any]:
    """Parse a .jext file's text content into a tool schema dict.

    Returns:
        Dict with keys: name, description, parameters (OpenAI function schema).
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty .jext content")

    # --- Tool name ---
    name_match = re.search(r"^\s*Tool:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if not name_match:
        raise ValueError(".jext file is missing 'Tool: <name>' line")
    name = name_match.group(1).strip()

    # --- Description ---
    desc_match = re.search(
        r"^\s*Description:\s*\n(.*?)(?=\n\s*\n|\n\s*Parameters:|\Z)",
        text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    description = ""
    if desc_match:
        description = desc_match.group(1).strip()

    # --- Parameters ---
    params_section = ""
    params_match = re.search(
        r"^\s*Parameters:\s*\n(.*?)\Z",
        text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if params_match:
        params_section = params_match.group(1).strip()

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for line in params_section.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Split on first colon only
        pname, pdesc = line.split(":", 1)
        pname = pname.strip()
        pdesc = pdesc.strip()
        if not pname:
            continue

        ptype = _infer_type(pname, pdesc)
        prop: Dict[str, Any] = {"type": ptype, "description": pdesc}

        enum_values = _enum_from_desc(pdesc)
        if enum_values:
            prop["enum"] = enum_values

        # Heuristic: names containing 'name', 'city', 'text', 'query' tend to be required
        if any(k in pname.lower() for k in ("name", "city", "text", "query", "id", "message")):
            required.append(pname)

        properties[pname] = prop

    schema: Dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "strict": True,
        },
    }
    if required:
        schema["parameters"]["required"] = sorted(set(required))

    return schema


def parse_jext_file(path: str | Path) -> Dict[str, Any]:
    """Parse a single .jext file into a tool schema dict."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_jext(f.read())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jext_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load one .jext file, returning a list with one tool schema."""
    return [parse_jext_file(path)]


def load_jext_directory(dir_path: str | Path) -> List[Dict[str, Any]]:
    """Load all .jext files from a directory, returning a list of tool schemas."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    schemas: List[Dict[str, Any]] = []
    for p in sorted(dir_path.glob("*.jext")):
        try:
            schemas.append(parse_jext_file(p))
        except Exception as exc:  # pragma: no cover - report and continue
            print(f"[jext] Skipping {p}: {exc}")
    return schemas


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _default_handler(name: str) -> Callable[..., str]:
    """Return a placeholder handler for .jext tools with no Python callable."""

    def handler(**kwargs) -> str:
        return json.dumps(
            {
                "error": f"Tool '{name}' is declared but not implemented.",
                "hint": "Define a Python function named '{name}' and register it with @tool or jarvis.tools.",
                "received_args": kwargs,
            }
        )

    return handler


def register_jext_tools(
    schemas: List[Dict[str, Any]],
    jarvis_or_registry: Any,
) -> List[str]:
    """Register .jext tool schemas with a JARVIS instance or tool registry.

    Args:
        schemas: List of tool schema dicts (as produced by parse_jext / load_jext_*).
        jarvis_or_registry: A JARVIS instance (uses jarvis.tools.registry) or
            anything exposing .registry with register/execute_tool, or the
            registry module itself.

    Returns:
        List of registered tool names.
    """
    from jarvis.tools.registry import tool as tool_decorator, TOOLS, execute_tool

    registry = None
    if hasattr(jarvis_or_registry, "tools"):
        # JARVIS instance - use its tools registry module
        registry = TOOLS
    elif hasattr(jarvis_or_registry, "registry"):
        registry = TOOLS
    else:
        registry = TOOLS

    registered: List[str] = []
    for schema in schemas:
        name = schema["name"]
        description = schema.get("description", "")
        parameters = schema.get("parameters", {})

        if name in TOOLS:
            # Already has a Python callable; just ensure schema matches
            registered.append(name)
            continue

        # Build a callable that uses the default handler
        def _jext_handler(**kwargs):
            return _default_handler(name)(**kwargs)

        _jext_handler.__name__ = name
        _jext_handler.__qualname__ = name
        tool_decorator(
            description=description,
            properties=parameters.get("properties", {}),
            required=parameters.get("required", []),
        )(_jext_handler)

        # Store the raw schema so the LLM sees the .jext definition
        TOOLS[name]["schema"] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        registered.append(name)

    return registered


def register_jext_directory(
    dir_path: str | Path,
    jarvis_or_registry: Any,
) -> List[str]:
    """Load all .jext files from a directory and register them."""
    schemas = load_jext_directory(dir_path)
    return register_jext_tools(schemas, jarvis_or_registry)


__all__ = [
    "parse_jext",
    "parse_jext_file",
    "load_jext_file",
    "load_jext_directory",
    "register_jext_tools",
    "register_jext_directory",
]
