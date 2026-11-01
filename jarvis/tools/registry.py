"""
Jarvis Voice Assistant - Tool Registry

Provides a decorator-based system for registering LLM-callable tools
with OpenAI function-calling schema.
"""

import json
import inspect
from typing import Callable, Any, Dict, Optional, List
from functools import wraps


# Global tool registry
TOOLS: Dict[str, Dict] = {}


def tool(
    description: str,
    properties: Optional[Dict] = None,
    required: Optional[List[str]] = None,
):
    """
    Register a function as an LLM-callable tool (OpenAI function-calling schema).

    Args:
        description: Human-readable description of what the tool does.
        properties: JSON schema properties for the tool's parameters.
        required: List of required parameter names.

    Usage:
        @tool("Get the current time.")
        def get_current_time() -> str:
            return datetime.now().strftime("%I:%M %p")

        @tool(
            "Calculate an expression.",
            {"expression": {"type": "string", "description": "Math expression"}},
            ["expression"]
        )
        def calculate(expression: str) -> str:
            return str(eval(expression))
    """
    def register(fn: Callable) -> Callable:
        # Auto-generate schema from function signature if not provided
        if properties is None:
            sig = inspect.signature(fn)
            auto_properties = {}
            auto_required = []
            for name, param in sig.parameters.items():
                if param.annotation == inspect.Parameter.empty:
                    auto_properties[name] = {"type": "string"}
                elif param.annotation == int:
                    auto_properties[name] = {"type": "integer"}
                elif param.annotation == float:
                    auto_properties[name] = {"type": "number"}
                elif param.annotation == bool:
                    auto_properties[name] = {"type": "boolean"}
                else:
                    auto_properties[name] = {"type": "string"}

                if param.default == inspect.Parameter.empty:
                    auto_required.append(name)

            schema_props = auto_properties
            schema_required = auto_required
        else:
            schema_props = properties
            schema_required = required or []

        TOOLS[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": schema_props,
                        "required": schema_required,
                    },
                },
            },
        }

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return register


def get_tool_schemas() -> List[Dict]:
    """Return all registered tool schemas for LLM function calling."""
    return [t["schema"] for t in TOOLS.values()]


def execute_tool(name: str, raw_args: str) -> str:
    """
    Execute a registered tool by name with JSON arguments.

    Args:
        name: Tool function name.
        raw_args: JSON string of arguments.

    Returns:
        Tool result as JSON string.
    """
    entry = TOOLS.get(name)
    if entry is None:
        return json.dumps({"error": f"unknown tool '{name}'"})

    try:
        args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
    except json.JSONDecodeError:
        return json.dumps({"error": "arguments were not valid JSON"})

    try:
        result = entry["fn"](**args)
    except Exception as e:
        return json.dumps({"error": str(e)})

    return result if isinstance(result, str) else json.dumps(result)


def list_tools() -> List[str]:
    """Return list of registered tool names."""
    return list(TOOLS.keys())


def clear_tools():
    """Clear all registered tools (useful for testing)."""
    TOOLS.clear()