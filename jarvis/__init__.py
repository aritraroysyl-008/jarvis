"""
Jarvis - Modular Full-Duplex Voice Assistant with Tool Calling.

A professional, extensible voice assistant package for building AI
voice agents with streaming LLM + TTS, barge-in interruption, and
tool calling.

Usage:
    from jarvis import JARVIS, tools, mcp

    jarvis = JARVIS()
    jarvis.run()

    # tools: JSON schemas for LLM function calling
    # mcp: MCP server endpoint configuration
"""

from jarvis.app import JARVIS
from jarvis.tools.registry import get_tool_schemas, tool, execute_tool, list_tools, TOOLS

# MCP server endpoint configuration
mcp = {
    "tts": "http://127.0.0.1:8080/v1/audio/speech",
    "llm": "http://127.0.0.1:1337/v1/chat/completions",
    "llm_models": "http://127.0.0.1:1337/v1/models",
}

# JSON tool schemas for LLM function calling
tools = get_tool_schemas()

__version__ = "2.0.0"
__author__ = "Jarvis Team"

__all__ = [
    "JARVIS",
    "tools",
    "mcp",
    "tool",
    "execute_tool",
    "list_tools",
    "TOOLS",
]