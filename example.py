"""
example.py

How to extend Jarvis with your own tools and run it as a package.

Run:
    python example.py
"""

from jarvis import JARVIS, tools, mcp
from jarvis.tools.registry import tool, list_tools


# ---------------------------------------------------------------------------
# 1. Define your own tools with the @tool decorator
# ---------------------------------------------------------------------------

@tool("Get the current weather for a city.")
def get_weather(city: str) -> str:
    """Return a fake weather report (replace with a real API call)."""
    return f"The weather in {city} is sunny and 72°F."


@tool("Calculate a tip amount.")
def calculate_tip(bill: float, tip_percent: float = 20.0) -> str:
    """Return the tip and total for a restaurant bill."""
    tip = bill * (tip_percent / 100.0)
    total = bill + tip
    return f"Tip: ${tip:.2f}  Total: ${total:.2f}"


@tool("Search the web for a query.")
def web_search(query: str) -> str:
    """Placeholder web search tool."""
    return f"Search results for '{query}' (placeholder)."


# ---------------------------------------------------------------------------
# 2. Inspect what's registered
# ---------------------------------------------------------------------------

print("MCP endpoints:", mcp)
print("Registered tools:", list_tools())
print("Tool schemas count:", len(tools))


# ---------------------------------------------------------------------------
# 3. Run Jarvis with your custom tools loaded
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jarvis = JARVIS()
    jarvis.run()
