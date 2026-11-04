"""
example.py

How to extend Jarvis with your own tools and run it as a package.

Run:
    python example.py
"""

from jarvis import JARVIS, tools, mcp
from jarvis.apps import App, register_app, WebApp
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
# 2. Create a custom app by subclassing App
# ---------------------------------------------------------------------------

class GreetingApp(App):
    """Example custom app: responds to greetings with a friendly message."""

    name = "greeting"

    def on_text(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ("hello", "hi", "hey")):
            return "Hello there! How can I help you today?"
        if "bye" in lower or "goodbye" in lower:
            return "Goodbye! Have a great day."
        return f"I heard you say: {text}"


register_app(GreetingApp())


# ---------------------------------------------------------------------------
# 3. Inspect what's registered
# ---------------------------------------------------------------------------

print("MCP endpoints:", mcp)
print("Registered tools:", list_tools())
print("Tool schemas count:", len(tools))
print("Registered apps:", list(register_app.__globals__["_APPS"].keys()))


# ---------------------------------------------------------------------------
# 4. (Optional) Expose Jarvis as a web app
# ---------------------------------------------------------------------------
# Uncomment to run a FastAPI web server on port 8000:
#   pip install fastapi uvicorn
#   python example.py --web
#
# jarvis = JARVIS()
# web = WebApp(jarvis)
# web.run(port=8000)


if __name__ == "__main__":
    import sys

    if "--web" in sys.argv:
        try:
            jarvis = JARVIS()
            web = WebApp(jarvis)
            print("Starting web app at http://127.0.0.1:8000")
            web.run(port=8000)
        except ImportError as exc:
            print(f"Web mode requires fastapi + uvicorn: {exc}")
    else:
        jarvis = JARVIS()
        jarvis.run()
