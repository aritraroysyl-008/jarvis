"""
example.py

How to extend Jarvis with your own tools and run it as a package.

Run:
    python example.py
"""

from jarvis import JARVIS, tools, mcp
from jarvis.apps import App, Components, register_app, WebApp
from jarvis.jext import load_jext_directory, register_jext_tools
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


@tool(
    "Write and execute Python code to perform calculations or other tasks.",
    {"code": {"type": "string", "description": "Python source code to execute."}},
    ["code"],
)
def run_python(code: str) -> str:
    """Write Python code to a temp file and execute it, capturing stdout.

    WARNING: This runs arbitrary Python code on the local machine.
    """
    import subprocess, sys, tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(code)
        tmp.flush(); tmp.close()
        result = subprocess.run([sys.executable, tmp.name], capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Python script timed out after 30 seconds."
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 2. Create a custom app by subclassing App
# ---------------------------------------------------------------------------

class GreetingApp(App):
    """Example custom app: responds to greetings with a friendly message.

    Demonstrates using components (speak, llm, tools) inside on_text.
    """

    name = "greeting"

    def on_text(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ("hello", "hi", "hey")):
            # Use components directly - speak and llm are callable
            self.components.speak("Hello there! How can I help you today?")
            return "Hello there! How can I help you today?"
        if "bye" in lower or "goodbye" in lower:
            return "Goodbye! Have a great day."

        # Example: call a tool through components
        if "weather" in lower:
            result = self.components.tools("get_weather", '{"city": "New York"}')
            return f"Weather: {result}"

        # Example: ask the LLM through components
        reply = self.components.llm(text)
        return reply


# register_app now needs the jarvis instance (called in __main__ below)


# ---------------------------------------------------------------------------
# 3. Load tools from .jext files (no Python code needed for the schema)
# ---------------------------------------------------------------------------
# These define the tool schema for the LLM. Tools without a matching Python
# callable get a default "not implemented" handler. Tools that DO have a
# Python implementation (like get_weather above) are left untouched.

jext_schemas = load_jext_directory("tools_example")
print("Loaded .jext tools:", [s["name"] for s in jext_schemas])


# ---------------------------------------------------------------------------
# 4. Inspect what's registered
# ---------------------------------------------------------------------------

print("MCP endpoints:", mcp)
print("Registered tools:", list_tools())
print("Tool schemas count:", len(tools))
print("Components: speak, transcribe, llm, tools, play, history, clear_history")


# ---------------------------------------------------------------------------
# 5. (Optional) Expose Jarvis as a web app
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

    jarvis = JARVIS()

    # Register .jext tools with the jarvis instance (wires up components)
    register_jext_tools(jext_schemas, jarvis)

    # Register the custom app with the jarvis instance so components are wired up
    register_app(GreetingApp(), jarvis)

    if "--web" in sys.argv:
        try:
            web = WebApp(jarvis)
            print("Starting web app at http://127.0.0.1:8000")
            web.run(port=8000)
        except ImportError as exc:
            print(f"Web mode requires fastapi + uvicorn: {exc}")
    else:
        jarvis.run()
