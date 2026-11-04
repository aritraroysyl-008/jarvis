"""
example.py

How to extend Jarvis with custom tools, apps, triggers, and multi-agent mode.

Run:
    python example.py
    python example.py --web
    python example.py --mic-device 1 --output-device 2
    python example.py --no-history
"""

import sys

from jarvis import JARVIS, tools, mcp
from jarvis.apps import App, register_app, WebApp, Agent, AgentRegistry
from jarvis.jext import load_jext_directory, register_jext_tools
from jarvis.tools.registry import tool, list_tools, set_tool_cache_ttl


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


# ---------------------------------------------------------------------------
# 2. Create a custom app by subclassing App
# ---------------------------------------------------------------------------

class GreetingApp(App):
    """Example custom app: responds to greetings using components."""

    name = "greeting"

    def on_text(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ("hello", "hi", "hey")):
            # Use components directly - speak and llm are callable
            self.components.speak("Hello there! How can I help you today?")
            return "Hello there! How can I help you today?"
        if "bye" in lower or "goodbye" in lower:
            return "Goodbye! Have a great day."
        if "weather" in lower:
            result = self.components.tools("get_weather", '{"city": "New York"}')
            return f"Weather: {result}"
        reply = self.components.llm(text)
        return reply


# ---------------------------------------------------------------------------
# 3. Load tools from .jext files (includes triggers)
# ---------------------------------------------------------------------------

jext_schemas = load_jext_directory("tools_example")
print("Loaded .jext tools:", [s["name"] for s in jext_schemas])

# ---------------------------------------------------------------------------
# 4. Configure tool cache (TTL in seconds)
# ---------------------------------------------------------------------------

set_tool_cache_ttl(300)  # Cache tool results for 5 minutes


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse CLI args
    mic_device = None
    output_device = None
    history_file = "conversation_history.json"

    args = sys.argv[1:]
    if "--mic-device" in args:
        idx = args.index("--mic-device")
        mic_device = int(args[idx + 1])
    if "--output-device" in args:
        idx = args.index("--output-device")
        output_device = int(args[idx + 1])
    if "--no-history" in args:
        history_file = None

    # Set device settings globally (imported from settings module)
    from jarvis.config.settings import MIC_DEVICE, OUTPUT_DEVICE
    MIC_DEVICE = mic_device
    OUTPUT_DEVICE = output_device

    jarvis = JARVIS(history_file=history_file)

    # Register .jext tools with trigger manager
    register_jext_tools(jext_schemas, jarvis, trigger_manager=jarvis.trigger_manager)

    # Register custom app
    register_app(GreetingApp(), jarvis)

    # Set up multi-agent registry
    registry = AgentRegistry(jarvis)
    registry.register(Agent("assistant", "You are a helpful voice assistant.", jarvis))
    registry.register(Agent("coder", "You are an expert programmer. Keep code examples short and explain clearly.", jarvis))
    registry.register(Agent("poet", "You are a poetic voice assistant. Respond in short verses.", jarvis))

    print("Multi-agent registry:", registry.list_agents())
    print("Current agent:", registry.current_agent())

    if "--web" in args:
        try:
            web = WebApp(jarvis)
            print("Starting web app at http://127.0.0.1:8000")
            web.run(port=8000)
        except ImportError as exc:
            print(f"Web mode requires fastapi + uvicorn: {exc}")
    else:
        jarvis.run()
