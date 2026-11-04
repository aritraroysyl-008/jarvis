"""
jarvis.apps - Build your own apps and webapps on top of Jarvis.

Subclass :class:`App` to create a custom voice app, then register it
with :func:`register_app`. Use :class:`WebApp` to expose Jarvis over
HTTP (FastAPI) for frontend frontends, dashboards, or API integrations.

Usage:
    from jarvis.apps import App, register_app, WebApp

    class MyApp(App):
        name = "my_app"
        def on_text(self, text: str) -> str:
            return f"You said: {text}"

    register_app(MyApp())

    # Web app (requires FastAPI + uvicorn)
    web = WebApp(jarvis_instance)
    web.run(port=8000)
"""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from jarvis.app import JARVIS  # noqa: F401

# ---------------------------------------------------------------------------
# Components - callable building blocks for custom apps
# ---------------------------------------------------------------------------

class Components:
    """Callable Jarvis components for use in custom apps.

    Each attribute is a function you can call directly:

        app.components.speak("Hello")
        text = app.components.transcribe(audio_array)
        reply = app.components.llm("What is the weather?")
        result = app.components.tools("get_weather", '{"city": "NYC"}')
        app.components.play(audio_bytes)

    Attributes:
        jarvis: Parent JARVIS instance.
    """

    def __init__(self, jarvis: "JARVIS"):
        self.jarvis = jarvis

    # ------------------------------------------------------------------
    # TTS - text to speech
    # ------------------------------------------------------------------

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        """Speak text through the TTS pipeline.

        Args:
            text: Text to speak.
            voice: Optional voice name override.
        """
        jarvis = self.jarvis
        if jarvis.player is None:
            raise RuntimeError("Jarvis player not initialized. Call jarvis.run() first.")

        import queue
        import threading
        from jarvis.pipeline.tts import speak as _speak
        from jarvis.core.turn import Turn

        turn = Turn()
        tts_q: "queue.Queue[str]" = queue.Queue()
        tts_pending = threading.Event()

        tts_thread = threading.Thread(
            target=self._tts_worker,
            args=(turn, tts_q, jarvis.player, tts_pending),
            daemon=True,
        )
        tts_thread.start()

        tts_q.put(text)
        tts_pending.set()
        tts_thread.join(timeout=60)

        # Record the spoken text in conversation history so the assistant
        # doesn't "forget" what it just said.
        jarvis.session.history.append({"role": "assistant", "content": text})

    @staticmethod
    def _tts_worker(turn, tts_q, player, tts_pending):
        from jarvis.pipeline.tts import tts_worker
        tts_worker(turn, tts_q, player, tts_pending)

    # ------------------------------------------------------------------
    # STT - speech to text
    # ------------------------------------------------------------------

    def transcribe(self, audio) -> str:
        """Transcribe audio to text.

        Args:
            audio: numpy int16 array of audio samples.

        Returns:
            Transcribed text string.
        """
        from jarvis.audio.transcribe import transcribe as _transcribe
        return _transcribe(audio)

    # ------------------------------------------------------------------
    # LLM - language model
    # ------------------------------------------------------------------

    def llm(self, text: str, messages: Optional[List[dict]] = None) -> str:
        """Get a response from the LLM.

        Args:
            text: User text to send.
            messages: Optional conversation history (list of role/content dicts).

        Returns:
            LLM response text.
        """
        from jarvis.pipeline.llm import stream_llm
        from jarvis.core.turn import Turn

        jarvis = self.jarvis
        turn = Turn()

        if messages is None:
            messages = list(jarvis.session.history)
        else:
            messages = list(messages)

        import queue
        tts_q: "queue.Queue[str]" = queue.Queue()

        full_text, _ = stream_llm(turn, messages, tts_q)
        return full_text

    # ------------------------------------------------------------------
    # Tools - function calling
    # ------------------------------------------------------------------

    def tools(self, name: str, raw_args: str = "{}") -> str:
        """Execute a registered tool by name.

        Args:
            name: Tool function name.
            raw_args: JSON string of arguments.

        Returns:
            Tool result string.
        """
        from jarvis.tools.registry import execute_tool
        return execute_tool(name, raw_args)

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        from jarvis.tools.registry import list_tools
        return list_tools()

    def tool_schemas(self) -> List[dict]:
        """Return JSON schemas for all registered tools."""
        from jarvis.tools.registry import get_tool_schemas
        return get_tool_schemas()

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------

    def play(self, audio_bytes: bytes) -> None:
        """Push raw PCM16 audio bytes to the output player.

        Args:
            audio_bytes: PCM16-encoded audio bytes.
        """
        jarvis = self.jarvis
        if jarvis.player is None:
            raise RuntimeError("Jarvis player not initialized.")
        jarvis.player.push(None, "", audio_bytes, True)

    # ------------------------------------------------------------------
    # Session / conversation
    # ------------------------------------------------------------------

    def history(self) -> List[dict]:
        """Return the current conversation history."""
        return list(self.jarvis.session.history)

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        from jarvis.config.settings import SYSTEM_PROMPT
        self.jarvis.session.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]


# ---------------------------------------------------------------------------
# App base class
# ---------------------------------------------------------------------------

class App:
    """Base class for a custom Jarvis app.

    Subclass and override ``on_text`` (and optionally ``on_start`` /
    ``on_stop``) to handle voice input however you want.

    Attributes:
        name: Human-readable app identifier.
        jarvis: The parent JARVIS instance, set at registration time.
    """

    name: str = "app"

    def __init__(self):
        self.jarvis = None
        self.components: Optional[Components] = None

    def _bind(self, jarvis: "JARVIS") -> None:
        """Called internally by register_app to wire up Jarvis."""
        self.jarvis = jarvis
        self.components = Components(jarvis)

    def on_start(self):
        """Called when the app is activated."""
        pass

    def on_stop(self):
        """Called when the app is deactivated."""
        pass

    def on_text(self, text: str) -> str:
        """Handle a user utterance and return a spoken response.

        Args:
            text: Transcribed user speech.

        Returns:
            Response text that Jarvis will speak back.
        """
        return f"App '{self.name}' received: {text}"

    def on_tool_result(self, tool_name: str, result: str) -> str:
        """Optional: post-process a tool result before it's spoken."""
        return result


# ---------------------------------------------------------------------------
# App registry
# ---------------------------------------------------------------------------

_APPS: Dict[str, App] = {}


def register_app(app: App, jarvis: "JARVIS") -> None:
    """Register an app instance by its ``name`` attribute, binding it to Jarvis.

    Args:
        app: App instance to register.
        jarvis: Parent JARVIS instance (provides components).
    """
    app._bind(jarvis)
    _APPS[app.name] = app


def get_app(name: str) -> Optional[App]:
    """Look up a registered app by name."""
    return _APPS.get(name)


def list_apps() -> List[str]:
    """Return names of all registered apps."""
    return list(_APPS.keys())


def apps() -> Dict[str, App]:
    """Return the full app registry dict."""
    return dict(_APPS)


# ---------------------------------------------------------------------------
# Web app wrapper (FastAPI)
# ---------------------------------------------------------------------------

try:  # FastAPI is optional
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - FastAPI not installed
    FastAPI = None  # type: ignore[assignment]
    WebSocket = None  # type: ignore[assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment, misc]
    _FASTAPI_AVAILABLE = False

# Import run_turn lazily so the module imports without the full pipeline
try:
    from jarvis.pipeline.run_turn import run_turn  # type: ignore
except Exception:  # pragma: no cover - runtime import
    run_turn = None  # type: ignore[assignment]


class WebApp:
    """Expose Jarvis as a FastAPI web application.

    Endpoints:
        GET  /health        - health check
        POST /speak         - send text, get spoken response (JSON)
        WS   /ws            - websocket for streaming text/audio

    Requires: ``pip install fastapi uvicorn``

    Usage:
        from jarvis import JARVIS
        from jarvis.apps import WebApp

        jarvis = JARVIS()
        web = WebApp(jarvis)
        web.run(port=8000)
    """

    def __init__(self, jarvis: "JARVIS"):
        if not _FASTAPI_AVAILABLE:
            raise ImportError(
                "WebApp requires FastAPI and uvicorn. "
                "Install with: pip install fastapi uvicorn"
            )
        assert FastAPI is not None  # narrowed by _FASTAPI_AVAILABLE check
        self.jarvis = jarvis
        self.app = FastAPI(title="Jarvis WebApp", version="2.0.0")
        self._setup_routes()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        app = self.app
        jarvis = self.jarvis

        @app.get("/health")
        def health():
            return {"status": "ok", "app": "jarvis"}

        @app.post("/speak")
        def speak(payload: dict):
            text = payload.get("text", "")
            if not text:
                return {"error": "text is required"}
            # Queue a turn and speak (blocking-ish, best-effort)
            try:
                jarvis.session.start_turn(text, jarvis.player)
                turn = jarvis.session.turn
                turn.thread = __import__("threading").Thread(
                    target=run_turn,
                    args=(turn, list(jarvis.session.history), jarvis.player),
                    daemon=True,
                )
                turn.thread.start()
                return {"status": "speaking", "text": text}
            except Exception as exc:  # pragma: no cover - runtime
                return {"error": str(exc)}

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):  # type: ignore[valid-type]
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    # Echo back for now; extend to drive Jarvis turns
                    await websocket.send_text(f"echo: {data}")
            except WebSocketDisconnect:  # type: ignore[valid-type]
                pass

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the web app with uvicorn."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# ---------------------------------------------------------------------------
# Multi-Agent Mode
# ---------------------------------------------------------------------------

class Agent:
    """A named agent with its own system prompt and JARVIS instance.

    Agents can be switched at runtime. Each agent has its own conversation
    history (system prompt) but shares the same underlying audio pipeline.

    Usage:
        agent_a = Agent("assistant", "You are a helpful assistant.")
        agent_b = Agent("coder", "You are an expert programmer.")
        registry = AgentRegistry()
        registry.register(agent_a)
        registry.register(agent_b)
        registry.switch_to("coder")
    """

    def __init__(self, name: str, system_prompt: str, jarvis: "JARVIS"):
        self.name = name
        self.system_prompt = system_prompt
        self.jarvis = jarvis

    def activate(self) -> None:
        """Apply this agent's system prompt to the shared JARVIS session."""
        from jarvis.config.settings import SYSTEM_PROMPT
        self.jarvis.session.history[0] = {"role": "system", "content": self.system_prompt}
        print(f"[agent] Switched to '{self.name}'")


class AgentRegistry:
    """Manage multiple agents and switching between them."""

    def __init__(self, jarvis: "JARVIS"):
        self.jarvis = jarvis
        self.agents: Dict[str, Agent] = {}
        self.current: Optional[Agent] = None

    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self.agents[agent.name] = agent
        if self.current is None:
            self.current = agent
            agent.activate()
        print(f"[agent] Registered '{agent.name}'")

    def switch_to(self, name: str) -> bool:
        """Switch to the named agent. Returns True if found."""
        agent = self.agents.get(name)
        if agent is None:
            print(f"[agent] Unknown agent: {name}")
            return False
        self.current = agent
        agent.activate()
        return True

    def list_agents(self) -> List[str]:
        """Return names of all registered agents."""
        return list(self.agents.keys())

    def current_agent(self) -> Optional[str]:
        """Return the name of the current agent."""
        return self.current.name if self.current else None


__all__ = [
    "App",
    "Components",
    "Agent",
    "AgentRegistry",
    "register_app",
    "get_app",
    "list_apps",
    "apps",
    "WebApp",
]
