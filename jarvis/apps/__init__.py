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


def register_app(app: App) -> None:
    """Register an app instance by its ``name`` attribute."""
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

__all__ = [
    "App",
    "register_app",
    "get_app",
    "list_apps",
    "apps",
    "WebApp",
]
