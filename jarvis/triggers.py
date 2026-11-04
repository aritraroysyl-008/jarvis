"""
jarvis.triggers - Event trigger system for .jext tools.

A .jext file can declare a ``Trigger:`` section that tells Jarvis to
automatically invoke the tool on a schedule or when a keyword is spoken:

    Tool: post_status

    Description:
    POST current status to a webhook every 5 minutes.

    Endpoint:
    POST https://hooks.example.com/status

    Body:
    {"ok": true}

    Trigger:
    every 5 minutes

Supported trigger types:

    Trigger: every N seconds     - run every N seconds
    Trigger: every N minutes     - run every N minutes
    Trigger: every N hours       - run every N hours
    Trigger: cron <expr>         - run on a cron schedule
    Trigger: on keyword <word>   - run when user says <word>

The trigger manager is wired into JARVIS so keyword triggers are checked
against every utterance, and interval/cron triggers run on background
threads.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:  # croniter is optional
    from croniter import croniter
    _CRONITER_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    croniter = None  # type: ignore[assignment]
    _CRONITER_AVAILABLE = False


class Trigger:
    """A single trigger bound to a tool handler.

    Attributes:
        name: Tool name this trigger fires.
        handler: Callable to invoke when triggered.
        trigger_type: ``interval``, ``cron``, or ``keyword``.
        every: Interval count (interval triggers only).
        unit: ``second``, ``minute``, or ``hour`` (interval only).
        expression: Cron expression (cron triggers only).
        keyword: Keyword to match (keyword triggers only).
        interval_seconds: Computed sleep interval (interval triggers only).
        next_run: Next scheduled run time (interval/cron only).
        enabled: Whether the trigger is active.
    """

    def __init__(
        self,
        name: str,
        handler: Callable[..., str],
        trigger_type: str,
        every: Optional[int] = None,
        unit: Optional[str] = None,
        expression: Optional[str] = None,
        keyword: Optional[str] = None,
    ):
        self.name = name
        self.handler = handler
        self.trigger_type = trigger_type
        self.every = every
        self.unit = unit
        self.expression = expression
        self.keyword = keyword
        self.enabled = True

        if trigger_type == "interval" and every and unit:
            mult = {"second": 1, "minute": 60, "hour": 3600}.get(unit, 1)
            self.interval_seconds = every * mult
        else:
            self.interval_seconds = 60

        if trigger_type == "interval":
            self.next_run = time.time() + self.interval_seconds
        elif trigger_type == "cron" and _CRONITER_AVAILABLE and expression:
            base = croniter(expression)
            self.next_run = base.get_next(float)
        else:
            self.next_run = None

    def is_due(self) -> bool:
        """Return True if an interval/cron trigger should fire now."""
        if not self.enabled or self.trigger_type not in ("interval", "cron"):
            return False
        if self.next_run is None:
            return False
        return time.time() >= self.next_run

    def schedule_next(self) -> None:
        """Compute the next run time after firing."""
        if self.trigger_type == "interval":
            self.next_run = time.time() + self.interval_seconds
        elif self.trigger_type == "cron" and _CRONITER_AVAILABLE and self.expression:
            base = croniter(self.expression, start_time=time.time())
            self.next_run = base.get_next(float)

    def matches_keyword(self, text: str) -> bool:
        """Return True if *text* contains the keyword (keyword triggers only)."""
        if self.trigger_type != "keyword" or not self.keyword:
            return False
        return self.keyword in text.lower()

    def fire(self, **kwargs) -> str:
        """Invoke the bound handler."""
        try:
            return self.handler(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime
            return f"[trigger {self.name}] error: {exc}"


class TriggerManager:
    """Manage and run triggers declared in .jext files.

    Usage:

        from jarvis.triggers import TriggerManager

        tm = TriggerManager()
        tm.register_trigger(name="my_tool", handler=my_handler, ...)
        tm.start()          # background thread for interval/cron triggers
        tm.check_keyword("some user text")  # call on each utterance
        tm.stop()
    """

    def __init__(self):
        self.triggers: Dict[str, Trigger] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_trigger(
        self,
        name: str,
        handler: Callable[..., str],
        trigger_type: str,
        every: Optional[int] = None,
        unit: Optional[str] = None,
        expression: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> None:
        """Register a trigger bound to *handler*."""
        trigger = Trigger(
            name=name,
            handler=handler,
            trigger_type=trigger_type,
            every=every,
            unit=unit,
            expression=expression,
            keyword=keyword,
        )
        with self._lock:
            self.triggers[name] = trigger
        print(f"[trigger] registered '{name}' ({trigger_type})")

    def register_from_schema(self, schema: Dict[str, Any], handler: Callable[..., str]) -> None:
        """Register a trigger from a parsed .jext schema dict."""
        trigger = schema.get("trigger")
        if not trigger:
            return
        ttype = trigger.get("type")
        name = schema["name"]

        if ttype == "interval":
            self.register_trigger(
                name=name,
                handler=handler,
                trigger_type="interval",
                every=trigger.get("every"),
                unit=trigger.get("unit"),
            )
        elif ttype == "cron":
            self.register_trigger(
                name=name,
                handler=handler,
                trigger_type="cron",
                expression=trigger.get("expression"),
            )
        elif ttype == "keyword":
            self.register_trigger(
                name=name,
                handler=handler,
                trigger_type="keyword",
                keyword=trigger.get("keyword"),
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background thread for interval/cron triggers."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[trigger] manager started")

    def stop(self) -> None:
        """Stop the background trigger thread."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        """Background loop: fire due interval/cron triggers."""
        while not self._stop.is_set():
            now = time.time()
            due: List[Trigger] = []
            with self._lock:
                for t in self.triggers.values():
                    if t.is_due():
                        due.append(t)
            for t in due:
                try:
                    result = t.fire()
                    print(f"[trigger] {t.name} fired -> {result[:120]}")
                except Exception as exc:  # pragma: no cover - runtime
                    print(f"[trigger] {t.name} error: {exc}")
                finally:
                    t.schedule_next()
            # Sleep a short interval; 1 second resolution is fine
            self._stop.wait(1.0)

    # ------------------------------------------------------------------
    # Keyword checking (called from JARVIS main loop on each utterance)
    # ------------------------------------------------------------------

    def check_keyword(self, text: str) -> Optional[str]:
        """Check *text* against all keyword triggers.

        Returns the handler result if a keyword matched, else None.
        """
        if not text:
            return None
        with self._lock:
            matches = [t for t in self.triggers.values() if t.matches_keyword(text)]
        for t in matches:
            try:
                result = t.fire()
                print(f"[trigger] keyword '{t.keyword}' matched -> {result[:120]}")
                return result
            except Exception as exc:  # pragma: no cover - runtime
                print(f"[trigger] {t.name} keyword error: {exc}")
        return None

    def list_triggers(self) -> List[Dict[str, Any]]:
        """Return info about all registered triggers."""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "type": t.trigger_type,
                    "enabled": t.enabled,
                    "keyword": t.keyword,
                    "every": t.every,
                    "unit": t.unit,
                    "expression": t.expression,
                    "next_run": t.next_run,
                }
                for t in self.triggers.values()
            ]
