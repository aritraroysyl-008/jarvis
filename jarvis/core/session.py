"""
Jarvis Voice Assistant - Core: Session Management

Manages conversation history, turn lifecycle, and history trimming.
"""

import threading

from jarvis.config.settings import SYSTEM_PROMPT, MAX_HISTORY
from jarvis.core.turn import Turn


class Session:
    """
    Manages the conversation session: history, current turn, and lifecycle.

    Responsibilities:
        - Maintain conversation history (with system prompt)
        - Start/cancel/commit turns
        - Trim history to stay within token limits
        - Handle barge-in and turn completion
    """

    def __init__(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turn: Turn | None = None
        self._lock = threading.Lock()

    def assistant_active(self) -> bool:
        """Check if assistant is currently speaking/processing."""
        with self._lock:
            t = self.turn
            return t is not None and not t.finished.is_set()

    def _trim(self):
        """Trim history to MAX_HISTORY, preserving system prompt and recent turns."""
        while len(self.history) > MAX_HISTORY:
            # Find the second user message (index 1 is first user after system)
            idx = next(
                (i for i in range(2, len(self.history)) if self.history[i]["role"] == "user"),
                None,
            )
            if idx is None:
                break
            del self.history[1:idx]

    def commit(self, turn: Turn):
        """
        Fold a finished/cancelled turn into history (main thread only).

        For cancelled turns: records what was actually spoken.
        For completed turns: records the full response.
        """
        if turn.committed:
            return

        if turn.thread is not None:
            turn.thread.join(timeout=5)

        turn.committed = True

        # Add any tool call / tool result messages
        self.history.extend(turn.new_messages)

        if turn.cancel.is_set():
            # Turn was interrupted - record what user actually heard
            spoken = " ".join(turn.spoken).strip()
            if spoken:
                self.history.append({"role": "assistant", "content": spoken + " ..."})
            elif turn.new_messages:
                self.history.append({"role": "assistant", "content": "..."})
        elif turn.final_text.strip():
            # Turn completed normally
            self.history.append({"role": "assistant", "content": turn.final_text.strip()})

    def commit_if_finished(self):
        """Commit the current turn if it's finished and not yet committed."""
        with self._lock:
            t = self.turn
            if t is not None and t.finished.is_set() and not t.committed:
                self.commit(t)

    def cancel_current(self):
        """Cancel the current turn (barge-in)."""
        with self._lock:
            t = self.turn
            if t is None:
                return
            if not t.finished.is_set():
                t.abort()
            self.commit(t)

    def start_turn(self, user_text: str, player) -> Turn:
        """
        Start a new assistant turn with the user's input.

        Args:
            user_text: Transcribed user speech
            player: Player instance (for flushing on barge-in)

        Returns:
            The new Turn object
        """
        self.cancel_current()

        # Append or merge user message
        if self.history[-1]["role"] == "user":
            self.history[-1]["content"] += " " + user_text
        else:
            self.history.append({"role": "user", "content": user_text})

        self._trim()

        # Create and start new turn
        turn = Turn()
        self.turn = turn
        return turn