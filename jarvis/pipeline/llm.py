"""
Jarvis Voice Assistant - Pipeline: LLM Streaming

Handles streaming LLM requests with function calling support.
"""

import json
import queue
import re
import uuid

import requests

from jarvis.config.settings import (
    LLM_CHAT_URL,
    LLM_MODEL,
    MAX_CHUNK_CHARS,
    CHUNK_SENTENCES,
    FIRST_CHUNK_SENTENCES,
)
from jarvis.tools import get_tool_schemas, execute_tool
from jarvis.core.turn import Turn

# Sentence splitting regex
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_for_tts(text: str) -> str:
    """Remove markdown/formatting for cleaner TTS."""
    text = re.sub(r"[*_#`~>|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_ready(buffer: str):
    """
    Split buffer into complete sentences and remainder.

    Returns:
        (ready_sentences, remaining_buffer)
    """
    parts = SENTENCE_SPLIT_RE.split(buffer)
    ready, rest = parts[:-1], parts[-1]
    if len(rest) > MAX_CHUNK_CHARS:
        i = rest.rfind(", ")
        if i > 40:
            ready.append(rest[:i + 1])
            rest = rest[i + 2:]
    return ready, rest


def queue_sentence(turn: Turn, tts_q: queue.Queue, sentence: str):
    """Queue a sentence for TTS if turn not cancelled."""
    sentence = clean_for_tts(sentence)
    if sentence and re.search(r"\w", sentence) and not turn.cancel.is_set():
        turn.said.append(sentence)
        tts_q.put(sentence)


def stream_llm(turn: Turn, messages: list, tts_q: queue.Queue) -> tuple[str, list]:
    """
    Stream one LLM completion. Speaks text as it arrives, collects tool calls.

    Args:
        turn: Current turn (for cancellation)
        messages: Conversation history
        tts_q: Queue to send sentences for TTS

    Returns:
        (full_text, tool_calls_list)
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }

    schemas = get_tool_schemas()
    if schemas:
        payload["tools"] = schemas
        payload["tool_choice"] = "auto"

    full, buffer, calls = [], "", {}
    sentences = []
    target = FIRST_CHUNK_SENTENCES

    resp = requests.post(LLM_CHAT_URL, json=payload, stream=True, timeout=(10, 120))
    turn.llm_resp = resp

    try:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if turn.cancel.is_set():
                break
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data.strip() == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta") or {}

            # Handle text content
            content = delta.get("content") or ""
            if content:
                full.append(content)
                print(content, end="", flush=True)
                buffer += content

                ready, buffer = split_ready(buffer)
                for sent in ready:
                    if sent.strip():
                        sentences.append(sent.strip())
                    if len(sentences) >= target:
                        queue_sentence(turn, tts_q, " ".join(sentences))
                        sentences = []
                        target = CHUNK_SENTENCES

            # Handle tool calls
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

        # Flush remaining sentences
        tail = " ".join(sentences + [buffer.strip()]).strip()
        if tail:
            queue_sentence(turn, tts_q, tail)

    finally:
        try:
            resp.close()
        except Exception:
            pass
        turn.llm_resp = None

    # Normalize tool calls
    tool_calls = []
    for idx in sorted(calls):
        c = calls[idx]
        if c["name"]:
            c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(c)

    return "".join(full), tool_calls


def run_llm_with_tools(turn: Turn, messages: list, tts_q: queue.Queue, max_rounds: int = 4) -> str:
    """
    Run LLM with tool calling loop.

    Args:
        turn: Current turn
        messages: Initial conversation history
        tts_q: TTS queue
        max_rounds: Maximum tool calling rounds

    Returns:
        Final assistant text response
    """
    msgs = list(messages)

    for _ in range(max_rounds):
        text, calls = stream_llm(turn, msgs, tts_q)
        if turn.cancel.is_set():
            return ""

        if not calls:
            turn.final_text = text
            return text

        # Add assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
                for c in calls
            ],
        }

        # Execute tools
        results = []
        for c in calls:
            if turn.cancel.is_set():
                return ""
            print(f"\n[tool] {c['name']}({c['arguments']})", flush=True)
            out = execute_tool(c["name"], c["arguments"])
            print(f"[tool] -> {out}", flush=True)
            results.append({"role": "tool", "tool_call_id": c["id"], "content": out})

        # Add to conversation and turn tracking
        msgs.append(assistant_msg)
        msgs.extend(results)
        turn.new_messages.extend([assistant_msg, *results])

    # Max rounds reached
    queue_sentence(turn, tts_q, "Sorry, I got stuck on that one.")
    return ""