# DEVELOPMENT.md - Jarvis App Flows

This document describes the key data flows in Jarvis. No em dashes are used
in this file.

## 1. Main Voice Loop

```
Mic input -> VAD (Silero) detects speech -> transcribe (Whisper)
  -> LLM streams tokens -> TTS streams audio -> Player plays it back
```

Runs continuously in `JARVIS.start()`. The assistant listens and speaks at the
same time (full-duplex).

### Steps

1. VAD worker reads microphone frames from `mic_queue`.
2. Echo cancellation is applied using the reference signal from the player.
3. Silero VAD produces a speech probability for each frame.
4. When speech is detected, frames are collected into an utterance.
5. On silence or max length, the utterance is sent to `utterance_queue`.
6. Main loop transcribes with Whisper.
7. Keyword triggers are checked against the transcript.
8. Stop commands interrupt the current turn.
9. A new turn starts: LLM streams, TTS generates audio, player plays it.

## 2. Barge-In / Interruption Flow

```
User speaks while assistant talking -> VAD detects barge-in
  -> Player.pause() + Session.cancel_current() stops TTS
  -> Turn is committed to history
  -> New turn starts normally
```

### Behavior

- While `session.assistant_active()` is true, VAD uses a stricter threshold
  (`BARGE_IN_THRESHOLD`) and fewer frames (`BARGE_IN_FRAMES`).
- When barge-in triggers, the player pauses immediately and the current turn
  is cancelled.
- The user's speech after barge-in is recorded and transcribed normally.

## 3. Tool Calling Flow

```
LLM streams response -> detects tool call -> execute_tool()
  -> result fed back to LLM -> LLM speaks final answer via TTS
```

### Steps

1. `stream_llm()` receives streaming tokens from the LLM.
2. If `tool_calls` appear in the delta, they are collected.
3. After streaming completes, `run_llm_with_tools()` iterates over tool calls.
4. Each tool is executed via `execute_tool()` with the `speak_fn` callback.
5. Tool results are appended to the conversation as `tool` role messages.
6. The LLM is called again with the results to generate the final response.
7. The final text is sent to TTS for playback.

## 4. Auto-Speak Tool Results

Tool results are spoken aloud automatically instead of being read silently.

### Wiring

- `JARVIS.speak()` sends text through the TTS pipeline.
- `run_turn()` passes `speak_fn=self.speak` to `run_llm_with_tools()`.
- `run_llm_with_tools()` passes `speak_fn` to `execute_tool()`.
- `execute_tool()` calls `speak_fn(result_str)` after each tool result.

This means every tool call made by the LLM is spoken to the user.

## 5. Event Trigger Flow (.jext)

A `.jext` file can declare a `Trigger:` section for automatic invocation.

### Supported Trigger Types

- `Trigger: every N seconds` - interval trigger (background thread)
- `Trigger: every N minutes` - interval trigger
- `Trigger: every N hours` - interval trigger
- `Trigger: cron <expression>` - cron schedule (requires croniter)
- `Trigger: on keyword <word>` - fires when user says the word

### Interval / Cron Flow

```
Trigger declared in .jext -> TriggerManager.register_from_schema()
  -> Background thread checks is_due() every second
  -> When due, handler fires -> HTTP POST or Python function executes
  -> next_run is rescheduled
```

### Keyword Flow

```
User speaks -> VAD transcribes -> JARVIS.main loop calls
  trigger_manager.check_keyword(text) -> matching handler fires immediately
```

Keyword triggers are checked on every transcribed utterance, so saying the
keyword triggers the tool instantly.

## 6. HTTP API Tool Flow (.jext Endpoint)

```
Tool declares Endpoint: POST https://api.example.com/
  -> LLM or trigger calls tool -> _make_http_handler substitutes {params}
  -> urllib makes the request -> returns {status, body} JSON
```

### Placeholder Substitution

- `{param}` in the URL is replaced with the argument value.
- `{param}` in the Body is replaced with the argument value.
- Unused GET arguments are appended as query string.
- `Content-Type: application/json` is added automatically for POST bodies.

### Example .jext File

```
Tool: post_status

Description:
POST status to webhook.

Endpoint:
POST https://hooks.example.com/status

Headers:
Content-Type: application/json

Body:
{"ok": true}

Trigger:
every 60 seconds
```

## 7. Conversation Persistence Flow

```
JARVIS(history_file="conversation_history.json")
  -> Session.load_history() on start
  -> Conversation proceeds normally
  -> JARVIS.stop() -> Session.save_history() on exit
```

### Files

- `session.save_history(path)` - writes history to JSON.
- `session.load_history(path)` - loads history from JSON.
- History is a list of `{role, content}` dicts.
- System prompt is preserved as the first message.

## 8. Multi-Agent Flow

```
AgentRegistry registers multiple agents
  -> registry.switch_to("coder") changes system prompt
  -> All agents share the same audio pipeline and JARVIS instance
```

### Classes

- `Agent(name, system_prompt, jarvis)` - named agent with its own prompt.
- `AgentRegistry(jarvis)` - manages agents and switching.

### Methods

- `register(agent)` - adds an agent; first registered becomes current.
- `switch_to(name)` - switches active agent, applies system prompt.
- `list_agents()` - returns all agent names.
- `current_agent()` - returns the current agent name.

### Usage

```python
registry = AgentRegistry(jarvis)
registry.register(Agent("coder", "You are an expert programmer.", jarvis))
registry.switch_to("coder")
```

## 9. Config Validation Flow

```
JARVIS.start() -> validate_config()
  -> Checks TTS server at http://127.0.0.1:8080
  -> Checks LLM server at http://127.0.0.1:1337
  -> Prints errors and returns if any server is unreachable
```

### Checks

- TTS server: `OPTIONS` request to `TTS_URL`, expects 200/204/405.
- LLM server: `GET` request to `LLM_MODELS_URL`, expects 200.
- Errors are printed with clear messages before Jarvis exits.

## 10. Audio Device Selection

```
python example.py --mic-device 1 --output-device 2
  -> Sets MIC_DEVICE and OUTPUT_DEVICE in settings
  -> Passed to sd.InputStream and sd.RawOutputStream
```

### Settings

- `MIC_DEVICE` - input device index (None = system default).
- `OUTPUT_DEVICE` - output device index (None = system default).

## 11. Tool Caching Flow

```
execute_tool(name, args, use_cache=True)
  -> Check _TOOL_CACHE[(name, args)]
  -> If not expired, return cached result
  -> Else execute tool, cache result with TTL
```

### Settings

- `_TOOL_CACHE_TTL` - default 300 seconds (5 minutes).
- `set_tool_cache_ttl(seconds)` - change TTL.
- `clear_tool_cache()` - clear all cached results.

## 12. WebApp Flow (FastAPI)

```
HTTP client -> POST /speak or WS /ws -> JARVIS session.start_turn()
  -> run_turn in background -> TTS plays -> response returned
```

### Endpoints

- `GET /health` - health check.
- `POST /speak` - send text, start a turn, return speaking status.
- `WS /ws` - WebSocket for streaming text/audio.

### Requirements

- `pip install fastapi uvicorn`

## Coding Conventions

- No em dashes (-) in code comments or documentation. Use hyphens (-) or
  colons (:) instead.
- All Python files must compile with `py_compile`.
- All tests must pass with `pytest tests/ -v`.
- New tools are added with the `@tool` decorator in `jarvis/tools/builtin.py`.
- New `.jext` files go in `tools_example/`.
- Triggers are declared in `.jext` files with a `Trigger:` section.
