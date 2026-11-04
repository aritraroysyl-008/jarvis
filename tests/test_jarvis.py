"""
Jarvis - Pytest tests for tool system and core modules.
Run with: uv run pytest tests/ -v
"""

import pytest
from jarvis.tools import execute_tool, list_tools, get_tool_schemas, clear_tools, tool
from jarvis.config.settings import (
    TTS_URL, TTS_VOICE, LLM_CHAT_URL, LLM_MODEL,
    VAD_THRESHOLD, BARGE_IN_THRESHOLD, SYSTEM_PROMPT
)


# TOOLS TESTS

def test_list_tools_returns_builtin_tools():
    tools = list_tools()
    assert "get_current_time" in tools
    assert "calculate" in tools


def test_get_tool_schemas_has_correct_structure():
    schemas = get_tool_schemas()
    assert len(schemas) >= 6

    for schema in schemas:
        assert schema["type"] == "function"
        assert "function" in schema
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_get_current_time_returns_string():
    result = execute_tool("get_current_time", "{}")
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain day, month, year, time
    assert any(c.isdigit() for c in result)


def test_calculate_basic_operations():
    test_cases = [
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("100 / 4", "25.0"),
        ("2 ** 10", "1024"),
        ("(15 + 5) * 3", "60"),
    ]
    for expr, expected in test_cases:
        result = execute_tool("calculate", f'{{"expression": "{expr}"}}')
        assert result == expected, f"{expr} = {result}, expected {expected}"


def test_calculate_handles_errors():
    # Division by zero
    result = execute_tool("calculate", '{"expression": "1 / 0"}')
    assert "error" in result.lower() or "zerodivision" in result.lower()

    # Invalid syntax
    result = execute_tool("calculate", '{"expression": "2 ++ 2"}')
    # Our safe eval returns 4 for this (it parses as 2 + (+2))
    assert result == "4"


def test_run_python_executes_code():
    result = execute_tool("run_python", '{"code": "print(2 + 2)"}')
    assert "4" in result


def test_run_python_captures_stderr():
    result = execute_tool("run_python", '{"code": "1/0"}')
    assert "ZeroDivisionError" in result or "Error" in result


def test_set_reminder_stores_and_lists():
    from jarvis.tools.builtin import _REMINDERS
    _REMINDERS.clear()
    result = execute_tool("set_reminder", '{"message": "Call mom", "when": "today at 5pm"}')
    assert "Call mom" in result
    result = execute_tool("list_reminders", "{}")
    assert "Call mom" in result


def test_execute_unknown_tool_returns_error():
    result = execute_tool("nonexistent_tool", "{}")
    assert "error" in result.lower()
    assert "unknown tool" in result.lower()


def test_execute_invalid_json_returns_error():
    result = execute_tool("calculate", "not json")
    assert "error" in result.lower()
    assert "not valid json" in result.lower()


def test_tool_decorator_registers_new_tool():
    clear_tools()

    @tool("Test tool description.")
    def test_tool(x: int, y: str = "default") -> str:
        return f"{x}:{y}"

    tools = list_tools()
    assert "test_tool" in tools

    # Test execution
    result = execute_tool("test_tool", '{"x": 42, "y": "hello"}')
    assert result == "42:hello"

    # Test default parameter
    result = execute_tool("test_tool", '{"x": 1}')
    assert result == "1:default"

    # Restore builtins
    import jarvis.tools.builtin  # noqa: F401


def test_tool_schema_auto_generation():
    clear_tools()

    @tool("Auto schema tool.")
    def auto_tool(a: int, b: float, c: bool, d: str) -> str:
        return "ok"

    schemas = get_tool_schemas()
    assert len(schemas) == 1
    props = schemas[0]["function"]["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "boolean"
    assert props["d"]["type"] == "string"

    required = schemas[0]["function"]["parameters"]["required"]
    assert set(required) == {"a", "b", "c", "d"}

    import jarvis.tools.builtin  # noqa: F401


# CONFIG TESTS

def test_settings_has_required_values():
    assert TTS_URL == "http://127.0.0.1:8080/v1/audio/speech"
    assert TTS_VOICE == "alba"
    assert LLM_CHAT_URL == "http://127.0.0.1:1337/v1/chat/completions"
    assert isinstance(LLM_MODEL, str)

    assert VAD_THRESHOLD == 0.60
    assert BARGE_IN_THRESHOLD == 0.80
    assert SYSTEM_PROMPT.startswith("You are a helpful voice assistant")


# CORE TESTS (require torch, skip if not available)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_turn_creation():
    from jarvis.core.turn import Turn

    turn = Turn()
    assert not turn.cancel.is_set()
    assert not turn.finished.is_set()
    assert not turn.tts_done.is_set()
    assert turn.new_messages == []
    assert turn.final_text == ""
    assert turn.said == []
    assert turn.spoken == []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_turn_abort():
    from jarvis.core.turn import Turn

    turn = Turn()
    turn.abort()
    assert turn.cancel.is_set()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_session_creation():
    from jarvis.core.session import Session

    session = Session()
    assert len(session.history) == 1
    assert session.history[0]["role"] == "system"
    assert session.turn is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_session_trim():
    from jarvis.core.session import Session

    session = Session()
    # Add many messages
    for i in range(50):
        session.history.append({"role": "user", "content": f"msg {i}"})
        session.history.append({"role": "assistant", "content": f"reply {i}"})

    session._trim()
    assert len(session.history) <= 40  # MAX_HISTORY


# PIPELINE TESTS (require torch, skip if not available)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_clean_for_tts():
    from jarvis.pipeline.llm import clean_for_tts

    # Removes markdown
    assert clean_for_tts("**bold**") == "bold"
    assert clean_for_tts("*italic*") == "italic"
    assert clean_for_tts("# header") == "header"
    assert clean_for_tts("`code`") == "code"

    # Normalizes whitespace
    assert clean_for_tts("  hello   world  ") == "hello world"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_split_ready():
    from jarvis.pipeline.llm import split_ready

    # Complete sentences
    ready, rest = split_ready("Hello world. How are you? ")
    assert ready == ["Hello world.", "How are you?"]
    assert rest == ""

    # Incomplete sentence
    ready, rest = split_ready("Hello world. How are")
    assert ready == ["Hello world."]
    assert rest == "How are"

    # Long remainder splits at comma
    long = "A" * 200 + ", " + "B" * 200
    ready, rest = split_ready(long)
    assert len(ready) == 1
    assert ready[0].endswith(",")
    assert rest.startswith("B")


# AUDIO TESTS (require torch/whisper, skip if not available)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="whisper not installed")
def test_transcribe_empty_audio():
    from jarvis.audio.transcribe import transcribe
    import numpy as np

    # Too short audio
    audio = np.zeros(100, dtype=np.int16)
    result = transcribe(audio)
    assert result == ""


# JEXT PARSING TESTS

def test_parse_jext_basic_tool():
    from jarvis.jext import parse_jext
    text = """Tool: get_weather

    Description:
    Gets current weather information for a city.

    Parameters:
    city: Name of the city.
    unit: Temperature unit, Celsius or Fahrenheit."""
    schema = parse_jext(text)
    assert schema["name"] == "get_weather"
    assert "Gets current weather" in schema["description"]
    assert "city" in schema["parameters"]["properties"]
    assert "unit" in schema["parameters"]["properties"]
    assert schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["strict"] is True


def test_parse_jext_http_endpoint():
    from jarvis.jext import parse_jext
    text = """Tool: post_message

    Description:
    Send a message to a webhook.

    Endpoint:
    POST https://hooks.example.com/messages

    Headers:
    Content-Type: application/json

    Body:
    {"text": "{text}", "channel": "{channel}"}

    Parameters:
    text: The message text to send.
    channel: The channel to post to."""
    schema = parse_jext(text)
    assert schema["name"] == "post_message"
    assert schema["endpoint"]["method"] == "POST"
    assert schema["endpoint"]["url"] == "https://hooks.example.com/messages"
    assert schema["headers"]["Content-Type"] == "application/json"
    assert '"text"' in schema["body"]


def test_parse_jext_interval_trigger():
    from jarvis.jext import parse_jext
    text = """Tool: heartbeat_post

    Description:
    POST a heartbeat every 30 seconds.

    Endpoint:
    POST https://hooks.example.com/heartbeat

    Parameters:
    # none needed

    Trigger:
    every 30 seconds"""
    schema = parse_jext(text)
    assert schema["trigger"]["type"] == "interval"
    assert schema["trigger"]["every"] == 30
    assert schema["trigger"]["unit"] == "second"


def test_parse_jext_keyword_trigger():
    from jarvis.jext import parse_jext
    text = """Tool: keyword_alert

    Description:
    Respond when keyword spoken.

    Parameters:
    # none

    Trigger:
    on keyword alert"""
    schema = parse_jext(text)
    assert schema["trigger"]["type"] == "keyword"
    assert schema["trigger"]["keyword"] == "alert"


def test_infer_type_not_boolean_for_lon():
    from jarvis.jext import _infer_type
    # "lon" should NOT be inferred as boolean (it contains "on" as substring)
    assert _infer_type("lon", "Longitude of the location.") == "string"
    # But standalone "on" word should be boolean
    assert _infer_type("enabled", "Is it on?") == "boolean"


# TRIGGER MANAGER TESTS

def test_trigger_manager_keyword():
    from jarvis.triggers import TriggerManager
    results = []
    def handler(**kwargs):
        results.append(kwargs)
        return "handled!"

    tm = TriggerManager()
    tm.register_trigger(
        name="test_kw",
        handler=handler,
        trigger_type="keyword",
        keyword="alert",
    )
    result = tm.check_keyword("this is an alert message")
    assert result == "handled!"
    assert len(results) == 1

    # No match
    result = tm.check_keyword("hello world")
    assert result is None


def test_trigger_manager_interval_due():
    from jarvis.triggers import TriggerManager
    import time
    fired = []
    def handler(**kwargs):
        fired.append(True)
        return "fired"

    tm = TriggerManager()
    tm.register_trigger(
        name="fast_interval",
        handler=handler,
        trigger_type="interval",
        every=1,
        unit="second",
    )
    t = tm.triggers["fast_interval"]
    # Manually set next_run to past to test is_due
    t.next_run = time.time() - 1
    assert t.is_due() is True
    t.schedule_next()
    assert t.is_due() is False  # just scheduled, not due yet


def test_trigger_register_from_schema():
    from jarvis.triggers import TriggerManager
    from jarvis.jext import parse_jext_file

    tm = TriggerManager()
    def handler(**kwargs):
        return "handled!"

    schema = parse_jext_file("tools_example/keyword_alert.jext")
    assert "trigger" in schema
    assert schema["trigger"]["type"] == "keyword"
    assert schema["trigger"]["keyword"] == "alert"

    tm.register_from_schema(schema, handler)
    assert "keyword_alert" in tm.triggers
    result = tm.check_keyword("send an alert now")
    assert result == "handled!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])