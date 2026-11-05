#!/usr/bin/env python
"""End-to-end test suite for faaah."""

import http.client
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from faaah import (
    ToolReplyError,
    flatten_content,
    normalize_tool_choice,
    normalize_tools,
    parse_tool_reply,
    render_history,
    render_tools,
)

ROOT = Path(__file__).resolve().parent


def _cmd(*args: str) -> list[str]:
    # Prefer the project's own console script installed by `uv sync`
    local = ROOT / ".venv" / "bin" / "faaah"
    if local.is_file():
        return [str(local), *args]
    return ["uv", "run", "faaah", *args]


def faaah(*args: str, cwd: str | Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(_cmd(*args), cwd=cwd, capture_output=True, text=True, timeout=30)


def free_port() -> int:
    """return a free port number"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestHelpers(unittest.TestCase):
    def test_flatten_content_passes_through_string(self):
        self.assertEqual(flatten_content("hello"), "hello")

    def test_flatten_content_handles_none(self):
        self.assertEqual(flatten_content(None), "")

    def test_flatten_content_joins_text_parts(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        self.assertEqual(flatten_content(parts), "a\nb")

    def test_flatten_content_marks_non_text_parts(self):
        parts = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}]
        self.assertEqual(flatten_content(parts), "look\n[image_url omitted]")

    def test_flatten_content_accepts_plain_strings_in_parts(self):
        self.assertEqual(flatten_content(["a", "b"]), "a\nb")

    def test_flatten_content_stringifies_other_values(self):
        self.assertEqual(flatten_content(42), "42")


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class TestToolHelpers(unittest.TestCase):
    def setUp(self):
        self.tools = normalize_tools([WEATHER_TOOL])

    def test_render_tools_includes_tool_definition(self):
        prompt = render_tools(self.tools)
        self.assertIn("get_weather", prompt)
        self.assertIn('"tool_calls"', prompt)

    def test_render_tools_required_and_single_call(self):
        prompt = render_tools(self.tools, mode="required", parallel=False)
        self.assertIn("You MUST request at least one tool call", prompt)
        self.assertIn("Request exactly one tool call", prompt)

    def test_normalize_tools_extracts_function_fields(self):
        self.assertEqual(
            self.tools,
            [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": WEATHER_TOOL["function"]["parameters"],
                }
            ],
        )

    def test_normalize_tools_rejects_non_list(self):
        with self.assertRaises(ValueError):
            normalize_tools({"type": "function"})

    def test_normalize_tools_rejects_nameless_tool(self):
        with self.assertRaises(ValueError):
            normalize_tools([{"type": "function", "function": {}}])

    def test_parse_plain_text_returns_none(self):
        self.assertIsNone(parse_tool_reply("just an answer", self.tools, "00001"))

    def test_parse_tool_call_returns_openai_message(self):
        answer = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]}'
        message = parse_tool_reply(answer, self.tools, "00042")
        assert message is not None
        self.assertEqual(message["role"], "assistant")
        self.assertIsNone(message["content"])
        call = message["tool_calls"][0]
        self.assertEqual(call["id"], "call_00042_0")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Lisbon"})

    def test_parse_strips_code_fences(self):
        answer = (
            "```json\n"
            '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Porto"}}]}\n'
            "```"
        )
        message = parse_tool_reply(answer, self.tools, "00007")
        assert message is not None
        self.assertEqual(message["tool_calls"][0]["id"], "call_00007_0")

    def test_parse_accepts_arguments_as_json_string(self):
        answer = (
            '{"tool_calls": [{"name": "get_weather", "arguments": "{\\"city\\": \\"Braga\\"}"}]}'
        )
        message = parse_tool_reply(answer, self.tools, "00008")
        assert message is not None
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]), {"city": "Braga"}
        )

    def test_parse_ignores_json_with_extra_keys(self):
        answer = '{"tool_calls": [], "note": "hello"}'
        self.assertIsNone(parse_tool_reply(answer, self.tools, "00009"))

    def test_parse_rejects_empty_call_list(self):
        with self.assertRaises(ToolReplyError):
            parse_tool_reply('{"tool_calls": []}', self.tools, "00010")

    def test_parse_rejects_unknown_tool(self):
        answer = '{"tool_calls": [{"name": "get_wether", "arguments": {}}]}'
        with self.assertRaises(ToolReplyError):
            parse_tool_reply(answer, self.tools, "00011")

    def test_parse_rejects_missing_required_argument(self):
        answer = '{"tool_calls": [{"name": "get_weather", "arguments": {}}]}'
        with self.assertRaises(ToolReplyError):
            parse_tool_reply(answer, self.tools, "00012")

    def test_parse_rejects_non_object_arguments(self):
        answer = '{"tool_calls": [{"name": "get_weather", "arguments": ["Lisbon"]}]}'
        with self.assertRaises(ToolReplyError):
            parse_tool_reply(answer, self.tools, "00013")

    def test_render_history_preserves_tool_turns(self):
        messages = [
            {"role": "user", "content": "weather in Lisbon?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_00001_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Lisbon"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_00001_0", "content": "Sunny, 24C"},
        ]
        rendered = render_history(messages)
        self.assertIn("<assistant_tool_calls>", rendered)
        self.assertIn("call_00001_0: get_weather", rendered)
        self.assertIn('call_id="call_00001_0" name="get_weather"', rendered)
        self.assertIn("Sunny, 24C", rendered)

    def test_render_history_resolves_tool_name_from_assistant_call(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_00001_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_00001_0", "content": "Sunny"},
        ]
        self.assertIn('name="get_weather"', render_history(messages))

    def test_tool_choice_defaults_to_auto(self):
        self.assertEqual(normalize_tool_choice(None, self.tools), ("auto", None))

    def test_tool_choice_defaults_to_none_without_tools(self):
        self.assertEqual(normalize_tool_choice(None, []), ("none", None))

    def test_tool_choice_none(self):
        self.assertEqual(normalize_tool_choice("none", self.tools), ("none", None))

    def test_tool_choice_required(self):
        self.assertEqual(normalize_tool_choice("required", self.tools), ("required", None))

    def test_tool_choice_required_needs_tools(self):
        with self.assertRaises(ValueError):
            normalize_tool_choice("required", [])

    def test_tool_choice_forced_function(self):
        choice = {"type": "function", "function": {"name": "get_weather"}}
        self.assertEqual(normalize_tool_choice(choice, self.tools), ("forced", "get_weather"))

    def test_tool_choice_rejects_unknown_function(self):
        choice = {"type": "function", "function": {"name": "nope"}}
        with self.assertRaises(ValueError):
            normalize_tool_choice(choice, self.tools)

    def test_tool_choice_rejects_garbage(self):
        with self.assertRaises(ValueError):
            normalize_tool_choice("sometimes", self.tools)

    def test_parse_required_rejects_plain_text(self):
        with self.assertRaises(ToolReplyError):
            parse_tool_reply("hello", self.tools, "00014", required=True)

    def test_parse_required_accepts_tool_call(self):
        answer = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Faro"}}]}'
        message = parse_tool_reply(answer, self.tools, "00015", required=True)
        assert message is not None
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")

    def test_parse_forced_rejects_other_tool(self):
        tools = normalize_tools([WEATHER_TOOL, TIME_TOOL])
        answer = '{"tool_calls": [{"name": "get_time", "arguments": {"city": "Faro"}}]}'
        with self.assertRaises(ToolReplyError):
            parse_tool_reply(answer, tools, "00016", forced_name="get_weather")

    def test_parse_forced_accepts_named_tool(self):
        tools = normalize_tools([WEATHER_TOOL, TIME_TOOL])
        answer = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Faro"}}]}'
        message = parse_tool_reply(answer, tools, "00017", forced_name="get_weather")
        assert message is not None
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")

    def test_parse_parallel_false_rejects_multiple_calls(self):
        tools = normalize_tools([WEATHER_TOOL, TIME_TOOL])
        answer = json.dumps(
            {
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"city": "Faro"}},
                    {"name": "get_time", "arguments": {"city": "Faro"}},
                ]
            }
        )
        with self.assertRaises(ToolReplyError):
            parse_tool_reply(answer, tools, "00018", parallel=False)

    def test_parse_parallel_true_accepts_multiple_calls(self):
        tools = normalize_tools([WEATHER_TOOL, TIME_TOOL])
        answer = json.dumps(
            {
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"city": "Faro"}},
                    {"name": "get_time", "arguments": {"city": "Faro"}},
                ]
            }
        )
        message = parse_tool_reply(answer, tools, "00019", parallel=True)
        assert message is not None
        self.assertEqual([c["id"] for c in message["tool_calls"]], ["call_00019_0", "call_00019_1"])


class TestCLI(unittest.TestCase):
    def test_help_exits_zero(self):
        result = faaah("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--watch", result.stdout)

    def test_copy_flag_copies_agent_prompt_to_clipboard(self):
        """`-c/--copy` pipes the startup agent prompt into a clipboard helper."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clipboard_out = tmp_path / "clipboard.txt"
            # Fake `wl-copy` that records whatever it receives on stdin.
            (tmp_path / "wl-copy").write_text(f'#!/bin/sh\ncat > "{clipboard_out}"\n')
            (tmp_path / "wl-copy").chmod(0o755)
            env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
            proc = subprocess.Popen(
                [
                    *_cmd(),
                    "--port",
                    str(free_port()),
                    "--queue",
                    str(tmp_path / "queue"),
                    "--log-dir",
                    "",
                    "-c",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(100):
                    if clipboard_out.exists() and clipboard_out.read_text():
                        break
                    time.sleep(0.1)
                self.assertTrue(clipboard_out.exists(), "clipboard helper was never called")
                self.assertTrue(clipboard_out.read_text().startswith("You coordinate"))
            finally:
                proc.terminate()
                proc.wait(timeout=5)

    def test_agent_message_prints_coordinator_instructions(self):
        result = faaah("--agent-message")
        self.assertEqual(result.returncode, 0)
        self.assertIn("`faaah --watch`", result.stdout)

    def test_agent_message_echoes_custom_queue(self):
        result = faaah("--agent-message", "--queue", "/tmp/mycustomq")
        self.assertIn("--queue /tmp/mycustomq", result.stdout)

    def test_agent_message_default_queue_referenced(self):
        # Passing no --queue should fall back to the default queue dir, which
        # means the coordinator uses bare `faaah --watch` (no --queue flag).
        result = faaah("--agent-message")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith("You coordinate"))
        self.assertNotIn("--queue", result.stdout)

    def test_watch_prints_pending_prompt_with_tilde(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as q:
            Path(q, "prompt-00001.txt").write_text("hey")
            result = faaah("--watch", "--queue", q)
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.stdout.strip().startswith("~/"))
            self.assertIn("prompt-00001.txt", result.stdout)

    def test_watch_without_queue_uses_default_dir(self):
        # Place a prompt in the default queue dir and confirm bare `--watch`
        # (no --queue) finds it. Clean up afterwards.
        default = (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "faaah" / "queue"
        )
        default.mkdir(parents=True, exist_ok=True)
        marker = default / "prompt-00099.txt"
        marker.write_text("x")
        try:
            result = faaah("--watch")
            self.assertEqual(result.returncode, 0)
            self.assertIn("prompt-00099.txt", result.stdout)
        finally:
            marker.unlink(missing_ok=True)

    def test_watch_blocks_when_nothing_pending(self):
        with tempfile.TemporaryDirectory() as q:
            try:
                subprocess.run(
                    _cmd("--watch", "--queue", q),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                self.fail("--watch returned despite no pending prompt")
            except subprocess.TimeoutExpired:
                pass

    def test_watch_returns_on_pending_prompt(self):
        with tempfile.TemporaryDirectory() as q:
            Path(q, "prompt-00003.txt").write_text("x")
            result = subprocess.run(
                _cmd("--watch", "--queue", q),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("prompt-00003.txt", result.stdout)


class TestHTTPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.queue = Path(cls.tmp.name)
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [*_cmd(), "--port", str(cls.port), "--queue", str(cls.queue), "--log-dir", ""],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for readiness by polling /v1/models.
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=1)
                c.request("GET", "/v1/models")
                c.getresponse().read()
                c.close()
                return
            except OSError:
                time.sleep(0.1)
        cls.proc.kill()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.tmp.cleanup()

    def _post(self, body: dict, path: str = "/v1/chat/completions"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            path,
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def _communicate(self, body: dict, answer: str):
        """POST body, answer the pending prompt, return (status, data, prompt, rid)."""
        results: list = []
        t = threading.Thread(target=lambda: results.append(self._post(body)), daemon=True)
        t.start()

        _, prompt, rid = self._next_prompt()
        (self.queue / f"response-{rid}.txt").write_text(answer)

        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        status, data = results[0]
        return status, data, prompt, rid

    def _next_prompt(self):
        """Block for the next pending prompt, return (path, text, rid)."""
        watch_result = faaah("--watch", "--queue", str(self.queue), cwd=ROOT)
        self.assertEqual(watch_result.returncode, 0)
        path = Path(watch_result.stdout.strip())
        return path, path.read_text(), path.stem.removeprefix("prompt-")

    def _wait_for_new_prompt(self, previous_rid: str):
        """Poll the queue until a prompt with a different id appears."""
        for _ in range(100):
            for path in sorted(self.queue.glob("prompt-*.txt")):
                rid = path.stem.removeprefix("prompt-")
                if rid != previous_rid and not (self.queue / f"response-{rid}.txt").exists():
                    return path, path.read_text(), rid
            time.sleep(0.05)
        self.fail(f"no new prompt appeared after {previous_rid}")

    def _wait_for_prompt(self, previous: set[Path]) -> Path:
        """Poll the queue until a prompt not in `previous` appears."""
        for _ in range(100):
            for path in sorted(self.queue.glob("prompt-*.txt")):
                if path not in previous:
                    return path
            time.sleep(0.05)
        self.fail("no new prompt appeared")

    def test_models_endpoint(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertEqual(body["data"][0]["id"], "faaah")

    def test_invalid_json_returns_400(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/chat/completions", body="not json")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertIn("error", body)

    def test_unknown_path_returns_404(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/nope")
        resp = conn.getresponse()
        conn.close()
        self.assertEqual(resp.status, 404)

    def test_multipart_content_is_flattened_into_prompt(self):
        """Newer LangChain clients send content as a list of parts."""
        body = {
            "model": "faaah",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "be terse"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                        {"type": "text", "text": "world"},
                    ],
                },
            ],
        }
        status, data, prompt, rid = self._communicate(body, "hi")
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "hi")
        self.assertIn("be terse", prompt)
        self.assertIn("hello", prompt)
        self.assertIn("world", prompt)
        self.assertIn("[image_url omitted]", prompt)
        self.assertIn(f"=== END MESSAGES {rid} ===", prompt)

    def test_chat_round_trip_via_agent(self):
        """POST a chat; a `faaah --watch` subprocess finds the prompt; write the
        agent answer; confirm the server returns it as OpenAI JSON."""
        results: list = []
        t = threading.Thread(
            target=lambda: results.append(
                self._post({"model": "faaah", "messages": [{"role": "user", "content": "hi"}]})
            ),
            daemon=True,
        )
        t.start()

        # The agent asks faaah --watch for the pending prompt path.
        watch_result = faaah("--watch", "--queue", str(self.queue), cwd=ROOT)
        self.assertEqual(watch_result.returncode, 0)
        prompt_path = Path(watch_result.stdout.strip())
        self.assertTrue(prompt_path.exists())

        # Worker writes the answer file.
        rid = prompt_path.stem.removeprefix("prompt-")
        (self.queue / f"response-{rid}.txt").write_text("hello agent")

        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        status, data = results[0]
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "hello agent")

    def test_tool_call_round_trip_and_follow_up(self):
        """A tool request returns finish_reason tool_calls; the follow-up with the
        tool result returns a plain final answer."""
        first = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": [WEATHER_TOOL],
        }
        reply = json.dumps(
            {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]}
        )
        status, data, prompt, rid = self._communicate(first, reply)
        self.assertEqual(status, 200)
        self.assertIn("get_weather", prompt)
        self.assertIn("TOOLS", prompt)
        begin = prompt.index(f"=== BEGIN MESSAGES {rid} ===")
        self.assertLess(prompt.index("TOOLS"), begin)
        self.assertLess(prompt.index("INSTRUCTIONS"), begin)
        self.assertLess(prompt.index("RULES"), begin)

        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["id"], f"call_{rid}_0")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Lisbon"})

        history = [
            {"role": "user", "content": "weather in Lisbon?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": choice["message"]["tool_calls"],
            },
            {"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 24C"},
        ]
        second = {"model": "faaah", "messages": history, "tools": [WEATHER_TOOL]}
        status2, data2, prompt2, _ = self._communicate(second, "It is sunny at 24C in Lisbon.")
        self.assertEqual(status2, 200)
        self.assertEqual(data2["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data2["choices"][0]["message"]["content"], "It is sunny at 24C in Lisbon.")
        self.assertIn("<tool_result", prompt2)
        self.assertIn(f'call_id="{call["id"]}"', prompt2)
        self.assertIn("Sunny, 24C", prompt2)

    def test_tools_present_plain_answer_returns_stop(self):
        body = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [WEATHER_TOOL],
        }
        status, data, _, _ = self._communicate(body, "hello there")
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["choices"][0]["message"]["content"], "hello there")

    def test_stream_true_returns_400(self):
        status, data = self._post(
            {"model": "faaah", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(status, 400)
        self.assertIn("stream", data["error"]["message"])

    def test_invalid_tool_choice_returns_400(self):
        status, _ = self._post(
            {
                "model": "faaah",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "sometimes",
            }
        )
        self.assertEqual(status, 400)

    def test_invalid_messages_returns_400(self):
        status, _ = self._post({"model": "faaah", "messages": "hello"})
        self.assertEqual(status, 400)

    def test_tool_choice_none_omits_tools_and_keeps_text(self):
        body = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        }
        reply = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]}'
        status, data, prompt, _ = self._communicate(body, reply)
        self.assertEqual(status, 200)
        self.assertNotIn('"name": "get_weather"', prompt)
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["choices"][0]["message"]["content"], reply)

    def test_forced_tool_choice_names_the_tool(self):
        body = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": [WEATHER_TOOL, TIME_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
        reply = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]}'
        status, data, prompt, _ = self._communicate(body, reply)
        self.assertEqual(status, 200)
        self.assertIn("You MUST request a call to the tool `get_weather`", prompt)
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_choice_required_retries_plain_text(self):
        body = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        }
        results: list = []
        t = threading.Thread(target=lambda: results.append(self._post(body)), daemon=True)
        t.start()

        _, prompt1, rid1 = self._next_prompt()
        self.assertIn("You MUST request at least one tool call", prompt1)
        (self.queue / f"response-{rid1}.txt").write_text("I would rather not.")

        _, prompt2, rid2 = self._wait_for_new_prompt(rid1)
        self.assertIn("PREVIOUS ATTEMPT REJECTED", prompt2)
        self.assertIn("I would rather not.", prompt2)
        (self.queue / f"response-{rid2}.txt").write_text(
            '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]}'
        )

        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        status, data = results[0]
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")

    def test_malformed_tool_reply_returns_502_after_retries(self):
        body = {
            "model": "faaah",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        }
        results: list = []
        t = threading.Thread(target=lambda: results.append(self._post(body)), daemon=True)
        t.start()

        _, _, rid = self._next_prompt()
        for _ in range(2):
            (self.queue / f"response-{rid}.txt").write_text("still plain text")
            _, prompt, rid = self._wait_for_new_prompt(rid)
            self.assertIn("PREVIOUS ATTEMPT REJECTED", prompt)
        (self.queue / f"response-{rid}.txt").write_text("still plain text")

        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        status, data = results[0]
        self.assertEqual(status, 502)
        self.assertEqual(data["error"]["raw"], "still plain text")

    def test_embeddings_echoes_openai_shape(self):
        """With the embeddings extra installed, POST /v1/embeddings returns
        OpenAI-shaped vectors (one per input, correctly indexed)."""
        status, data = self._post(
            {"model": "en_core_web_md", "input": ["hello world", "another phrase"]},
            path="/v1/embeddings",
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 2)
        for ent in data["data"]:
            self.assertEqual(ent["object"], "embedding")
            self.assertGreater(len(ent["embedding"]), 0)
        self.assertEqual([d["index"] for d in data["data"]], [0, 1])
        self.assertIn("usage", data)
        self.assertEqual(len(data["data"][0]["embedding"]), len(data["data"][1]["embedding"]))

    def test_embeddings_accepts_single_string(self):
        """OpenAI allows `input` as a plain string; faaah should too."""
        status, data = self._post(
            {"model": "en_core_web_md", "input": "single string"},
            path="/v1/embeddings",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(data["data"]), 1)

    def test_embeddings_invalid_json_returns_400(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/embeddings", body="not json")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertIn("error", body)

    def test_client_disconnect_removes_prompt(self):
        """A client that hangs up must not leave a prompt queued forever."""
        before = set(self.queue.glob("prompt-*.txt"))
        body = json.dumps(
            {"model": "faaah", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Connection: close\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        conn.sendall(request)
        prompt = self._wait_for_prompt(before)
        conn.close()

        for _ in range(100):
            if not prompt.exists():
                break
            time.sleep(0.05)
        self.assertFalse(prompt.exists(), "prompt was not removed after client disconnect")

        # The server must still serve new requests after the abandoned one.
        status, data, _, _ = self._communicate(
            {"model": "faaah", "messages": [{"role": "user", "content": "still alive?"}]},
            "yes",
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "yes")


class TestTimeoutCleanup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.queue = Path(cls.tmp.name)
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [
                *_cmd(),
                "--port",
                str(cls.port),
                "--queue",
                str(cls.queue),
                "--log-dir",
                "",
                "--timeout",
                "2",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=1)
                c.request("GET", "/v1/models")
                c.getresponse().read()
                c.close()
                return
            except OSError:
                time.sleep(0.1)
        cls.proc.kill()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.tmp.cleanup()

    def _post(self, body: dict):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def _wait_for_prompt(self, previous: set[Path]) -> Path:
        for _ in range(100):
            for path in sorted(self.queue.glob("prompt-*.txt")):
                if path not in previous:
                    return path
            time.sleep(0.05)
        self.fail("no prompt appeared")

    def test_timeout_removes_prompt(self):
        before = set(self.queue.glob("prompt-*.txt"))
        results: list = []
        t = threading.Thread(
            target=lambda: results.append(
                self._post({"model": "faaah", "messages": [{"role": "user", "content": "hi"}]})
            ),
            daemon=True,
        )
        t.start()
        prompt = self._wait_for_prompt(before)
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        status, data = results[0]
        self.assertEqual(status, 504)
        self.assertIn("timeout", data["error"]["message"])
        self.assertFalse(prompt.exists(), "prompt was not removed after timeout")


if __name__ == "__main__":
    unittest.main()
