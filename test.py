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


class TestCLI(unittest.TestCase):
    def test_help_exits_zero(self):
        result = faaah("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--watch", result.stdout)

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
            [*_cmd(), "--port", str(cls.port), "--queue", str(cls.queue)],
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


if __name__ == "__main__":
    unittest.main()
