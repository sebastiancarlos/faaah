"""faaah - Filesystem As An AI Handler.

An OpenAI-compatible HTTP server with zero dependencies. It speaks standard
`POST /v1/chat/completions`: each call is turned into a plain-text prompt file
in a shared queue directory, an agent (opencode, claude, vLLM, a human,
whatever) reads it, writes the matching response file, and the server returns
the answer as an OpenAI-shaped JSON response.

Drop any OpenAI client's `base_url` at this server and it is backed by *your*
agent instead of a hosted LLM.
"""

import argparse
import json
import logging
import os
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

# --- ANSI terminal colors ---
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[90m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

logger = logging.getLogger("faaah")


class _ColoredFormatter(logging.Formatter):
    """Compact, colored log lines: `HH:MM:SS LEVEL  message`."""

    _LEVEL_COLOR: ClassVar[dict[str, str]] = {
        "DEBUG": DIM,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": RED + BOLD,
    }

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt or "%H:%M:%S")
        level_color = self._LEVEL_COLOR.get(record.levelname, "")
        prefix = f"{level_color}{record.levelname:<7}{RESET}"
        line = f"{timestamp}  {prefix}  {record.getMessage()}"
        if record.exc_info:
            line += self.formatException(record.exc_info)
        return line


def default_queue() -> Path:
    """Default queue dir under the user cache dir, respecting XDG_CACHE_HOME."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "faaah" / "queue"


def display_path(path: Path) -> str:
    """Render a path with `~` in place of the user's home directory."""
    resolved = path.resolve()
    home = str(Path.home())
    text = str(resolved)
    if text.startswith(home):
        return "~" + text[len(home) :]
    return text


def agent_message(queue: str | Path) -> str:
    """Instructions to paste into an AI coding agent to start the coordinator."""
    q = Path(queue)
    default = default_queue()
    watch_command = "faaah --watch" if q == default else f"faaah --watch --queue {display_path(q)}"
    return f"""You coordinate work to be performed by subagents. You never read
the actual work description, only delegate it. Repeat forever:

1. Run `{watch_command}` (blocking helper). It prints the path of a pending file
   prompt-<id>.txt. Blocks until one exists.
2. For the printed path, spawn ONE fresh worker subagent. Its task message
   carries only the prompt path: "Process the file at <path>." Do not include
   the prompt contents.
3. Ignore the worker's reply. Go back to step 1.

Never read any file yourself. Spawn a new worker per path; never batch or reuse them.

Failure handling is automatic: if a worker leaves no response file, faaah
--watch prints the same path again - just spawn a new worker."""


# --- Internal configuration ---
POLL_INTERVAL = 0.2

# --- Configuration, modifiable via CLI flags ---
QUEUE_DIR: Path = Path("faaah_queue")
HOST = "127.0.0.1"
PORT = 8000
TIMEOUT_SECONDS = 0.0

# --- Work ID global data ---
_COUNTER = 0
_COUNTER_LOCK = threading.Lock()


def next_id() -> str:
    """Return a short, monotonic, sortable work id."""
    global _COUNTER
    with _COUNTER_LOCK:
        _COUNTER += 1
        return f"{_COUNTER:05d}"


def atomic_write(path: Path, content: str) -> None:
    """Write a file atomically so the agent only ever sees complete files."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def prompt_id(path: Path) -> int | None:
    """Extract id from a file which contains it on the name."""
    name = path.name
    if not (name.startswith("prompt-") and name.endswith(".txt")):
        return None
    try:
        return int(name[len("prompt-") : -len(".txt")])
    except ValueError:
        return None


def ready_paths(queue_dir: str | Path) -> list[Path]:
    """Get list of prompt files with no response yet, oldest first."""
    root = Path(queue_dir)
    found: list[Path] = []
    try:
        entries = os.scandir(root)
    except OSError, PermissionError:
        return found
    with entries:
        for entry in entries:
            if not entry.name.startswith("prompt-") or not entry.name.endswith(".txt"):
                continue
            rid = prompt_id(Path(entry.name))
            if rid is None:
                continue
            if not (root / f"response-{rid:05d}.txt").is_file():
                found.append(root / entry.name)
    found.sort(key=lambda p: prompt_id(p) or 0)
    return found


def watch(queue_dir: str | Path) -> int:
    """Block until a pending request is ready, then print its path.

    Prints the next ready prompt path and returns 0. The agent calls this
    repeatedly to drive the loop; it blocks until a new prompt arrives.
    """
    while True:
        ready = ready_paths(queue_dir)
        if ready:
            shown = display_path(ready[0])
            logger.info(f"Ready prompt: {shown}")
            print(shown)
            return 0
        time.sleep(POLL_INTERVAL)


class Server(BaseHTTPRequestHandler):
    server_version = "faaah/0.1"

    def log_message(self, format, *args):
        status = format % args if args else format
        match = re.search(r"\s(\d{3})\s", " " + status + " ")
        code = int(match.group(1)) if match else 0
        color = GREEN if 200 <= code < 300 else (YELLOW if 400 <= code < 500 else RED)
        verb = status.split()[0].strip('"')
        path = status.split()[1] if len(status.split()) > 1 else "?"
        action = f"{verb} {path} responded with {color}{code}{RESET}"
        logger.info(f"{self.address_string()} {action}")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Mock some GET endpoints, in case a client relies on them."""
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "faaah",
                            "object": "model",
                            "created": 0,
                            "owned_by": "faaah",
                        }
                    ],
                },
            )
        else:
            self._json(
                404,
                {"error": {"message": "not found", "type": "invalid_request_error"}},
            )

    def do_POST(self):
        """On POST, write a prompt-ID.txt file. Wait for response-ID.txt file,
        and return response."""
        if self.path != "/v1/chat/completions":
            self._json(
                404,
                {"error": {"message": "not found", "type": "invalid_request_error"}},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
        except ValueError, json.JSONDecodeError:
            logger.warning("Rejected request: invalid JSON body.")
            self._json(
                400,
                {
                    "error": {
                        "message": "invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        messages: list = req.get("messages", [])
        model = req.get("model", "faaah")
        response_format = (req.get("response_format") or {}).get("type", "text")

        # --- Write prompt file

        rid = next_id()
        req_path = QUEUE_DIR / f"prompt-{rid}.txt"
        res_path = QUEUE_DIR / f"response-{rid}.txt"

        prompt = self._build_prompt(rid, messages, response_format, QUEUE_DIR)
        atomic_write(req_path, prompt)
        logger.info(
            f"Request {rid} published "
            f"(format={YELLOW}{response_format}{RESET}): "
            f"{GREEN}{display_path(req_path)}{RESET}"
        )
        logger.info(
            f"Request {rid} awaiting answer at file: {GREEN}{display_path(res_path)}{RESET}"
        )

        # --- Wait for response file

        try:
            answer = self._wait(res_path, req_path)
        except TimeoutError:
            logger.error(f"Request {rid} timed out after {TIMEOUT_SECONDS:.0f}s.")
            self._json(504, {"error": {"message": "agent timeout", "type": "server_error"}})
            return

        # --- Return POST response

        preview = f"{answer[:10]}..." if len(answer) > 10 else answer
        logger.info(f'Request {rid} answered ({len(answer)} chars). "{preview}"')

        self._json(
            200,
            {
                "id": f"chatcmpl-{rid}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    @staticmethod
    def _build_prompt(rid: str, messages: list, response_format: str, queue_dir: Path) -> str:
        """Build string content of a prompt-ID.txt file."""
        block = "\n".join(f"<{m.get('role', 'user')}>: {m.get('content', '')}" for m in messages)
        json_rule = (
            "- The committed file MUST contain a single valid JSON value.\n"
            if response_format != "text"
            else ""
        )
        draft = display_path(queue_dir / f"response-{rid}.txt.draft")
        final = display_path(queue_dir / f"response-{rid}.txt")
        queue_path = display_path(queue_dir)
        return f"""CHAT REQUEST
============
request_id: {rid}
response_format: {response_format}
=== BEGIN MESSAGES ===
{block}
=== END MESSAGES ===

INSTRUCTIONS
------------
Answer the conversation above using your LLM capabilities, and hand
in your answer in TWO phases so it is never read half-written:

1. DRAFT. Write your answer to the file
   `{draft}`.
2. REVIEW. Read that draft back and check it against the
   response_format above; fix problems by rewriting the draft.
3. COMMIT. Once the draft is correct, atomically rename it (a single
   `mv`) to `{final}`. This rename is the instant your
   answer becomes final. It can't be changed anymore.

RULES
-----
- Never modify, rename, or delete any other file in `{queue_path}`.
- The committed file must contain ONLY your raw answer, no preamble,
  no markdown fences, no extra commentary.
{json_rule}"""

    def _wait(self, res_path: Path, req_path: Path) -> str:
        """Poll until the agent writes the response file; clean up after."""
        start = time.monotonic()
        while True:
            if res_path.exists():
                try:
                    text = res_path.read_text(encoding="utf-8")
                except OSError:
                    time.sleep(POLL_INTERVAL)
                    continue
                req_path.unlink(missing_ok=True)
                return text
            if TIMEOUT_SECONDS > 0 and time.monotonic() - start > TIMEOUT_SECONDS:
                raise TimeoutError("agent did not respond in time")
            time.sleep(POLL_INTERVAL)


def main(argv: list | None = None) -> None:
    """Set up CLI, print banner message, start server."""
    parser = argparse.ArgumentParser(
        prog="faaah",
        description="Filesystem As An AI Handler: an OpenAI-compatible proxy "
        "backed by an AI agent working over text files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000).")
    parser.add_argument(
        "--queue",
        default=str(default_queue()),
        help="Directory where prompt/response files live (default: ~/.cache/faaah/queue).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Abort each call after N seconds. 0 (default) waits forever.",
    )
    parser.add_argument(
        "--agent-message",
        action="store_true",
        help="Print ONLY the agent prompt and exit (it's also printed on launch).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="For agent-side use: block until a pending prompt exists, print its path.",
    )
    args = parser.parse_args(argv)

    global QUEUE_DIR, HOST, PORT, TIMEOUT_SECONDS

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    _fmt = _ColoredFormatter()
    for handler in logging.getLogger().handlers:
        handler.setFormatter(_fmt)

    if args.agent_message:
        print(agent_message(Path(args.queue)))
        raise SystemExit(0)

    QUEUE_DIR = Path(args.queue)

    if args.watch:
        raise SystemExit(watch(QUEUE_DIR))

    HOST = args.host
    PORT = args.port
    TIMEOUT_SECONDS = args.timeout
    shutil.rmtree(QUEUE_DIR, ignore_errors=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Queue reset: {display_path(QUEUE_DIR)}")

    print(f"{BOLD}{CYAN}faaah{RESET} - {BOLD}Filesystem As An AI Handler{RESET}")
    print(f"  - listening on {BOLD}{HOST}:{PORT}{RESET}")
    print(f"  - queue dir:   {BLUE}{display_path(QUEUE_DIR)}{RESET}")
    print(f"\n{GREEN}Agent prompt {DIM}(also available at `faaah --agent-message`){RESET}:")
    print("-" * 70)
    print(agent_message(QUEUE_DIR))
    print("-" * 70 + "\n")
    print(f"{GREEN}Point OpenAI clients at{RESET} {BOLD}http://{HOST}:{PORT}/v1{RESET}")
    print(f"{GREEN}Or try:{RESET}\r")
    print(f"    curl -s http://{HOST}:{PORT}/v1/chat/completions \\{RESET}")
    print("      -H 'Content-Type: application/json' \\")
    print('      -d \'{"model":"faaah","messages":[{"role":"user","content":"hi"}]}\'\n')
    logger.info(f"Server starting on {HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped via KeyboardInterrupt.")
        print(f"{YELLOW}faaah stopped.{RESET}")


if __name__ == "__main__":
    main()
