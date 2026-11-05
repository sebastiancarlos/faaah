#!/usr/bin/env python3
"""Real faaah tool-calling client using only the Python standard library.

1. Start faaah:
       $ faaah

2. Paste the printed agent prompt into your coding agent, so it starts the
   coordinator loop and begins answering prompts.

3. In another terminal, run:
       $ ./examples/tools.py "What's the weather in Lisbon?"
"""

import json
import os
import sys
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("FAAAH_BASE_URL", "http://127.0.0.1:8000/v1")
DEBUG = os.environ.get("FAAAH_DEBUG") == "1"

TOOLS = [
    {
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
]


def get_weather(city: str) -> str:
    """Get the current weather for a city from wttr.in."""
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%l:+%C+%t"
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8").strip()


def post_chat(messages: list[dict]) -> dict:
    """POST one chat request to faaah and return the first choice."""
    payload = json.dumps({"model": "faaah", "messages": messages, "tools": TOOLS}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer not-used",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        choice = json.loads(response.read().decode("utf-8"))["choices"][0]
    if DEBUG:
        print(json.dumps(choice, indent=2))
    return choice


def main(question: str) -> None:
    """Ask the question, run any requested tools, and print the final answer."""
    messages: list[dict] = [{"role": "user", "content": question}]
    while True:
        choice = post_chat(messages)
        message = choice["message"]
        messages.append(message)

        if choice["finish_reason"] != "tool_calls":
            print(message["content"])
            return

        print(f"{len(message['tool_calls'])} tool call(s) requested")
        for call in message["tool_calls"]:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            print(f"tool call: {name}({arguments})")
            if name == "get_weather":
                try:
                    result = get_weather(**arguments)
                except OSError as exc:
                    result = f"tool error: {exc}"
            else:
                result = f"unknown tool: {name}"
            print(f"tool result: {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
