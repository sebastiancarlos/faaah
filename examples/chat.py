#!/usr/bin/env python3
"""Minimal faaah client using only the Python standard library.

Because faaah speaks the OpenAI API, this is all it takes — no SDK needed:
    $ ./examples/chat.py "your message here"
"""

import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8000/v1"

def chat(message: str) -> str:
    payload = json.dumps(
        {
            "model": "faaah",
            "messages": [{"role": "user", "content": message}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer not-used",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(chat(" ".join(sys.argv[1:])))
