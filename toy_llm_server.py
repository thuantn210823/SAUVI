#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat completions server for testing.

Returns random responses that pick one of the options parsed from the
instruction text. No audio processing, no model loading.

Usage:
  python toy_llm_server.py --port 8902
"""

import argparse
import json
import random
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

RESPONSE_TEMPLATES = [
    "The answer is {choice}.",
    "Based on the audio, the speaker is {choice}.",
    "I think the correct option is {choice}.",
    "After listening to the recording, {choice}.",
    "My answer is {choice}.",
    "{choice}",
]

OPTION_RE = re.compile(r"\(([a-z])\)\s*([^\s(]+)")


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        # Extract text prompt from messages
        messages = body.get("messages", [])
        text = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
                break
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        break
                if text:
                    break

        # Parse options and pick one randomly
        options = OPTION_RE.findall(text)
        if options:
            letter, value = random.choice(options)
            choice_str = f"({letter}) {value}"
        else:
            choice_str = text[:50] if text else "(a) unknown"

        # Small random delay
        time.sleep(random.uniform(0.05, 0.2))

        content = random.choice(RESPONSE_TEMPLATES).format(choice=choice_str)

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[toy-llm] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f"[toy-llm] listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[toy-llm] stopped")
        server.server_close()


if __name__ == "__main__":
    main()
