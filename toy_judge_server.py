#!/usr/bin/env python3
"""Minimal OpenAI-compatible server that mimics an LLM judge.

Compares the model response against the ground truth answer from the user
prompt and returns "correct" or "incorrect" with a short explanation.
For testing evaluation/llm_judge.py without a real LLM.

Usage:
  python toy_judge_server.py --port 8903
"""

import argparse
import json
import random
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER_RE = re.compile(r"Ground truth answer:\s*(\(?\w+\)?)")
RESPONSE_RE = re.compile(r"Model generated response:\s*(.*?)(?:\n|$)")
LETTER_RE = re.compile(r"\(([a-z])\)", re.IGNORECASE)


def extract_answer_letter(text):
    m = LETTER_RE.search(text)
    return m.group(1).lower() if m else None


class JudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        # Extract user prompt
        messages = body.get("messages", [])
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

        # Parse ground truth and model response
        gt_match = ANSWER_RE.search(user_prompt)
        resp_match = RESPONSE_RE.search(user_prompt)
        ground_truth = gt_match.group(1) if gt_match else ""
        model_response = resp_match.group(1).strip() if resp_match else ""

        gt_letter = extract_answer_letter(ground_truth)
        mr_letter = extract_answer_letter(model_response)

        # Decide correctness
        if gt_letter and mr_letter and gt_letter == mr_letter:
            judgement = "correct"
            explanation = (
                f"The model chose {model_response} which matches "
                f"the ground truth {ground_truth}."
            )
        else:
            judgement = "incorrect"
            if mr_letter:
                explanation = (
                    f"The model chose {model_response} but the "
                    f"correct answer is {ground_truth}."
                )
            else:
                explanation = (
                    f"The model did not select a valid option. "
                    f"The correct answer is {ground_truth}."
                )

        time.sleep(random.uniform(0.05, 0.2))

        content = f"Explanation: {explanation}\nJudgement: {judgement}"

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
        print(f"[toy-judge] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8903)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), JudgeHandler)
    print(f"[toy-judge] listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[toy-judge] stopped")
        server.server_close()


if __name__ == "__main__":
    main()
