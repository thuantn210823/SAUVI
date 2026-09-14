#!/usr/bin/env python3
"""Evaluate a Large Audio Language Model (LALM) on a JSONL manifest.

The manifest is a JSONL file following the MMAU schema (one JSON object per line).

For each item the script:
  1. Randomizes the option order and builds an instruction string.
  2. Calls the LALM via an OpenAI-compatible chat-completions endpoint
     (text prompt + local audio via file:// URL, same as call.sh).
  3. Enriches the item with "instruction", "response", and "label" keys.
  4. Writes all enriched items to <model_name>.jsonl.

Usage:
  python run_lalm.py [--manifest FILE] [--threads N] ...
"""

import argparse
import json
import os
import random
import re
import string
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_MANIFEST = "regional_dialect_recognition.jsonl"
DEFAULT_ENDPOINT = "http://127.0.0.1:8901/v1/chat/completions"
DEFAULT_MODEL = "/home/voice/data/voice/voice-models/Qwen3-Omni-30B-A3B-Instruct/"
LETTERS = string.ascii_lowercase  # a-z


def load_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def audio_url(audio_path):
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", audio_path):
        return audio_path
    abs_path = os.path.abspath(audio_path)
    return "file://" + urllib.parse.quote(abs_path)


def build_instruction(question, choices):
    parts = [question.rstrip().rstrip("?") + "?"] + [
        f"({LETTERS[i]}) {c}" for i, c in enumerate(choices)
    ]
    return " ".join(parts)


def build_label(answer, choices):
    idx = choices.index(answer)
    return f"({LETTERS[idx]}) {answer}"


def call_lalm(endpoint, model, prompt, audio, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url(audio)},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected response shape: {data}")


def run_item(args, item):
    shuffled = list(item["choices"])
    random.shuffle(shuffled)

    instruction = build_instruction(item["question"], shuffled)
    label = build_label(item["answer"], shuffled)
    audio = item.get("audio_id") or item.get("audio") or item.get("audio_path")

    result = dict(item)
    result["instruction"] = instruction
    result["label"] = label

    if not audio:
        result["response"] = f"[error] missing audio path"
        return result

    last_exc = None
    for attempt in range(args.retries + 1):
        try:
            response = call_lalm(
                args.endpoint, args.model, instruction, audio,
                args.max_tokens, args.timeout,
            )
            result["response"] = response
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt < args.retries:
                time.sleep(2 ** attempt)
    if last_exc is not None:
        result["response"] = f"[error] {type(last_exc).__name__}: {last_exc}"

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.output is None:
        model_name = args.model.rstrip("/").split("/")[-1]
        args.output = f"{model_name}.jsonl"

    items = load_manifest(args.manifest)
    print(f"[info] loaded {len(items)} items from {args.manifest}")

    lock = threading.Lock()
    done_count = 0
    total = len(items)
    results = [None] * total

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        future_to_idx = {
            pool.submit(run_item, args, item): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            with lock:
                done_count += 1
                print(f"[{done_count}/{total}] done")

    with open(args.output, "w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[info] saved {len(results)} items to {args.output}")


if __name__ == "__main__":
    main()
