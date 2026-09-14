# SAUVI

Evaluation pipeline for Large Audio Language Models (LALMs) on multiple-choice audio question answering.

## Project Structure

```
Benchmark/
├── run_lalm.py                 # Stage 1: query LALM, enrich manifest
├── toy_llm_server.py           # Mock LALM server for testing (port 8902)
├── toy_judge_server.py         # Mock judge server for testing (port 8903)
├── call.sh                     # Manual curl example
├── serve_instruct.sh           # Launch vLLM server (port 8901)
├── evaluation/
│   ├── llm_judge.py            # Stage 2: judge responses with LLM
│   ├── calculate_acc.py        # Stage 3: compute accuracy
│   └── README.md
├── utils/
│   └── common.py               # Shared JSONL loader
├── requirements.txt
└── README.md
```

## Prerequisites

```bash
pip install -r requirements.txt
```

## Pipeline Overview

```
[0] Start LALM server
         │
[1] run_lalm.py              ──> <model_name>.jsonl
         │
[2] evaluation/llm_judge.py  ──> <model_name>_judgements.jsonl
         │
[3] evaluation/calculate_acc.py ──> accuracy
```

## Step-by-Step Usage

### Stage 0: Start the LALM Server

For production (real model, 2x GPU):

```bash
bash serve_instruct.sh
# Serves Qwen3-Omni-30B on http://127.0.0.1:8901
```

For testing (no GPU needed):

```bash
python toy_llm_server.py --port 8902
```

### Stage 1: Run the LALM

Query the model for each item in a JSONL manifest. Builds an instruction string
(`question + (a) choice_a (b) choice_b ...` with shuffled option order), sends it
to the model via OpenAI-compatible API, and writes enriched output.

```bash
# With real model
python run_lalm.py \
  --manifest mmau-test-mini.jsonl \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --threads 8 \
  --output Qwen3-Omni-30B-A3B-Instruct.jsonl

# With toy server (for debugging)
python run_lalm.py \
  --manifest mmau-test-mini.jsonl \
  --endpoint http://127.0.0.1:8902/v1/chat/completions \
  --model toy-model \
  --threads 8
```

**Output format** (JSONL, one item per line):

```json
{
  "id": "...",
  "question": "Based on the given audio, identify the source of the speaking voice.",
  "choices": ["Man", "Woman", "Child", "Robot"],
  "answer": "Man",
  "audio_id": "./test-mini-audios/xxx.wav",
  "instruction": "Based on the given audio, identify the source of the speaking voice? (a) Robot (b) Woman (c) Man (d) Child",
  "label": "(c) Man",
  "response": "The answer is (c) Man."
}
```

### Stage 2: Judge Responses

Send each item to an LLM judge (GPT-4o or a local model) to determine if the
response is correct. Adds `Explanation` and `Judgement` keys to each item.

```bash
# With real OpenAI
python evaluation/llm_judge.py \
  -i Qwen3-Omni-30B-A3B-Instruct.jsonl \
  -o results/ \
  --api_key YOUR_OPENAI_API_KEY

# With toy judge server (for debugging)
python evaluation/llm_judge.py \
  -i Qwen3-Omni-30B-A3B-Instruct.jsonl \
  -o results/ \
  --api_key dummy \
  --base_url http://127.0.0.1:8903/v1 \
  --judge_model dummy
```

**Output format** (JSONL, same fields + Explanation/Judgement):

```json
{
  "instruction": "...",
  "response": "...",
  "label": "(c) Man",
  "Explanation": "The model chose (c) Man which matches the ground truth (c).",
  "Judgement": "correct"
}
```

If the judge fails to parse the response, `Explanation` and `Judgement` are set
to `null`. These items need manual verification.

### Stage 3: Calculate Accuracy

```bash
python evaluation/calculate_acc.py -i results/Qwen3-Omni-30B-A3B-Instruct_judgements.jsonl
```

Output:
```
Correct count: 750
Incorrect count: 230
Invalid judgement for id: ...
Total count: 1000
Accuracy: 76.53%
```

## Toy Servers for Debugging

Two mock servers let you test the full pipeline without a real model or OpenAI API key.

### Toy LLM Server (port 8902)

Parses the `(a) ... (b) ...` options from the prompt and picks one randomly.

```bash
python toy_llm_server.py --port 8902
```

### Toy Judge Server (port 8903)

Extracts the ground truth letter and model response letter, compares them, and
returns `correct`/`incorrect` with an explanation.

```bash
python toy_judge_server.py --port 8903
```

### Full Debug Pipeline

Open 3 terminals:

```bash
# Terminal 1: toy LLM
python toy_llm_server.py --port 8902

# Terminal 2: toy judge
python toy_judge_server.py --port 8903

# Terminal 3: run pipeline
python run_lalm.py \
  --manifest mmau-test-mini.jsonl \
  --endpoint http://127.0.0.1:8902/v1/chat/completions \
  --model toy-model

python evaluation/llm_judge.py \
  -i toy-model.jsonl \
  -o results/ \
  --api_key dummy \
  --base_url http://127.0.0.1:8903/v1

python evaluation/calculate_acc.py -i results/toy-model_judgements.jsonl
```

## Standalone Pipeline (run_lalm_benchmark.py)

Alternative script for manifests with **dict-choice** schema (e.g.
`regional_dialect_recognition.jsonl` where `choices` is `{"A": "Nam", "B": "Trung", ...}`).
Does inline answer parsing and accuracy computation without a separate judge.

```bash
python run_lalm_benchmark.py \
  --manifest regional_dialect_recognition.jsonl \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --threads 8
```

## CLI Reference

### run_lalm.py

| Flag | Default | Description |
|---|---|---|
| `--manifest` | `regional_dialect_recognition.jsonl` | Input JSONL manifest |
| `--endpoint` | `http://127.0.0.1:8901/v1/chat/completions` | LALM API endpoint |
| `--model` | Qwen3-Omni path | Model name sent in payload |
| `--threads` | `8` | Concurrent worker threads |
| `--max-tokens` | `512` | Max tokens in response |
| `--timeout` | `120` | HTTP timeout (seconds) |
| `--retries` | `2` | Retries with exponential backoff |
| `--output` | `<model_name>.jsonl` | Output JSONL path |

### evaluation/llm_judge.py

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | (required) | Input JSONL file |
| `--output_dir` / `-o` | (required) | Output directory |
| `--api_key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--judge_model` | `gpt-4o-2024-11-20` | Judge model name |
| `--base_url` | `None` (real OpenAI) | Base URL for OpenAI-compatible API |
| `--output_name` | `<stem>_judgements.jsonl` | Output filename |

### evaluation/calculate_acc.py

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | (required) | Input JSONL with `Judgement` field |

## Data Format

### Manifest Schema (MMAU)

JSONL, one object per line:

```json
{
  "id": "3fe64f3d-...",
  "audio_id": "./test-mini-audios/3fe64f3d-....wav",
  "question": "Based on the given audio, identify the source of the speaking voice.",
  "choices": ["Man", "Woman", "Child", "Robot"],
  "answer": "Man",
  "dataset": "AudioSet",
  "task": "sound",
  "split": "test-mini",
  "category": "Reasoning",
  "sub-category": "Acoustic Source Inference",
  "difficulty": "medium"
}
```

- `choices`: list of strings (option text)
- `answer`: string matching one of `choices`
- `audio_id`: path to audio file (resolved relative to CWD)

### Audio Paths

`run_lalm.py` converts audio paths to `file://` URLs. Absolute paths are used
as-is. Relative paths are resolved against the current working directory.
