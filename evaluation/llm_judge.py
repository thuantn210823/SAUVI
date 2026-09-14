import os
import re
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help="Input JSONL file path")
    parser.add_argument('--output_dir', '-o', type=str, required=True, help="Output directory")
    parser.add_argument('--api_key', type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument('--judge_model', type=str, default="gpt-4o-2024-11-20", help="Judge model name")
    parser.add_argument('--base_url', type=str, default=None, help="Base URL for OpenAI-compatible API (e.g. http://127.0.0.1:8903/v1)")
    parser.add_argument('--output_name', type=str, default=None, help="Output filename (default: <input_stem>_judgements.jsonl)")
    return parser.parse_args()

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {path}")

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def query_llm(client, user_prompt, model_name="gpt-4o-2024-11-20", temperature=0.0, top_p=1.0):

    system_prompt = '''You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response. 
                       You have to determine whether the model generated answer is correct.'''

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model=model_name,
        temperature=temperature,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content

def extract_judgement(text):
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        explanation = match.group(1)
        judgement = match.group(2)
    else:
        explanation = "No extracted explanation"
        judgement = "No extracted judgement"
    
    results = {"Explanation": explanation, "Judgement": judgement}
    return results

if __name__ == "__main__":
    from openai import OpenAI
    from tqdm import tqdm

    args = get_args_parser()
    check_and_create_folder(args.output_dir)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set OPENAI_API_KEY env var")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    model_name = args.judge_model
    temperature = 0.0
    top_p = 0.9
    max_tokens = 1024

    user_prompt_template = '''
    You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
    1. Since there is one and only one corect answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
    2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
    3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

    Considering the following examples:
    Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
    Ground truth answer: (a) Paris
    If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
    If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
    If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
    If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
    Ground truth answer: (a) Happy
    If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
    If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".
    
    Now here is the question and the model generated response for you to judge:
    Question: [QUESTION]
    Ground truth answer: [GROUND_TRUTH_ANSWER]
    Model generated response: [MODEL_GENERATED_RESPONSE]

    Carefully make your decision based on the above criteria. Return your judgement with the following format:
    Explanation: <Your explanation on your judgement>
    Judgement: <Your judgement, either "correct" or "incorrect">
    '''

    items = read_jsonl(args.input)

    for item in tqdm(items, desc="Judging"):
        question = item["instruction"]
        response = item["response"]
        ground_truth_answer = item["label"]
        user_prompt = user_prompt_template.replace("[QUESTION]", question).replace("[MODEL_GENERATED_RESPONSE]", response).replace("[GROUND_TRUTH_ANSWER]", ground_truth_answer)

        judgement = query_llm(client, user_prompt, model_name=model_name, temperature=temperature, top_p=top_p)
        results = extract_judgement(judgement)

        item["Explanation"] = results["Explanation"]
        item["Judgement"] = results["Judgement"]

    if args.output_name is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output_name = f"{stem}_judgements.jsonl"

    write_jsonl(items, os.path.join(args.output_dir, args.output_name))
