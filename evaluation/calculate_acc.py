import argparse
import json

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help="Input JSONL file path")
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

if __name__ == "__main__":
    args = get_args_parser()
    items = read_jsonl(args.input)

    correct_count = 0
    incorrect_count = 0
    total_count = 0
    for item in items:
        judgement = item.get("Judgement", "").lower().strip()
        if judgement == "correct":
            correct_count += 1
        elif judgement == "incorrect":
            incorrect_count += 1
        else:
            print(f"Invalid judgement for id: {item.get('id', '?')}")
        total_count += 1

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")
    print(f"Total count: {total_count}")
    if total_count > 0:
        print(f"Accuracy: {(correct_count / total_count)*100:.2f}%")
