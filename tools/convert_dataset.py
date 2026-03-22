import argparse
import os

from datasets import DatasetDict, load_dataset


def _first_existing(example, keys, default=""):
    for key in keys:
        value = example.get(key, None)
        if value is not None:
            return value
    return default


def process_qa(examples):
    convs = []
    for idx in range(len(next(iter(examples.values())))):
        q = _first_existing(
            {k: examples[k][idx] for k in examples},
            ["input", "question", "instruction", "prompt", "query"],
            "",
        )
        a = _first_existing(
            {k: examples[k][idx] for k in examples},
            ["output", "answer", "response", "target", "completion"],
            "",
        )
        if not q or not a:
            continue
        convs.append([
            {"from": "human", "value": str(q)},
            {"from": "gpt", "value": str(a)},
        ])
    return {"conversations": convs}


def process_alpaca(examples):
    convs = []
    for idx in range(len(next(iter(examples.values())))):
        row = {k: examples[k][idx] for k in examples}
        instruction = _first_existing(row, ["instruction", "question", "prompt"], "")
        inp = _first_existing(row, ["input", "context"], "")
        output = _first_existing(row, ["output", "answer", "response", "target", "completion"], "")
        if inp and len(str(inp).strip()) > 0:
            instruction = f"{instruction}\n\n{inp}"
        if not instruction or not output:
            continue
        convs.append([
            {"from": "human", "value": str(instruction)},
            {"from": "gpt", "value": str(output)},
        ])
    return {"conversations": convs}


def process_sharegpt(examples):
    if "conversations" in examples:
        return {"conversations": examples["conversations"]}
    if "items" in examples:
        return {"conversations": examples["items"]}
    if "messages" in examples:
        conversations = []
        for msgs in examples["messages"]:
            one = []
            for m in msgs:
                role = m.get("role", "")
                value = m.get("content", "")
                if role in ["user", "human"]:
                    one.append({"from": "human", "value": value})
                elif role in ["assistant", "gpt"]:
                    one.append({"from": "gpt", "value": value})
            conversations.append(one)
        return {"conversations": conversations}
    return process_qa(examples)


def detect_data_type(column_names, expected_data_type):
    if expected_data_type != "auto":
        return expected_data_type
    if "conversations" in column_names or "items" in column_names or "messages" in column_names:
        return "sharegpt"
    if "instruction" in column_names and "output" in column_names:
        return "alpaca"
    qa_input = any(c in column_names for c in ["input", "question", "prompt", "query"])
    qa_output = any(c in column_names for c in ["output", "answer", "response", "target", "completion"])
    if qa_input and qa_output:
        return "qa"
    return "qa"


def convert_split(ds, data_type):
    if data_type == "alpaca":
        return ds.map(process_alpaca, batched=True, remove_columns=ds.column_names, desc="Running process")
    if data_type == "qa":
        return ds.map(process_qa, batched=True, remove_columns=ds.column_names, desc="Running process")
    converted = ds.map(process_sharegpt, batched=True, remove_columns=ds.column_names, desc="Running process")
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default=None, help="input file name, csv/json/jsonl")
    parser.add_argument("--out_file", type=str, default=None, help="single output file, e.g. out.jsonl")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir for train/valid jsonl")
    parser.add_argument("--data_type", type=str, default="auto", help="auto, alpaca, qa, or sharegpt")
    parser.add_argument("--file_type", type=str, default="json", help="input file type: json/jsonl/csv")
    parser.add_argument("--dataset_name", type=str, default=None, help="hf dataset name")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="hf dataset config")
    parser.add_argument("--validation_split", type=str, default="validation", help="validation split name")
    parser.add_argument("--validation_ratio", type=float, default=0.01, help="split ratio when no validation split")
    args = parser.parse_args()
    print(args)

    if args.dataset_name:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        if not args.in_file:
            raise ValueError("Either --dataset_name or --in_file must be provided")
        data_files = {"train": args.in_file}
        if args.file_type == "csv":
            raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")
        elif args.file_type in ["json", "jsonl"]:
            raw_datasets = load_dataset("json", data_files=data_files)
        else:
            raise ValueError("File type not supported")

    if not isinstance(raw_datasets, DatasetDict):
        raw_datasets = DatasetDict({"train": raw_datasets})

    processed = {}
    for split_name, split_ds in raw_datasets.items():
        data_type = detect_data_type(split_ds.column_names, args.data_type)
        processed[split_name] = convert_split(split_ds, data_type)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        train_split_name = "train" if "train" in processed else list(processed.keys())[0]
        valid_split_name = args.validation_split if args.validation_split in processed else None
        train_ds = processed[train_split_name]
        if valid_split_name is None:
            split = train_ds.train_test_split(test_size=args.validation_ratio, seed=42)
            train_ds = split["train"]
            valid_ds = split["test"]
        else:
            valid_ds = processed[valid_split_name]
        train_ds.to_json(os.path.join(args.output_dir, "train.jsonl"), lines=True, force_ascii=False)
        valid_ds.to_json(os.path.join(args.output_dir, "valid.jsonl"), lines=True, force_ascii=False)
        return

    if not args.out_file:
        raise ValueError("When --output_dir is not provided, --out_file is required")
    train_split_name = "train" if "train" in processed else list(processed.keys())[0]
    processed[train_split_name].to_json(args.out_file, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
