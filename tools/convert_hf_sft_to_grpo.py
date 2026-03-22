import argparse
import os

from datasets import DatasetDict, load_dataset


def first_existing(row, keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def to_grpo_example(row):
    instruction = first_existing(row, ["instruction", "question", "prompt", "query"])
    user_input = first_existing(row, ["input", "context"])
    answer = first_existing(row, ["output", "answer", "response", "target", "completion"])
    question = instruction
    if user_input:
        question = f"{instruction}\n\n{user_input}" if instruction else user_input
    return {"question": question, "answer": answer}


def convert_split(split_ds):
    converted = split_ds.map(
        to_grpo_example,
        remove_columns=split_ds.column_names,
    )
    converted = converted.filter(lambda x: bool(x["question"].strip()) and bool(x["answer"].strip()))
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SylvanL/Traditional-Chinese-Medicine-Dataset-SFT")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--validation_ratio", type=float, default=0.01)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    args = parser.parse_args()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if not isinstance(raw_datasets, DatasetDict):
        raw_datasets = DatasetDict({"train": raw_datasets})

    converted = {}
    for split_name, ds in raw_datasets.items():
        converted[split_name] = convert_split(ds)

    train_split_name = "train" if "train" in converted else list(converted.keys())[0]
    train_ds = converted[train_split_name]
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    if args.validation_split in converted:
        valid_ds = converted[args.validation_split]
    else:
        split = train_ds.train_test_split(test_size=args.validation_ratio, seed=42)
        train_ds = split["train"]
        valid_ds = split["test"]

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds.to_json(os.path.join(args.output_dir, "train.jsonl"), lines=True, force_ascii=False)
    valid_ds.to_json(os.path.join(args.output_dir, "valid.jsonl"), lines=True, force_ascii=False)

    print(f"train samples: {len(train_ds)}")
    print(f"valid samples: {len(valid_ds)}")
    print(f"saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
