import argparse
import os

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


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


def detect_bucket(row):
    instruction = first_existing(row, ["instruction", "question", "prompt", "query"]).lower()
    user_input = first_existing(row, ["input", "context"]).lower()
    answer = first_existing(row, ["output", "answer", "response", "target", "completion"]).lower()
    meta = " ".join(
        [
            str(row.get("source", "")),
            str(row.get("subset", "")),
            str(row.get("dataset", "")),
            str(row.get("category", "")),
            str(row.get("file_name", "")),
            str(row.get("id", "")),
        ]
    ).lower()
    text = f"{meta}\n{instruction}\n{user_input}\n{answer}"
    if "nlpdiseasediagnosed" in text or ("疾病诊断" in instruction and "证型" not in instruction):
        return "disease"
    if "nlpsyndromediagnosed" in text or "证型诊断" in instruction:
        return "syndrome"
    if "structprescription" in text or "方剂中药组成" in instruction:
        return "prescription"
    if "medicalknowledge_source2" in text or "术语" in instruction or "名词解释" in instruction:
        return "knowledge2"
    return "other"


def convert_split_with_bucket(split_ds):
    def to_grpo_with_bucket(row):
        qa = to_grpo_example(row)
        return {"question": qa["question"], "answer": qa["answer"], "bucket": detect_bucket(row)}

    converted = split_ds.map(
        to_grpo_with_bucket,
        remove_columns=split_ds.column_names,
    )
    converted = converted.filter(lambda x: bool(x["question"].strip()) and bool(x["answer"].strip()))
    return converted


def resolve_config_name(all_configs, source_key):
    key = source_key.lower()
    candidates = [cfg for cfg in all_configs if key in cfg.lower()]
    if candidates:
        return candidates[0]
    prefixed_candidates = [cfg for cfg in all_configs if f"sft_{key}" in cfg.lower()]
    if prefixed_candidates:
        return prefixed_candidates[0]
    raise ValueError(f"Cannot find config for source_key={source_key}. available_configs={all_configs}")


def load_train_split(dataset_name, config_name):
    ds = load_dataset(dataset_name, config_name)
    if "train" in ds:
        return ds["train"]
    first_split_name = list(ds.keys())[0]
    return ds[first_split_name]


def load_train_split_from_repo_file(dataset_name, filename):
    local_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=filename)
    ds = load_dataset("json", data_files=local_path)
    if "train" in ds:
        return ds["train"]
    first_split_name = list(ds.keys())[0]
    return ds[first_split_name]


def sample_exact(ds, sample_size, seed):
    if len(ds) < sample_size:
        raise ValueError(f"Not enough samples in dataset. required={sample_size}, got={len(ds)}")
    return ds.shuffle(seed=seed).select(range(sample_size))


def sample_bucket(ds, bucket, train_size, valid_size, seed):
    filtered = ds.filter(lambda x: x["bucket"] == bucket)
    sampled = sample_exact(filtered, train_size + valid_size, seed=seed)
    train_part = sampled.select(range(train_size)).remove_columns(["bucket"])
    valid_part = sampled.select(range(train_size, train_size + valid_size)).remove_columns(["bucket"])
    return train_part, valid_part, len(filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SylvanL/Traditional-Chinese-Medicine-Dataset-SFT")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_disease", type=int, default=20000)
    parser.add_argument("--train_syndrome", type=int, default=20000)
    parser.add_argument("--train_prescription", type=int, default=6000)
    parser.add_argument("--train_knowledge2", type=int, default=4000)
    parser.add_argument("--valid_disease", type=int, default=400)
    parser.add_argument("--valid_syndrome", type=int, default=400)
    parser.add_argument("--valid_prescription", type=int, default=120)
    parser.add_argument("--valid_knowledge2", type=int, default=80)
    args = parser.parse_args()

    source_plan = [
        ("nlpDiseaseDiagnosed", "SFT_nlpDiseaseDiagnosed_61486.json", args.train_disease, args.valid_disease),
        ("nlpSyndromeDiagnosed", "SFT_nlpSyndromeDiagnosed_48665.json", args.train_syndrome, args.valid_syndrome),
        ("structPrescription", "SFT_structPrescription_92896.json", args.train_prescription, args.valid_prescription),
        ("medicalKnowledge_source2", "SFT_medicalKnowledge_source2_99334.json", args.train_knowledge2, args.valid_knowledge2),
    ]
    all_configs = get_dataset_config_names(args.dataset_name) or []

    train_parts = []
    valid_parts = []
    if len(all_configs) <= 1:
        for idx, (source_key, source_file, train_size, valid_size) in enumerate(source_plan):
            try:
                source_ds = load_train_split_from_repo_file(args.dataset_name, source_file)
                converted_source_ds = convert_split(source_ds)
                source_sampled = sample_exact(converted_source_ds, train_size + valid_size, seed=args.seed + idx)
                train_parts.append(source_sampled.select(range(train_size)))
                valid_parts.append(source_sampled.select(range(train_size, train_size + valid_size)))
                print(
                    f"source={source_key} file={source_file} total={len(converted_source_ds)} "
                    f"train={train_size} valid={valid_size}"
                )
            except Exception as e:
                print(f"file mode failed for {source_key} ({source_file}), fallback to bucket mode. error={e}")
                single_config = all_configs[0] if all_configs else None
                source_ds = load_train_split(args.dataset_name, single_config)
                converted_ds = convert_split_with_bucket(source_ds)
                bucket_mapping = {
                    "nlpDiseaseDiagnosed": "disease",
                    "nlpSyndromeDiagnosed": "syndrome",
                    "structPrescription": "prescription",
                    "medicalKnowledge_source2": "knowledge2",
                }
                bucket = bucket_mapping[source_key]
                train_part, valid_part, bucket_total = sample_bucket(
                    converted_ds, bucket, train_size, valid_size, seed=args.seed + idx
                )
                train_parts.append(train_part)
                valid_parts.append(valid_part)
                print(
                    f"source={source_key} bucket={bucket} total={bucket_total} "
                    f"train={train_size} valid={valid_size}"
                )
    else:
        for idx, (source_key, _source_file, train_size, valid_size) in enumerate(source_plan):
            config_name = resolve_config_name(all_configs, source_key)
            source_ds = load_train_split(args.dataset_name, config_name)
            converted_source_ds = convert_split(source_ds)
            source_sampled = sample_exact(converted_source_ds, train_size + valid_size, seed=args.seed + idx)
            train_parts.append(source_sampled.select(range(train_size)))
            valid_parts.append(source_sampled.select(range(train_size, train_size + valid_size)))
            print(
                f"source={source_key} config={config_name} total={len(converted_source_ds)} "
                f"train={train_size} valid={valid_size}"
            )

    train_ds = concatenate_datasets(train_parts).shuffle(seed=args.seed)
    valid_ds = concatenate_datasets(valid_parts).shuffle(seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds.to_json(os.path.join(args.output_dir, "train.jsonl"), lines=True, force_ascii=False)
    valid_ds.to_json(os.path.join(args.output_dir, "valid.jsonl"), lines=True, force_ascii=False)

    print(f"train samples: {len(train_ds)}")
    print(f"valid samples: {len(valid_ds)}")
    print(f"saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
