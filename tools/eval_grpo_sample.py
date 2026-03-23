import argparse
import json
import os
import random
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(text):
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)


def extract_answer(text):
    if text is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text).strip()


def char_f1(pred, gold):
    if not pred or not gold:
        return 0.0
    pred_counter = Counter(pred)
    gold_counter = Counter(gold)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def accuracy_score(pred_text, gold_text):
    predicted = normalize_text(extract_answer(pred_text))
    gold = normalize_text(gold_text)
    if not predicted or not gold:
        return 0.0
    if predicted == gold:
        return 1.0
    contain_score = 1.0 if len(gold) >= 2 and gold in predicted else 0.0
    reverse_contain_score = 0.8 if len(predicted) >= 2 and predicted in gold else 0.0
    f1_score = char_f1(predicted, gold)
    return min(max(max(f1_score, contain_score, reverse_contain_score), 0.0), 1.0)


def format_score(pred_text):
    return 1.0 if re.match(r"<think>.*?</think><answer>.*?</answer>$", pred_text or "") else 0.0


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return question


@torch.inference_mode()
def generate_for_questions(model_path, questions, max_new_tokens, temperature):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    outputs = []
    for q in questions:
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            repetition_penalty=1.0,
        )
        gen_tokens = generated[0][inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def summarize(scores):
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="/root/medical/grpo_tcm/valid.jsonl")
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--grpo_model", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="outputs-grpo-eval")
    args = parser.parse_args()

    rows = load_jsonl(args.data_file)
    if len(rows) < args.sample_size:
        raise ValueError(f"样本不足: 需要 {args.sample_size} 条, 实际 {len(rows)} 条")
    random.seed(args.seed)
    sampled_rows = random.sample(rows, args.sample_size)
    questions = [r["question"] for r in sampled_rows]
    answers = [r["answer"] for r in sampled_rows]

    sft_outputs = generate_for_questions(args.sft_model, questions, args.max_new_tokens, args.temperature)
    grpo_outputs = generate_for_questions(args.grpo_model, questions, args.max_new_tokens, args.temperature)

    sft_acc_scores = [accuracy_score(pred, gold) for pred, gold in zip(sft_outputs, answers)]
    grpo_acc_scores = [accuracy_score(pred, gold) for pred, gold in zip(grpo_outputs, answers)]
    sft_fmt_scores = [format_score(pred) for pred in sft_outputs]
    grpo_fmt_scores = [format_score(pred) for pred in grpo_outputs]

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "sample_eval_details.jsonl")
    summary_path = os.path.join(args.output_dir, "sample_eval_summary.json")

    with open(detail_path, "w", encoding="utf-8") as f:
        for i, (q, gold, sft_pred, grpo_pred) in enumerate(zip(questions, answers, sft_outputs, grpo_outputs)):
            row = {
                "idx": i,
                "question": q,
                "gold_answer": gold,
                "sft_pred": sft_pred,
                "grpo_pred": grpo_pred,
                "sft_accuracy": sft_acc_scores[i],
                "grpo_accuracy": grpo_acc_scores[i],
                "sft_format": sft_fmt_scores[i],
                "grpo_format": grpo_fmt_scores[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_size": args.sample_size,
        "sft_accuracy_avg": summarize(sft_acc_scores),
        "grpo_accuracy_avg": summarize(grpo_acc_scores),
        "accuracy_delta": round(summarize(grpo_acc_scores) - summarize(sft_acc_scores), 6),
        "sft_format_avg": summarize(sft_fmt_scores),
        "grpo_format_avg": summarize(grpo_fmt_scores),
        "format_delta": round(summarize(grpo_fmt_scores) - summarize(sft_fmt_scores), 6),
        "detail_file": detail_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
