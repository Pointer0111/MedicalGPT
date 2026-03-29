import json

def check_data_quality(file_path):
    print(f"Checking data quality for: {file_path}")
    
    total_samples = 0
    suppress_keywords = [
        "直接给出",
        "无需给出原因"
    ]
    
    issues_found = 0
    sample_preview = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)
            question = data.get("question", "")
            
            # 检查是否包含抑制性词汇
            has_issue = any(kw in question for kw in suppress_keywords)
            if has_issue:
                issues_found += 1
            
            if total_samples <= 3:
                sample_preview.append(data)

    print(f"\nTotal samples: {total_samples}")
    print(f"Samples with suppressing keywords: {issues_found} ({(issues_found/total_samples)*100:.2f}%)")
    
    print("\n--- Preview of first 3 samples ---")
    for i, sample in enumerate(sample_preview, 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample['question'][:200]}...") # 只打印前200个字符
        print(f"Answer: {sample['answer']}")

if __name__ == "__main__":
    check_data_quality("/root/medical/grpo_tcm/train.jsonl")
