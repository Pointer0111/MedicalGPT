import json
import os
from glob import glob

def convert_file(input_file, output_file):
    print(f"Converting {input_file} -> {output_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        # 尝试读取，可能是 list 也可能是 jsonl
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    new_data = []
    for item in data:
        # 补充缺失字段
        new_item = {
            "system": "你是一名专业的医生，请根据患者的主诉和病史进行诊断并给出治疗建议。", # 给一个默认的 system prompt
            "history": [], # 默认为空历史
            "question": item.get("question", ""),
            "response_chosen": item.get("response_chosen", ""),
            "response_rejected": item.get("response_rejected", "")
        }
        new_data.append(new_item)

    # 保存为 JSON List
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

def main():
    input_dir = "/root/medical/reward"
    output_dir = "/root/medical/reward_dpo_ready"
    os.makedirs(output_dir, exist_ok=True)

    files = glob(os.path.join(input_dir, "*.json"))
    for file in files:
        filename = os.path.basename(file)
        output_file = os.path.join(output_dir, filename)
        convert_file(file, output_file)

if __name__ == "__main__":
    main()
