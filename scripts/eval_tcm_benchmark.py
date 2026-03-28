import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_prompt(question, options):
    prompt = f"请回答以下单项选择题，只输出选项字母。\n题目：{question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "答案："
    return prompt

def evaluate_model(model_path, data_path, output_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    print(f"Loading dataset from {data_path}...")
    data = load_data(data_path)
    
    results = {}
    total_correct = 0
    total_questions = 0
    
    for category, questions in data.items():
        print(f"Evaluating category: {category}")
        category_correct = 0
        category_results = []
        
        for item in tqdm(questions, desc=f"Processing {category}"):
            # 只处理单项选择题
            if item.get("question_type") != "单项选择题":
                continue
                
            question = item["question"]
            options = item["options"]
            gold_answer_idx = item["answer_idx"]
            
            prompt = build_prompt(question, options)
            messages = [{"role": "user", "content": prompt}]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=10,
                    temperature=0.1, # 低温，降低随机性
                    do_sample=False
                )
                
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 简单的答案提取逻辑：取第一个出现的大写字母 A-D
            pred_idx = None
            for char in response.upper():
                if char in options.keys():
                    pred_idx = char
                    break
            
            is_correct = (pred_idx == gold_answer_idx)
            if is_correct:
                category_correct += 1
                total_correct += 1
            total_questions += 1
            
            category_results.append({
                "question": question,
                "gold": gold_answer_idx,
                "pred": pred_idx,
                "raw_response": response.strip(),
                "is_correct": is_correct
            })
            
        acc = category_correct / len(category_results) if category_results else 0
        print(f"{category} Accuracy: {acc:.2%} ({category_correct}/{len(category_results)})\n")
        
        results[category] = {
            "accuracy": acc,
            "correct": category_correct,
            "total": len(category_results),
            "details": category_results
        }
    
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{total_questions})")
    
    results["overall"] = {
        "accuracy": overall_acc,
        "correct": total_correct,
        "total": total_questions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-fs/Qwen2.5-3B-TCM-SFT")
    parser.add_argument("--data_path", type=str, default="/root/MedicalGPT/TCM-Text-Exams.json")
    parser.add_argument("--output_path", type=str, default="/root/MedicalGPT/eval_results.json")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_path, args.output_path)
