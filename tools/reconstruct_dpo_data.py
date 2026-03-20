import json
import os
import time
from tqdm import tqdm
from dashscope import Generation
import dashscope

# 配置 API Key
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

def call_qwen(messages, model="qwen-plus"):
    try:
        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=model,
            messages=messages,
            result_format="message",
            enable_search=False, 
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"API Error: {response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def reconstruct_prompt(original_question):
    """利用大模型重构 Prompt 为标准病历格式"""
    system_prompt = """你是一名专业的医疗信息整理员。你的任务是将用户的输入整理为结构化的医学问题描述。

请根据用户输入的内容类型，选择以下两种格式之一输出：

**情况A：用户描述了自己的症状、病史（病例咨询）**
请整理为：
主诉：[主要症状及持续时间]
现病史：[详细症状描述、起因、既往检查等]
核心问题：[患者最想解决的问题]

**情况B：用户询问医学知识、疾病定义（知识科普）**
请整理为：
咨询主题：[疾病/药物/检查名称]
具体问题：[用户想了解的具体知识点，如病因、治疗、副作用等]

⚠️ 严格遵守：
1. **绝对不要回答**问题。
2. **只提取事实**，不要编造。
3. 保持专业、简洁。

示例1（病例）：
输入：医生，我头疼三天了，是不是感冒了？
输出：
主诉：头痛3天。
现病史：患者自诉头痛持续3天，怀疑感冒。
核心问题：鉴别诊断及治疗建议。

示例2（知识）：
输入：暴盲的病因病机是什么？
输出：
咨询主题：暴盲。
具体问题：中医病因病机分析。
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户输入：{original_question}"}
    ]
    return call_qwen(messages)

def process_data(input_file, output_file, max_samples=10):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
    
    new_data = []
    
    print(f"开始处理数据，总数: {len(data)}, 目标处理数: {max_samples}")
    
    for item in tqdm(data[:max_samples]): 
        original_q = item.get('question', '')
        original_chosen = item.get('response_chosen', '')
        original_rejected = item.get('response_rejected', '')
        
        if not original_q or not original_chosen or not original_rejected:
            continue

        # 1. 重构 Prompt
        new_q = reconstruct_prompt(original_q)
        if not new_q: 
            print(f"重构失败，跳过: {original_q[:20]}...")
            continue
        
        time.sleep(0.5)
            
        new_data.append({
            "system": "你是一名专业的医生。请根据患者的描述，进行病理分析，并给出专业的诊断和治疗建议。",
            "question": new_q,
            "response_chosen": original_chosen,
            "response_rejected": original_rejected,
            "original_question": original_q 
        })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
    print(f"处理完成，共生成 {len(new_data)} 条重构 Prompt 的数据。")
    print(f"数据已保存至: {output_file}")

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置环境变量 DASHSCOPE_API_KEY")
    else:
        process_data(
            input_file="/root/medical/reward/train.json",
            output_file="/root/medical/dpo_data/dpo_hq_train.json",
            max_samples=100 # 增加到 10 条，覆盖更多 case
        )
