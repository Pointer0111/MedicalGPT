import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置路径
base_model_path = "/root/autodl-fs/qwen-2.5-3b-pt"  # 使用二次预训练后的模型作为基座
lora_model_path = "./outputs-sft-qwen-v1" # SFT 训练后的 LoRA 权重

# 测试用例 (医疗问诊对话)
test_prompts = [
    "医生你好，我最近经常头晕，有时候还恶心，这是怎么回事？",
    "孩子发烧39度，精神状态还可以，需要去医院吗？",
    "糖尿病患者可以吃西瓜吗？",
    "高血压药是饭前吃还是饭后吃好？",
    "孕妇感冒了能吃感冒药吗？"
]

def generate_chat(model, tokenizer, prompt):
    # 构造对话格式
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 只解码生成的回复部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

print(f"正在加载基座模型: {base_model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

results = []

print("\n=== Round 1: Base Model (PT Only) 测试 ===")
# 虽然 PT 模型没学过对话，但我们可以看看它是否能强行回答，或者只是续写
for prompt in test_prompts:
    print(f"正在生成: {prompt[:10]}...")
    # 这里我们也尝试用 chat template，看看基座本身的能力
    response = generate_chat(model, tokenizer, prompt)
    results.append({"prompt": prompt, "base": response})

print("\n=== Round 2: SFT Model (PT + SFT) 测试 ===")
print(f"正在加载 LoRA 权重: {lora_model_path}")
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval() 

for i, prompt in enumerate(test_prompts):
    print(f"正在生成: {prompt[:10]}...")
    response = generate_chat(model, tokenizer, prompt)
    results[i]["sft"] = response

# Print Results
print("\n" + "="*80)
print("             MEDICAL CHATBOT COMPARISON REPORT")
print("="*80)
for res in results:
    print(f"\n[User]: {res['prompt']}")
    print(f"-"*40)
    print(f"[Base Model (PT Only)]:\n{res['base']}")
    print(f"-"*40)
    print(f"[SFT Model (PT + SFT)]:\n{res['sft']}")
    print("="*80)
