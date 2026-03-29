import os
import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/root/autodl-fs/Qwen2.5-3B-TCM-SFT")
    parser.add_argument("--lora_model", type=str, default="/root/MedicalGPT/outputs-grpo-tcm-v2")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/Qwen2.5-3B-TCM-GRPO-v1")
    args = parser.parse_args()

    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.lora_model,
        trust_remote_code=True
    )

    print(f"Loading LoRA from: {args.lora_model} and merging...")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
