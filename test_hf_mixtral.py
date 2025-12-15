#!/usr/bin/env python3
"""Test original HuggingFace Mixtral to verify expected output."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def main():
    model_path = "/home/k/models/mixtral-8x7b-instruct"
    device = "cuda"

    print("Loading HuggingFace Mixtral...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load with device_map="auto" to offload to CPU/disk if needed
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Disable flash attention
    )

    print("Model loaded!")

    prompt = "[INST] What is the capital of France? Answer in one sentence. [/INST]"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

    print(f"\nResponse: {response}")
    print(f"\nGenerated {new_tokens} tokens in {elapsed:.1f}s ({new_tokens/elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
