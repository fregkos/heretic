"""
Generate paired (caveman, normal) response datasets for direction extraction.

Protocol:
1. Load prompt sets (Alpaca subset + benchmark prompts)
2. For each prompt, generate two responses from the SAME model:
   - system_prompt = caveman SKILL.md content → caveman response
   - system_prompt = "You are a helpful assistant." → normal response
3. Save paired datasets for residual extraction

Usage:
    python generate_data.py --model Qwen/Qwen3-4B-Instruct --output-dir ./data --num-prompts 300
    python generate_data.py --model Qwen/Qwen3-4B-Instruct --output-dir ./data --use-api --api-model claude-3-5-sonnet-20241022
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CAVEMAN_SYSTEM_PROMPT = """You are a terse assistant. Rules:

Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms (big not extensive, fix not "implement a solution for"). Technical terms exact. Code blocks unchanged. Errors quoted exact.

Pattern: [thing] [action] [reason]. [next step].

Not: "Sure! I'd be happy to help you with that. The issue you're experiencing is likely caused by..."
Yes: "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"

Respond in this terse style for ALL user queries."""

NORMAL_SYSTEM_PROMPT = "You are a helpful assistant."

BENCHMARK_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is the purpose of a transformer attention mechanism?",
    "Write a Python function that reverses a linked list.",
    "Explain how gradient descent works.",
    "What are the trade-offs between bias and variance?",
    "Describe the role of activation functions in neural networks.",
    "Implement binary search in Python.",
    "What is the difference between L1 and L2 regularization?",
    "Explain the concept of embedding in natural language processing.",
    "How does batch normalization help training?",
]


def load_alpaca_prompts(num_prompts: int, seed: int = 42) -> list[str]:
    dataset = load_dataset("mlabonne/harmless_alpaca", split=f"train[:{num_prompts}]")
    prompts = list(dataset["text"])
    return prompts


def generate_local(
    model_name: str,
    prompts: list[str],
    system_prompt: str,
    max_new_tokens: int = 200,
    quantization: str = "bnb_4bit",
    batch_size: int = 4,
) -> list[str]:
    print(f"Loading model {model_name}...")

    if quantization == "bnb_4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        chats = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch_prompts
        ]

        chat_prompts = tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=False
        )

        inputs = tokenizer(
            chat_prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            responses.append(response)

        print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} responses")

    return responses


def generate_via_api(
    prompts: list[str],
    system_prompt: str,
    api_model: str,
    api_key: str | None = None,
    max_tokens: int = 200,
) -> list[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    responses = []

    for i, prompt in enumerate(prompts):
        message = client.messages.create(
            model=api_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        responses.append(message.content[0].text)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} responses")

    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate paired caveman/normal datasets")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model to use for local generation")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--num-prompts", type=int, default=300, help="Number of prompts to use")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens per response")
    parser.add_argument("--quantization", default="bnb_4bit", choices=["bnb_4bit", "none"])
    parser.add_argument("--use-api", action="store_true", help="Use API instead of local model")
    parser.add_argument("--api-model", default="claude-3-5-sonnet-20241022")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prompts...")
    prompts = load_alpaca_prompts(args.num_prompts, seed=args.seed)

    all_prompts = prompts + BENCHMARK_PROMPTS
    print(f"Total prompts: {len(all_prompts)}")

    if args.use_api:
        print(f"Generating caveman responses via API ({args.api_model})...")
        caveman_responses = generate_via_api(
            all_prompts, CAVEMAN_SYSTEM_PROMPT, args.api_model, max_tokens=args.max_new_tokens
        )

        print(f"Generating normal responses via API ({args.api_model})...")
        normal_responses = generate_via_api(
            all_prompts, NORMAL_SYSTEM_PROMPT, args.api_model, max_tokens=args.max_new_tokens
        )
    else:
        print(f"Generating caveman responses locally ({args.model})...")
        caveman_responses = generate_local(
            args.model, all_prompts, CAVEMAN_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens, quantization=args.quantization,
        )

        print(f"Generating normal responses locally ({args.model})...")
        normal_responses = generate_local(
            args.model, all_prompts, NORMAL_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens, quantization=args.quantization,
        )

    paired_data = []
    for prompt, caveman_resp, normal_resp in zip(all_prompts, caveman_responses, normal_responses):
        paired_data.append({
            "prompt": prompt,
            "caveman_response": caveman_resp,
            "normal_response": normal_resp,
        })

    output_file = output_dir / "paired_responses.json"
    with open(output_file, "w") as f:
        json.dump(paired_data, f, indent=2)
    print(f"Saved {len(paired_data)} paired responses to {output_file}")

    token_stats = {
        "caveman_avg_tokens": sum(len(r.split()) for r in caveman_responses) / len(caveman_responses),
        "normal_avg_tokens": sum(len(r.split()) for r in normal_responses) / len(normal_responses),
    }
    print(f"Caveman avg word count: {token_stats['caveman_avg_tokens']:.1f}")
    print(f"Normal avg word count: {token_stats['normal_avg_tokens']:.1f}")

    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(token_stats, f, indent=2)


if __name__ == "__main__":
    main()