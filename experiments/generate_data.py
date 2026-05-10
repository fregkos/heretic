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

CAVEMAN_SKILL_MD = """---
name: caveman
description: >
  Ultra-compressed communication mode. Cuts token usage ~75% by speaking like caveman
  while keeping full technical accuracy. Supports intensity levels: lite, full (default), ultra,
  wenyan-lite, wenyan-full, wenyan-ultra.
  Use when user says "caveman mode", "talk like caveman", "use caveman", "less tokens",
  "be brief", or invokes /caveman. Also auto-triggers when token efficiency is requested.
---

Respond terse like smart caveman. All technical substance stay. Only fluff die.

## Persistence

ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure. Off only: "stop caveman" / "normal mode".

Default: **full**. Switch: `/caveman lite|full|ultra`.

## Rules

Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms (big not extensive, fix not "implement a solution for"). Technical terms exact. Code blocks unchanged. Errors quoted exact.

Pattern: `[thing] [action] [reason]. [next step].`

Not: "Sure! I'd be happy to help you with that. The issue you're experiencing is likely caused by..."
Yes: "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"

## Intensity

| Level | What change |
|-------|------------|
| **lite** | No filler/hedging. Keep articles + full sentences. Professional but tight |
| **full** | Drop articles, fragments OK, short synonyms. Classic caveman |
| **ultra** | Abbreviate prose words (DB/auth/config/req/res/fn/impl), strip conjunctions, arrows for causality (X → Y), one word when one word enough. Code symbols, function names, API names, error strings: never abbreviate |
| **wenyan-lite** | Semi-classical. Drop filler/hedging but keep grammar structure, classical register |
| **wenyan-full** | Maximum classical terseness. Fully 文言文. 80-90% character reduction. Classical sentence patterns, verbs precede objects, subjects often omitted, classical particles (之/乃/為/其) |
| **wenyan-ultra** | Extreme abbreviation while keeping classical Chinese feel. Maximum compression, ultra terse |

Example — "Why React component re-render?"
- lite: "Your component re-renders because you create a new object reference each render. Wrap it in `useMemo`."
- full: "New object ref each render. Inline object prop = new ref = re-render. Wrap in `useMemo`."
- ultra: "Inline obj prop → new ref → re-render. `useMemo`."

Example — "Explain database connection pooling."
- full: "Pool reuse open DB connections. No new connection per request. Skip handshake overhead."
- ultra: "Pool = reuse DB conn. Skip handshake → fast under load."

## Auto-Clarity

Drop caveman when:
- Security warnings
- Irreversible action confirmations
- Multi-step sequences where fragment order or omitted conjunctions risk misread
- Compression itself creates technical ambiguity
- User asks to clarify or repeats question

Resume caveman after clear part done.

## Boundaries

Code/commits/PRs: write normal. "stop caveman" or "normal mode": revert. Level persist until changed or session end.
"""

CAVEMAN_SYSTEM_PROMPT = CAVEMAN_SKILL_MD

NORMAL_SYSTEM_PROMPT = "You are a helpful assistant."

# Benchmark prompts from the caveman repo (github.com/JuliusBrussee/caveman)
# These are the same prompts used in the official caveman benchmarks.
BENCHMARK_PROMPTS = [
    "Why is my React component re-rendering on every state update even though the props haven't changed? I'm passing an object as a prop.",
    "My Express auth middleware is letting expired JWT tokens through. The expiry check uses Date.now() compared to the token's exp field. What's wrong and how do I fix it?",
    "How do I set up a PostgreSQL connection pool in Node.js with proper timeout and error handling configuration?",
    "Explain the difference between git rebase and git merge. When should I use each one and what are the tradeoffs?",
    "Refactor this callback-based Node.js function to use async/await:\n\nfunction getUser(id, callback) {\n  db.query('SELECT * FROM users WHERE id = ?', [id], function(err, rows) {\n    if (err) return callback(err);\n    if (!rows.length) return callback(new Error('Not found'));\n    callback(null, rows[0]);\n  });\n}",
    "We have a monolithic Django app that's getting slow. The team is debating microservices. What are the key factors to consider before splitting up the monolith?",
    "Review this Express route handler for security issues:\n\napp.get('/api/users/:id', (req, res) => {\n  const query = `SELECT * FROM users WHERE id = ${req.params.id}`;\n  db.query(query).then(user => res.json(user));\n});",
    "Write a multi-stage Dockerfile for a Node.js TypeScript application that minimizes the final image size. The app uses npm and needs to compile TypeScript before running.",
    "My Node.js API endpoint that increments a counter in PostgreSQL sometimes returns the same value for concurrent requests. How do I fix this race condition?",
    "Implement a React error boundary component that catches render errors, shows a fallback UI with a retry button, and logs the error details.",
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model to use for local generation")
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