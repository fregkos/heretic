"""
Evaluation suite for directionally amplified models.

Compares four arms:
1. Baseline: Original model, no modification
2. Prompt-based: Original model + caveman skill in system prompt
3. Amplified: Direction-amplified model, no skill prompt
4. Finetuned (optional): LoRA finetuned on caveman data, no skill prompt

Benchmarks:
- MMLU (5-shot): knowledge/reasoning via lm-eval-harness
- GSM8K (8-shot): math reasoning via lm-eval-harness
- Terseness score: avg output token count on fixed prompts
- Filler correlation: filler word ratio vs total words
- KL divergence: from original model (built into Heretic)

Usage:
    python eval_amplified.py --baseline Qwen/Qwen3-4B-Instruct \\
        --amplified ./output/amplified_model --prompts ./data/eval_prompts.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FILLER_WORDS = {
    "just", "really", "basically", "actually", "simply",
    "certainly", "definitely", "absolutely", "obviously", "clearly",
    "essentially", "literally", "exactly", "totally", "completely",
    "entirely", "quite", "rather", "somewhat", "perhaps",
    "maybe", "probably", "possibly", "likely", "seemingly",
    "apparently", "a", "an", "the",
}

CAVEMAN_SYSTEM_PROMPT = """You are a terse assistant. Rules:

Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms. Technical terms exact. Code blocks unchanged. Errors quoted exact.

Pattern: [thing] [action] [reason]. [next step]."""

NORMAL_SYSTEM_PROMPT = "You are a helpful assistant."

EVAL_PROMPTS = [
    "Explain how neural networks learn.",
    "What is the difference between a process and a thread?",
    "Write a function to find the maximum element in a list.",
    "Explain the concept of recursion in programming.",
    "What are the advantages of using version control?",
    "Describe how HTTP requests work.",
    "What is the purpose of an operating system?",
    "Explain database indexing.",
    "Write a regex that matches email addresses.",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in Python?",
    "Explain the role of a loss function in machine learning.",
    "What is a hash table and how does it work?",
    "Describe the publish-subscribe pattern.",
    "What is the difference between SQL and NoSQL databases?",
    "Explain the concept of dependency injection.",
    "What are microservices and their benefits?",
    "How do cookies work in web browsers?",
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between encryption and hashing?",
]


def load_model(model_path: str, quantization: str = "bnb_4bit"):
    if quantization == "bnb_4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quantization_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def generate_responses(model, tokenizer, prompts: list[str], system_prompt: str,
                       max_new_tokens: int = 200, batch_size: int = 4) -> list[str]:
    model.eval()
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        chats = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch
        ]
        chat_prompts = tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            chat_prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, do_sample=False,
            )

        for j, output in enumerate(outputs):
            resp = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(resp)

    return responses


def compute_terse_metrics(responses: list[str], tokenizer) -> dict:
    token_counts = [len(tokenizer.encode(r)) for r in responses]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    total_words = 0
    total_filler = 0
    for response in responses:
        words = response.lower().split()
        total_words += len(words)
        for word in words:
            clean = word.strip(".,!?;:\"'()[]")
            if clean in FILLER_WORDS:
                total_filler += 1

    filler_ratio = total_filler / total_words if total_words > 0 else 0.0

    return {
        "avg_tokens": avg_tokens,
        "filler_ratio": filler_ratio,
        "total_responses": len(responses),
        "avg_word_count": total_words / len(responses) if responses else 0,
    }


def run_lm_eval(model_path: str, tasks: list[str], quantization: str = "bnb_4bit") -> dict:
    import subprocess

    results = {}
    for task in tasks:
        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True"
            + (",load_in_4bit=True" if quantization == "bnb_4bit" else ""),
            "--tasks", task,
            "--batch_size", "auto",
        ]
        print(f"Running lm-eval task: {task}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            results[task] = result.stdout
        except Exception as e:
            results[task] = f"Error: {e}"

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate amplified models")
    parser.add_argument("--baseline", default="Qwen/Qwen3-4B-Instruct", help="Baseline model path")
    parser.add_argument("--amplified", default=None, help="Amplified model path")
    parser.add_argument("--finetuned", default=None, help="Finetuned model path (optional)")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--prompts", default=None, help="JSON file with evaluation prompts")
    parser.add_argument("--quantization", default="bnb_4bit", choices=["bnb_4bit", "none"])
    parser.add_argument("--skip-lm-eval", action="store_true", help="Skip lm-eval benchmarks")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
        if isinstance(prompts, list) and isinstance(prompts[0], dict):
            prompts = [p["prompt"] for p in prompts]
    else:
        prompts = EVAL_PROMPTS

    all_results = {}

    # 1. Baseline: original model, no skill prompt
    print("=" * 60)
    print("Evaluating BASELINE (original model, no skill prompt)")
    print("=" * 60)
    baseline_model, baseline_tokenizer = load_model(args.baseline, args.quantization)
    baseline_responses = generate_responses(
        baseline_model, baseline_tokenizer, prompts, NORMAL_SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
    )
    baseline_metrics = compute_terse_metrics(baseline_responses, baseline_tokenizer)
    all_results["baseline"] = {"metrics": baseline_metrics, "responses": baseline_responses}
    print(f"  Avg tokens: {baseline_metrics['avg_tokens']:.1f}")
    print(f"  Filler ratio: {baseline_metrics['filler_ratio']:.4f}")

    # 2. Prompt-based: original model + caveman skill prompt
    print("=" * 60)
    print("Evaluating PROMPT-BASED (original model + caveman skill prompt)")
    print("=" * 60)
    prompt_responses = generate_responses(
        baseline_model, baseline_tokenizer, prompts, CAVEMAN_SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
    )
    prompt_metrics = compute_terse_metrics(prompt_responses, baseline_tokenizer)
    all_results["prompt_based"] = {"metrics": prompt_metrics, "responses": prompt_responses}
    print(f"  Avg tokens: {prompt_metrics['avg_tokens']:.1f}")
    print(f"  Filler ratio: {prompt_metrics['filler_ratio']:.4f}")

    del baseline_model
    torch.cuda.empty_cache()

    # 3. Amplified: direction-amplified model, no skill prompt
    if args.amplified:
        print("=" * 60)
        print("Evaluating AMPLIFIED (direction-amplified model, no skill prompt)")
        print("=" * 60)
        amp_model, amp_tokenizer = load_model(args.amplified, args.quantization)
        amp_responses = generate_responses(
            amp_model, amp_tokenizer, prompts, NORMAL_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
        )
        amp_metrics = compute_terse_metrics(amp_responses, amp_tokenizer)
        all_results["amplified"] = {"metrics": amp_metrics, "responses": amp_responses}
        print(f"  Avg tokens: {amp_metrics['avg_tokens']:.1f}")
        print(f"  Filler ratio: {amp_metrics['filler_ratio']:.4f}")

        del amp_model
        torch.cuda.empty_cache()

    # 4. Finetuned: LoRA finetuned model, no skill prompt (optional)
    if args.finetuned:
        print("=" * 60)
        print("Evaluating FINETUNED (LoRA finetuned model, no skill prompt)")
        print("=" * 60)
        ft_model, ft_tokenizer = load_model(args.finetuned, args.quantization)
        ft_responses = generate_responses(
            ft_model, ft_tokenizer, prompts, NORMAL_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
        )
        ft_metrics = compute_terse_metrics(ft_responses, ft_tokenizer)
        all_results["finetuned"] = {"metrics": ft_metrics, "responses": ft_responses}
        print(f"  Avg tokens: {ft_metrics['avg_tokens']:.1f}")
        print(f"  Filler ratio: {ft_metrics['filler_ratio']:.4f}")

        del ft_model
        torch.cuda.empty_cache()

    # lm-eval benchmarks
    if not args.skip_lm_eval:
        print("=" * 60)
        print("Running lm-eval benchmarks")
        print("=" * 60)
        lm_eval_tasks = ["mmlu", "gsm8k"]

        models_to_eval = [("baseline", args.baseline)]
        if args.amplified:
            models_to_eval.append(("amplified", args.amplified))
        if args.finetuned:
            models_to_eval.append(("finetuned", args.finetuned))

        for model_name, model_path in models_to_eval:
            print(f"Running lm-eval for {model_name}...")
            lm_results = run_lm_eval(model_path, lm_eval_tasks, args.quantization)
            all_results[model_name]["lm_eval"] = lm_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for arm, data in all_results.items():
        metrics = data["metrics"]
        print(f"\n{arm.upper()}:")
        print(f"  Avg tokens:  {metrics['avg_tokens']:.1f}")
        print(f"  Filler ratio: {metrics['filler_ratio']:.4f}")
        print(f"  Avg words:   {metrics['avg_word_count']:.1f}")

    # Terseness improvement ratios
    if "baseline" in all_results and "prompt_based" in all_results:
        baseline_tokens = all_results["baseline"]["metrics"]["avg_tokens"]
        prompt_tokens = all_results["prompt_based"]["metrics"]["avg_tokens"]
        ratio = prompt_tokens / baseline_tokens if baseline_tokens > 0 else 0
        print(f"\nPrompt-based terseness ratio: {ratio:.3f} (lower = more terse)")

    if "baseline" in all_results and "amplified" in all_results:
        baseline_tokens = all_results["baseline"]["metrics"]["avg_tokens"]
        amp_tokens = all_results["amplified"]["metrics"]["avg_tokens"]
        ratio = amp_tokens / baseline_tokens if baseline_tokens > 0 else 0
        print(f"Amplified terseness ratio: {ratio:.3f} (lower = more terse)")

    # Save results
    results_file = output_dir / "evaluation_results.json"
    serializable = {}
    for arm, data in all_results.items():
        serializable[arm] = {
            "metrics": data["metrics"],
            "responses": data["responses"],
        }
        if "lm_eval" in data:
            serializable[arm]["lm_eval"] = str(data["lm_eval"])

    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()