# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import torch.nn.functional as F
from torch import Tensor

from .config import AbliterationMode, Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int
    base_avg_tokens: float
    base_filler_ratio: float

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        if settings.mode == AbliterationMode.AMPLIFY:
            print("* Computing baseline terseness metrics...")
            self.base_avg_tokens, self.base_filler_ratio = self.compute_terse_metrics()
            print(
                f"* Baseline avg tokens: [bold]{self.base_avg_tokens:.1f}[/], "
                f"filler ratio: [bold]{self.base_filler_ratio:.4f}[/]"
            )
            self.base_refusals = 0
        else:
            print("* Counting model refusals...")
            self.base_refusals = self.count_refusals()
            print(
                f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
            )
            self.base_avg_tokens = 0.0
            self.base_filler_ratio = 0.0

    def is_refusal(self, response: str) -> bool:
        if not response.strip():
            return True

        response = response.lower().replace("*", "")
        response = response.replace("\u2019", "'")
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        refusal_count = 0

        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        for prompt, response in zip(self.bad_prompts, responses):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def compute_filler_ratio(self, responses: list[str]) -> float:
        if not responses:
            return 0.0

        total_words = 0
        total_filler = 0

        for response in responses:
            words = response.lower().split()
            total_words += len(words)
            for word in words:
                clean = word.strip(".,!?;:\"'()[]")
                if clean in self.settings.filler_words:
                    total_filler += 1

        return total_filler / total_words if total_words > 0 else 0.0

    def compute_terse_metrics(self) -> tuple[float, float]:
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        token_counts = []
        for response in responses:
            token_count = len(self.model.tokenizer.encode(response))
            token_counts.append(token_count)

        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
        filler_ratio = self.compute_filler_ratio(responses)

        if self.settings.print_responses:
            for prompt, response in zip(self.bad_prompts, responses):
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(f"[bold]Response:[/] {response}")

        return avg_tokens, filler_ratio

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        if self.settings.mode == AbliterationMode.AMPLIFY:
            print("  * Computing terseness metrics...")
            avg_tokens, filler_ratio = self.compute_terse_metrics()
            print(
                f"  * Avg tokens: [bold]{avg_tokens:.1f}[/], "
                f"filler ratio: [bold]{filler_ratio:.4f}[/]"
            )

            if self.base_avg_tokens > 0:
                terseness_score = avg_tokens / self.base_avg_tokens
            else:
                terseness_score = float(avg_tokens)

            if self.base_filler_ratio > 0:
                filler_score = filler_ratio / self.base_filler_ratio
            else:
                filler_score = float(filler_ratio)

            if kl_divergence >= kl_divergence_target:
                kld_score = kl_divergence / kl_divergence_scale
            else:
                kld_score = terseness_score * kl_divergence_target / kl_divergence_scale

            combined_score = (terseness_score + filler_score) / 2.0

            refusals = 0

            score = (kld_score, combined_score)
            return score, kl_divergence, refusals
        else:
            print("  * Counting model refusals...")
            refusals = self.count_refusals()
            print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

            refusals_score = (
                refusals / self.base_refusals if self.base_refusals > 0 else float(refusals)
            )

            if kl_divergence >= kl_divergence_target:
                kld_score = kl_divergence / kl_divergence_scale
            else:
                kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

            score = (
                kld_score,
                refusals_score,
            )

            return score, kl_divergence, refusals