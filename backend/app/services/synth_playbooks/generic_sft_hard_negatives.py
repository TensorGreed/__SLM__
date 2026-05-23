"""HARD_NEGATIVES playbook for the `generic-sft` recipe.

Generates off-task / refusal prompts that the model should *not*
satisfy with a substantive answer. The completion is a refusal in
the same style as the user's brand voice (read from a sample
positive). Useful for tool-call SFT (the model should emit
``{"tool": "none"}`` instead of inventing a call) and chat-style
SFT (the model should decline rather than hallucinate).
"""

from __future__ import annotations

from typing import Any

from .base import (
    PlaybookContext,
    SynthMode,
    SynthRow,
    parse_jsonl_lines,
    register_playbook,
    sample_gold_rows,
)


_PROMPT_TEMPLATE = """\
You are generating training data for an instruction-tuning task.

Existing positive (prompt, completion) pairs:
{example_block}

Your task: generate {target_count} HARD NEGATIVE examples.

A hard negative is a prompt the model should NOT answer substantively. It's:
  - Off-topic for the system's apparent purpose
  - Out-of-scope (asks for something the system doesn't do)
  - Adversarial (asks the system to violate its rules)
  - Or just nonsensical / malformed

For each example, write a single JSON line:
{{"prompt": "<off-task or adversarial user input>", "completion": "<a polite refusal that matches the brand voice of the positive examples>"}}

Rules:
  - The completion must be a refusal, not a substantive answer.
  - Match the formality + tone of the positive examples' completions.
  - For tool-call style data, valid refusal is: {{"tool": "none", "args": {{}}}}.
  - Vary the kind of off-task prompt across examples.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


# Heuristic refusal phrases — any one of these in the completion
# raises confidence; absence drops it (the model probably generated
# a substantive answer instead of a refusal).
_REFUSAL_MARKERS = [
    "i can't",
    "i cannot",
    "i'm not able",
    "i am not able",
    "out of scope",
    "outside",
    "can't help",
    "cannot help",
    "decline",
    "tool\": \"none",  # tool-call recipe (escaped quote intentionally)
    "tool': 'none",
    "sorry",
    "unfortunately",
    "unable to",
    "i don't have",
]


class GenericSftHardNegativesPlaybook:
    recipe_id = "generic-sft"
    mode = SynthMode.HARD_NEGATIVES

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        examples = sample_gold_rows(gold, count=min(5, len(gold)), seed=0)
        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            prompt = self._extract_prompt(row)
            completion = self._extract_completion(row)
            lines.append(f"{i}. prompt: {prompt!r}\n   completion: {completion!r}")

        return _PROMPT_TEMPLATE.format(
            example_block="\n".join(lines) or "(none)",
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        accepted: list[SynthRow] = []
        for row in parsed_rows:
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            prompt, completion = prompt.strip(), completion.strip()
            if not prompt or not completion:
                continue
            if len(prompt) < 5 or len(prompt) > 4000 or len(completion) < 5 or len(completion) > 4000:
                continue
            lower = completion.lower()
            is_refusal = any(marker in lower for marker in _REFUSAL_MARKERS)
            # If the completion doesn't look like a refusal, the model
            # generated a substantive answer to the off-task prompt —
            # exactly the failure mode this playbook trains against.
            # Drop confidence sharply.
            confidence = 1.0 if is_refusal else 0.30
            accepted.append({
                "payload": {"prompt": prompt, "completion": completion},
                "synth_confidence": confidence,
                "synth_source": f"playbook:generic-sft:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_prompt(row: dict[str, Any]) -> str:
        for key in ("prompt", "input", "question", "instruction"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for sub_key in ("prompt", "text", "instruction", "draft", "question"):
                    inner = value.get(sub_key)
                    if isinstance(inner, str):
                        return inner
        return ""

    @staticmethod
    def _extract_completion(row: dict[str, Any]) -> str:
        for key in ("completion", "output", "answer", "response"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        expected = row.get("expected")
        if isinstance(expected, dict):
            for sub_key in ("completion", "output", "answer", "rewrite", "response"):
                inner = expected.get(sub_key)
                if isinstance(inner, str):
                    return inner
        if isinstance(expected, str):
            return expected
        return ""


register_playbook(GenericSftHardNegativesPlaybook())
