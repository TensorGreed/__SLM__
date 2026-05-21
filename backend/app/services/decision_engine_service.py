"""Decision engine — "do you even need SFT?" recommender (Theme 7).

`analyze_domain_brief` already infers task family + output schema from
a plain-English brief. This module extends that with one more output:
a recommended *approach* (prompt_only / rag_first / sft / dpo /
distillation) plus a short rationale that quotes the brief, so the
frontend can render a chip pre-create: "Confidence 0.36 — try RAG
first" or "Style-shifting → SFT is right."

Design notes:

- Pure-Python, deterministic heuristic. No LLM call — the decision
  needs to be fast (debounced on every keystroke after the user has
  typed ~40 chars), cheap (the user may abandon mid-flow), and
  reproducible.
- Signals are recorded so the UI can show a "(?)" hover with the
  matched cues. Easier to debug than a single confidence number.
- All five approaches return a rationale; the UI distinguishes
  visual treatment by `approach` (SFT → green, others → warning).
- Default = SFT. The whole platform is an SFT workbench; the
  decision engine's job is to *honestly* flag the cases where SFT
  is the wrong tool, not to discourage SFT generally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas.domain_blueprint import DomainBlueprintContract


ApproachKind = Literal["prompt_only", "rag_first", "sft", "dpo", "distillation"]


# Keyword groups — lowercased; matched as substrings against the brief.
# Order matters: earlier groups win on tie. The groups below are
# tuned for the *signals* we want to catch, not a single best phrase;
# overlap is intentional (e.g. "rag" matches both "rag" and "knowledge").
_KW_RAG = (
    "rag", "retrieval", "retrieve", "lookup", "look up", "look-up",
    "search", "knowledge base", "knowledge-base", "kb",
    "documentation", "docs", "wiki", "policies", "policy lookup",
    "facts", "factual", "find information", "answer from documents",
    "ground", "grounded", "citations",
)
_KW_STYLE = (
    "style", "tone", "voice", "format", "schema", "json output",
    "structured output", "consistent format", "structured json",
    "rewrite in", "summarize in", "respond in", "phrased like",
    "match the tone", "branded voice",
)
_KW_DISTILL = (
    "smaller", "cheaper", "compress", "compressed", "distill",
    "distillation", "on-device", "on device", "edge device",
    "mobile cpu", "tiny model", "lightweight model", "reduce cost",
    "shrink", "quantize", "quantization",
)
_KW_DPO = (
    "preference", "preferences", "ranked", "ranking",
    "human feedback", "rlhf", "dpo", "kto", "ipo", "orpo",
    "rejected response", "chosen response", "alignment",
)
_KW_PROMPT_ONLY = (
    "few-shot", "few shot", "in-context", "in context",
    "system prompt", "prompt template", "no training",
    "without training", "just prompt",
)


@dataclass
class ApproachRecommendation:
    approach: ApproachKind
    confidence: float          # 0..1; how sure the engine is
    headline: str              # one-line UI chip text
    rationale: str             # 1-2 sentences quoting/citing the brief
    signals: list[str] = field(default_factory=list)  # debug-friendly signal trail
    cta_label: str = ""        # short button text e.g. "Try RAG first"

    def model_dump(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "confidence": round(self.confidence, 4),
            "headline": self.headline,
            "rationale": self.rationale,
            "signals": list(self.signals),
            "cta_label": self.cta_label,
        }


def _matched_keywords(text_lower: str, keywords: tuple[str, ...]) -> list[str]:
    return [kw for kw in keywords if kw in text_lower]


def _quote_brief(brief: str, max_words: int = 14) -> str:
    """Return a short quoted snippet of the brief for the rationale.
    Strips newlines + collapses whitespace; truncates with ellipsis."""
    cleaned = " ".join((brief or "").split())
    if not cleaned:
        return ""
    words = cleaned.split(" ")
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]) + "…"


def infer_recommended_approach(
    brief_text: str,
    blueprint: DomainBlueprintContract,
) -> ApproachRecommendation:
    """Decide whether SFT is the right tool for this brief.

    Heuristic order (first match wins among the non-SFT branches; SFT
    is the default fallthrough):

      1. Distillation signals — explicit cost/size language.
      2. DPO signals — preference / ranking / RLHF terminology.
      3. RAG signals — knowledge/lookup/retrieval terminology, OR
         a `rag_qa` task_family from the existing inference, OR a
         very low blueprint confidence (the brief is too vague to
         confidently SFT against; RAG is more forgiving).
      4. Prompt-only signals — explicit "few-shot" framing, AND a
         small fixed output schema AND a short brief.
      5. Default → SFT. Rationale acknowledges what was matched
         (style/structure cues) when available.

    Mutually exclusive by design — the chip shows one recommendation,
    not a confidence vector. Signals captured for the (?) hover are
    additive so the user can see why.
    """
    brief = (brief_text or "").strip()
    brief_lower = brief.lower()
    confidence_score = float(blueprint.confidence_score or 0.0)
    task_family = (blueprint.task_family or "").strip().lower()
    output_schema = blueprint.expected_output_schema or {}
    schema_props = {}
    if isinstance(output_schema, dict):
        schema_props = output_schema.get("properties") or {}

    signals: list[str] = []

    # 1. Distillation — explicit cost/size cues.
    distill_hits = _matched_keywords(brief_lower, _KW_DISTILL)
    if distill_hits:
        signals.append(f"keywords:distillation:{','.join(distill_hits[:3])}")
        return ApproachRecommendation(
            approach="distillation",
            confidence=0.75,
            headline="Distillation could give a smaller, cheaper model.",
            rationale=(
                f"You mentioned {distill_hits[0]!r} in the brief — that's a "
                "size/cost goal. Distillation from a larger teacher model "
                "gives a small SLM matching most of the teacher's quality. "
                "SFT alone won't shrink the base model."
            ),
            signals=signals,
            cta_label="Look into distillation",
        )

    # 2. DPO / preference learning — explicit ranking/RLHF cues.
    dpo_hits = _matched_keywords(brief_lower, _KW_DPO)
    if dpo_hits:
        signals.append(f"keywords:dpo:{','.join(dpo_hits[:3])}")
        return ApproachRecommendation(
            approach="dpo",
            confidence=0.7,
            headline="DPO/preference learning might fit better than SFT.",
            rationale=(
                f"Your brief mentions {dpo_hits[0]!r} — that's preference "
                "data, not single-best-answer data. DPO trains on "
                "(chosen, rejected) pairs directly. BrewSLM's `preference` "
                "task profile supports this."
            ),
            signals=signals,
            cta_label="Use DPO instead",
        )

    # 3. RAG signals.
    rag_hits = _matched_keywords(brief_lower, _KW_RAG)
    low_confidence = confidence_score > 0 and confidence_score < 0.45
    if rag_hits or task_family == "rag_qa":
        for kw in rag_hits[:3]:
            signals.append(f"keyword:rag:{kw}")
        if task_family == "rag_qa":
            signals.append("task_family:rag_qa")
        return ApproachRecommendation(
            approach="rag_first",
            confidence=0.75,
            headline="Try RAG first — your brief points at retrieval.",
            rationale=(
                f"You said {_quote_brief(brief)!r}. That reads as "
                "looking-something-up, not behavior-shaping. RAG against "
                "your docs is cheaper, fully traceable, and updates with "
                "your data. Use SFT later if RAG isn't enough."
            ),
            signals=signals,
            cta_label="Try RAG first",
        )

    if low_confidence:
        signals.append(f"blueprint.confidence:{confidence_score:.2f}")
        return ApproachRecommendation(
            approach="rag_first",
            confidence=0.55,
            headline=f"Confidence {int(round(confidence_score * 100))}% — try RAG first.",
            rationale=(
                "The brief was too vague for me to infer a clean SFT shape. "
                "RAG is more forgiving when the task isn't crisp — set up "
                "retrieval against your data and only SFT once the pattern "
                "is concrete."
            ),
            signals=signals,
            cta_label="Try RAG first",
        )

    # 4. Prompt-only — explicit "few-shot" cues + a tiny schema + a short brief.
    prompt_hits = _matched_keywords(brief_lower, _KW_PROMPT_ONLY)
    is_small_schema = bool(schema_props) and len(schema_props) <= 2
    is_short_brief = len(brief.split()) <= 25
    if prompt_hits and is_small_schema and is_short_brief:
        signals.append(f"keywords:prompt:{','.join(prompt_hits[:2])}")
        signals.append("schema:small")
        signals.append("brief:short")
        return ApproachRecommendation(
            approach="prompt_only",
            confidence=0.6,
            headline="A few-shot prompt may be enough — try that first.",
            rationale=(
                f"Your brief is short ({len(brief.split())} words) and the "
                "output schema is small. Three to five in-context examples "
                "in the system prompt usually clear this bar without "
                "training. SFT helps later if accuracy plateaus."
            ),
            signals=signals,
            cta_label="Try prompt engineering first",
        )

    # 5. Default — SFT. Acknowledge style/structure cues when present.
    style_hits = _matched_keywords(brief_lower, _KW_STYLE)
    if style_hits:
        for kw in style_hits[:3]:
            signals.append(f"keyword:style:{kw}")
        return ApproachRecommendation(
            approach="sft",
            confidence=0.85,
            headline="SFT is the right call — style/structure shaping.",
            rationale=(
                f"You said {_quote_brief(brief)!r}. That's exactly what "
                "SFT is for: shape the model's outputs to match a target "
                "style or structure your in-context-prompt can't reliably "
                "enforce."
            ),
            signals=signals,
            cta_label="Proceed with SFT",
        )

    # Plain SFT default — no specific signals matched.
    signals.append("default:no_signals_matched")
    return ApproachRecommendation(
        approach="sft",
        confidence=0.65,
        headline="SFT is a reasonable default for this brief.",
        rationale=(
            "Nothing in the brief flagged RAG, distillation, or preference "
            "learning as a better fit. Fine-tuning with a curated dataset "
            "is the safe default — refine the brief if the recommendation "
            "looks off."
        ),
        signals=signals,
        cta_label="Proceed with SFT",
    )


__all__ = [
    "ApproachKind",
    "ApproachRecommendation",
    "infer_recommended_approach",
]
