"""Quality-Lift phase 5 slice 2 — Behavioral test runner.

Runs the slice 1 behavioral test schema against the trained
classifier checkpoint and emits per-test pass rates that the existing
gate evaluator picks up via the canonical
``behavioral.<test_id>.pass_rate`` metric key.

The runner has two halves:

  * **Perturbation engine** — pure dispatch over the closed
    perturbation grammar. Deterministic given a seed so the failing
    example surfaced in the UI is the same one the user can
    re-produce locally.

  * **Test scorer** — drives the engine over each test's seed
    examples, runs them through an injectable ``predict_fn``, applies
    per-kind pass criteria, and returns a snapshot dict structured
    so ``_flatten_behavioral_test_metrics`` can flatten it into
    gate-resolvable keys without further plumbing.

The ``predict_fn`` injection is intentional: slice 2's tests run with
a mocked predictor (no torch), and slice 3's UI driver can use the
same shape for fixture-driven previews. Production wiring builds the
predict_fn from the trained checkpoint in
``_safe_run_behavioral_tests`` inside ``evaluation_service``.

Honest framing locked with the user (2026-06-09):
  - Per-test prediction budget cap: 2000 predictions. Tests defining
    huge fanouts (100 seed × 8 perts × 50 n_per_seed) sample down
    deterministically rather than blowing through GB10 minutes.
  - ``failed_examples`` capped at 10 per test for JSON budget; full
    enumeration would balloon the EvalResult.metrics payload.
  - Multi-seed composition: this runs once on the leader's borrowed
    checkpoint (single set of behavioral metrics per training run,
    not N). The phase 1 aggregator already recurses dicts, so
    cross-seed behavioral variance comes free if a future slice
    wants it — slice 2 just emits flat metrics on the leader.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Sequence


PER_TEST_PREDICTION_BUDGET = 2000
FAILED_EXAMPLES_CAP = 10


PredictFn = Callable[[Sequence[str]], list[str]]


# ────────────────────────────────────────────────────────────────────────
# Perturbation engine — closed dispatch + deterministic per-trial seed
# ────────────────────────────────────────────────────────────────────────


def _stable_seed(*parts: Any) -> int:
    """Build a deterministic integer seed from a tuple of parts.

    Used to derive a per-trial RNG seed so re-running on the same
    checkpoint produces identical perturbed inputs. ``hashlib.sha256``
    is overkill but it's standard and avoids the random.seed()
    quirk where built-in ``hash()`` on strings is salted per process.
    """
    digest = hashlib.sha256(
        "::".join(str(p) for p in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _resolve_rng(perturbation: dict[str, Any], *fallback_parts: Any) -> random.Random:
    """Derive a deterministic RNG for one perturbation × seed_example
    × n trial. ``perturbation.seed`` (if set by the pack) takes
    precedence so the user can pin a specific spread; otherwise we
    hash the fallback parts.
    """
    seed_hint = perturbation.get("seed")
    if seed_hint is not None:
        return random.Random(_stable_seed(seed_hint, *fallback_parts))
    return random.Random(_stable_seed(*fallback_parts))


def _apply_typo(text: str, perturbation: dict[str, Any], rng: random.Random) -> str:
    """Character-level swap perturbation. ``intensity`` is the
    fraction of swappable positions to actually swap (rounded up to
    at least one). Identical to a tiny random-swap augmenter — not
    QWERTY-aware, but cheap and reproducible."""
    if len(text) < 2:
        return text
    intensity = float(perturbation.get("intensity") or 0.05)
    swappable = max(1, len(text) - 1)
    n_swaps = max(1, int(round(swappable * intensity)))
    n_swaps = min(n_swaps, swappable)
    chars = list(text)
    # Sample distinct positions without replacement so we don't
    # quietly cancel a swap by re-targeting the same index.
    positions = rng.sample(range(swappable), n_swaps)
    for pos in positions:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _apply_insert_token(text: str, perturbation: dict[str, Any]) -> str:
    """Insert ``params.token`` at ``params.position``. 0 = prepend,
    -1 = append, positive int = absolute char position (clamped)."""
    params = perturbation.get("params") or {}
    token = str(params.get("token") or "")
    position = int(params.get("position", 0))
    if not token:
        return text
    if position == -1 or position > len(text):
        return text + token
    if position <= 0:
        return token + text
    return text[:position] + token + text[position:]


def _apply_case_change(text: str, perturbation: dict[str, Any]) -> str:
    params = perturbation.get("params") or {}
    case_op = str(params.get("case") or "lower").lower()
    if case_op == "upper":
        return text.upper()
    if case_op == "title":
        return text.title()
    return text.lower()


def _apply_whitespace_jitter(
    text: str, perturbation: dict[str, Any], rng: random.Random,
) -> str:
    """Per-character coin flip — with probability ``intensity`` a
    space becomes a doubled space, and a non-space stays. Doesn't
    insert at non-space positions; preserves token boundaries while
    nudging whitespace structure.

    Caveat: not a deep augmenter, but the slice 2 contract is
    "demonstrate the runner dispatches on this kind." A real-world
    use case would just rely on case + insert_token for now.
    """
    intensity = float(perturbation.get("intensity") or 0.20)
    out: list[str] = []
    for ch in text:
        if ch == " " and rng.random() < intensity:
            out.append("  ")
        else:
            out.append(ch)
    return "".join(out)


def apply_perturbation(
    text: str,
    perturbation: dict[str, Any],
    *,
    rng: random.Random,
) -> str:
    """Closed dispatch over the slice-1 perturbation grammar. Unknown
    kinds raise — the schema validator should have caught these at
    pack save time."""
    kind = perturbation.get("kind")
    if kind == "typo":
        return _apply_typo(text, perturbation, rng)
    if kind == "insert_token":
        return _apply_insert_token(text, perturbation)
    if kind == "case_change":
        return _apply_case_change(text, perturbation)
    if kind == "whitespace_jitter":
        return _apply_whitespace_jitter(text, perturbation, rng)
    raise ValueError(f"unknown_perturbation_kind:{kind}")


# ────────────────────────────────────────────────────────────────────────
# Trial-list generation per test kind
# ────────────────────────────────────────────────────────────────────────


def _generate_inv_dir_trials(
    test: dict[str, Any],
) -> list[dict[str, Any]]:
    """For each (seed_example, perturbation, trial_index) combination,
    yield one trial with the perturbed input. Trials for INV and DIR
    share this shape — only the pass criterion differs at score time.
    """
    seed_examples = test.get("seed_examples") or []
    perturbations = test.get("perturbations") or []
    n_per_seed = max(1, int(test.get("n_perturbations_per_seed") or 1))
    test_id = test["test_id"]

    trials: list[dict[str, Any]] = []
    for seed_idx, seed in enumerate(seed_examples):
        original_text = str(seed.get("input") or "")
        given_label = seed.get("given_label")
        for pert_idx, perturbation in enumerate(perturbations):
            for n_idx in range(n_per_seed):
                rng = _resolve_rng(
                    perturbation, test_id, seed_idx, pert_idx, n_idx,
                )
                perturbed = apply_perturbation(original_text, perturbation, rng=rng)
                trials.append({
                    "original_input": original_text,
                    "perturbed_input": perturbed,
                    "perturbation_name": perturbation.get("name") or perturbation.get("kind"),
                    "given_label": given_label,
                    "seed_index": seed_idx,
                })
    return trials


def _generate_mft_trials(test: dict[str, Any]) -> list[dict[str, Any]]:
    """MFT trials carry their expected_label inline — no perturbation
    fanout, no original prediction needed."""
    out: list[dict[str, Any]] = []
    for idx, example in enumerate(test.get("examples") or []):
        out.append({
            "input": str(example.get("input") or ""),
            "expected_label": example.get("expected_label"),
            "example_index": idx,
        })
    return out


def _budget_sample(
    trials: list[dict[str, Any]],
    *,
    budget: int,
    test_id: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Cap at ``budget`` predictions per test. Returns ``(sampled,
    capped)``. ``capped`` lets the snapshot stamp a flag so the UI
    can show "tested 2000 of 4000 trials" instead of silently
    truncating."""
    if len(trials) <= budget:
        return trials, False
    # Deterministic sampling keyed on the test_id so re-runs against
    # the same pack pick the same subset.
    rng = random.Random(_stable_seed("budget_sample", test_id))
    sample = rng.sample(trials, budget)
    return sample, True


# ────────────────────────────────────────────────────────────────────────
# Per-kind scoring
# ────────────────────────────────────────────────────────────────────────


def _score_inv_test(
    test: dict[str, Any],
    trials: list[dict[str, Any]],
    predicted_originals_by_seed_idx: dict[int, str],
    perturbed_preds: list[str],
) -> dict[str, Any]:
    """INV: a trial passes when the perturbed prediction matches the
    ORIGINAL prediction (the model's own decision, not the
    ``given_label``). Catches cases where the model is consistently
    wrong but invariant — which is fine for INV — versus a row whose
    label flips under noise.
    """
    passed = 0
    failed_samples: list[dict[str, Any]] = []
    for trial, perturbed_label in zip(trials, perturbed_preds):
        original_label = predicted_originals_by_seed_idx.get(trial["seed_index"])
        ok = original_label is not None and perturbed_label == original_label
        if ok:
            passed += 1
        elif len(failed_samples) < FAILED_EXAMPLES_CAP:
            failed_samples.append({
                "original_input": trial["original_input"],
                "perturbed_input": trial["perturbed_input"],
                "perturbation_name": trial["perturbation_name"],
                "original_label": original_label,
                "perturbed_label": perturbed_label,
            })
    total = len(trials)
    return {
        "kind": "INV",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "failed_examples": failed_samples,
    }


def _score_dir_test(
    test: dict[str, Any],
    trials: list[dict[str, Any]],
    predicted_originals_by_seed_idx: dict[int, str],
    perturbed_preds: list[str],
) -> dict[str, Any]:
    """DIR: pass criterion depends on expectation.kind. ``must_change``
    accepts any flip; ``must_change_to`` requires a specific label;
    ``must_change_to_one_of`` accepts any label in a set.
    """
    expectation = test.get("expectation") or {}
    kind = str(expectation.get("kind") or "must_change").lower()
    target_label = expectation.get("target_label")
    target_set = set(expectation.get("target_labels") or [])

    def _passes(original: str | None, perturbed: str) -> bool:
        if original is None:
            return False
        if kind == "must_change":
            return perturbed != original
        if kind == "must_change_to":
            return perturbed == target_label and perturbed != original
        if kind == "must_change_to_one_of":
            return perturbed in target_set and perturbed != original
        return False

    passed = 0
    failed_samples: list[dict[str, Any]] = []
    for trial, perturbed_label in zip(trials, perturbed_preds):
        original_label = predicted_originals_by_seed_idx.get(trial["seed_index"])
        ok = _passes(original_label, perturbed_label)
        if ok:
            passed += 1
        elif len(failed_samples) < FAILED_EXAMPLES_CAP:
            failed_samples.append({
                "original_input": trial["original_input"],
                "perturbed_input": trial["perturbed_input"],
                "perturbation_name": trial["perturbation_name"],
                "original_label": original_label,
                "perturbed_label": perturbed_label,
                "expectation_kind": kind,
                "target": target_label if kind == "must_change_to"
                          else sorted(target_set) if kind == "must_change_to_one_of"
                          else None,
            })
    total = len(trials)
    return {
        "kind": "DIR",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "failed_examples": failed_samples,
    }


def _score_mft_test(
    test: dict[str, Any],
    trials: list[dict[str, Any]],
    predicted_labels: list[str],
) -> dict[str, Any]:
    """MFT: each example must match its expected_label exactly."""
    passed = 0
    failed_samples: list[dict[str, Any]] = []
    for trial, predicted in zip(trials, predicted_labels):
        expected = trial["expected_label"]
        ok = predicted == expected
        if ok:
            passed += 1
        elif len(failed_samples) < FAILED_EXAMPLES_CAP:
            failed_samples.append({
                "input": trial["input"],
                "expected_label": expected,
                "predicted_label": predicted,
            })
    total = len(trials)
    return {
        "kind": "MFT",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "failed_examples": failed_samples,
    }


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────


def run_behavioral_tests(
    behavioral_tests: list[dict[str, Any]],
    *,
    predict_fn: PredictFn,
) -> dict[str, dict[str, Any]]:
    """Run every test in ``behavioral_tests`` against the model
    provided by ``predict_fn`` (signature: ``list[str] -> list[str]``,
    one predicted label per input).

    Returns ``{test_id: {pass_rate, passed, total, failed_examples,
    kind, ...}}``. Slice 2's snapshot flattener turns this into
    canonical / short / scoped metric keys the gate evaluator
    already knows how to read.

    Empty / None ``behavioral_tests`` → ``{}`` (no behavioral metric
    in EvalResult.metrics; gates referencing behavioral keys then
    resolve as missing_metric_*, same as today's missing-eval flow).
    """
    if not behavioral_tests:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for test in behavioral_tests:
        test_id = test.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            continue
        kind = (test.get("kind") or "").upper()

        if kind == "MFT":
            trials = _generate_mft_trials(test)
            if not trials:
                continue
            trials, capped = _budget_sample(
                trials, budget=PER_TEST_PREDICTION_BUDGET, test_id=test_id,
            )
            inputs = [t["input"] for t in trials]
            predictions = list(predict_fn(inputs))
            result = _score_mft_test(test, trials, predictions)
        else:  # INV or DIR
            trials = _generate_inv_dir_trials(test)
            if not trials:
                continue
            # Budget split: leave headroom for one prediction per
            # unique seed_example (the originals). Practically this
            # is ~100 unique seeds at most so the dent in 2000 is
            # negligible.
            trials, capped = _budget_sample(
                trials,
                budget=PER_TEST_PREDICTION_BUDGET,
                test_id=test_id,
            )
            seed_originals_seen: dict[int, str] = {}
            originals_input_list: list[str] = []
            seed_idx_order: list[int] = []
            for trial in trials:
                if trial["seed_index"] not in seed_originals_seen:
                    seed_originals_seen[trial["seed_index"]] = trial["original_input"]
                    originals_input_list.append(trial["original_input"])
                    seed_idx_order.append(trial["seed_index"])
            original_preds = list(predict_fn(originals_input_list))
            predicted_originals_by_seed_idx: dict[int, str] = {
                seed_idx: label
                for seed_idx, label in zip(seed_idx_order, original_preds)
            }
            perturbed_preds = list(predict_fn([t["perturbed_input"] for t in trials]))
            if kind == "INV":
                result = _score_inv_test(
                    test, trials, predicted_originals_by_seed_idx, perturbed_preds,
                )
            else:
                result = _score_dir_test(
                    test, trials, predicted_originals_by_seed_idx, perturbed_preds,
                )

        if capped:
            result["capped_at_budget"] = PER_TEST_PREDICTION_BUDGET
        out[test_id] = result
    return out


# ────────────────────────────────────────────────────────────────────────
# Production wiring helper — classifier-head text prediction
# ────────────────────────────────────────────────────────────────────────


def build_classifier_predict_fn(
    *,
    model_path: str,
    label_space: list[str],
) -> PredictFn:
    """Build a ``predict_fn`` backed by the same classifier-head
    inference path Epic F Phase 1 already uses. Loads the model
    once, returns a closure that batches per-call.

    Slice 2's tests mock this whole function out — the runner takes
    ``predict_fn`` as a parameter precisely so a torchless test
    environment can still cover scoring + flattening end-to-end.
    """
    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_space),
        trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base, model_path)
    except Exception:
        model = base
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def _predict(texts: Sequence[str]) -> list[str]:
        out: list[str] = []
        with torch.inference_mode():
            for text in texts:
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)
                logits = model(**tokens).logits[0].float().tolist()
                pred_idx = max(range(len(logits)), key=lambda i: logits[i])
                out.append(str(label_space[pred_idx]))
        return out

    return _predict
