"""Quality-Lift phase 5 slice 1 — Behavioral test schema validator.

Behavioral tests (CheckList-style — Ribeiro et al. 2020) live on the
eval pack contract under ``task_specs[].behavioral_tests``. Three
kinds:

  * **MFT** (Minimum Functionality Test) — hand-authored
    ``(input, expected_label)`` canonical cases the model must pass.
  * **INV** (Invariance test) — label-preserving perturbations on a
    seed example; the model's prediction must NOT change.
  * **DIR** (Directional Expectation test) — meaning-changing
    perturbations on a seed example; the model's prediction MUST
    change in a specified direction.

Slice 1 ships the schema gatekeeper only — slice 2 wires the runner
into ``run_evaluation`` and emits ``metrics["behavioral"][test_id]``;
slice 3 surfaces failures in the ScorecardPanel drill-down.

**No string DSL** — every field is a closed enum or a typed value so
the editor can render exhaustive op pickers and there's no
eval-injection surface.

Design constraints locked with the user (2026-06-09):
  - Storage: on the pack contract, not on Project. Behavioral tests
    are task-shape-bound (sentiment invariance probes don't apply
    to extraction packs).
  - Perturbation set for slice 1: ``typo`` / ``insert_token`` /
    ``case_change`` / ``whitespace_jitter``. Paraphrase / synonym /
    back-translation deferred to phase 5b (need a paraphrase model
    or lexicon).
  - Caps: ≤30 tests per pack, ≤8 perturbations per test, ≤100
    seed examples per test. Hard caps catch carelessness; the
    runner's 2000-prediction budget (slice 2) catches deliberate
    overload.
  - ``pass_rate_floor`` is metadata-only — the authoritative
    threshold lives on the corresponding gate in
    ``task_specs.gates[]``. The floor exists as a default the
    Coach nudge (slice 3) uses when proposing a gate.
"""

from __future__ import annotations

import re
from typing import Any


# ── Closed enums ──────────────────────────────────────────────────────

# Test kinds. Adding here is a deliberate decision — slice 2's runner
# dispatches on this exact tuple, and slice 3's UI renders
# kind-specific icons / drill-down columns.
BEHAVIORAL_TEST_KINDS: tuple[str, ...] = ("INV", "DIR", "MFT")

# Perturbation kinds the slice 2 runner will dispatch on. Each entry
# requires specific ``params`` and may carry an ``intensity``.
PERTURBATION_KINDS: tuple[str, ...] = (
    "typo",
    "insert_token",
    "case_change",
    "whitespace_jitter",
)

# Directional expectation flavors. Same closed-grammar discipline as
# slice predicates from phase 2 — string DSLs are forbidden because
# they'd be unverifiable at write time.
DIR_EXPECTATION_KINDS: tuple[str, ...] = (
    "must_change",                # any label change passes
    "must_change_to",             # ``target_label`` required
    "must_change_to_one_of",      # ``target_labels`` required (list)
)

# ── Caps ─────────────────────────────────────────────────────────────

MAX_TESTS_PER_PACK = 30
MAX_PERTURBATIONS_PER_TEST = 8
MAX_SEED_EXAMPLES_PER_TEST = 100
MAX_MFT_EXAMPLES_PER_TEST = 100
MAX_INPUT_CHARS = 4096          # per seed / MFT example
MAX_TOKEN_INSERT_CHARS = 256    # cap insert_token's ``token`` param

# ── ID grammar ───────────────────────────────────────────────────────

# Same shape as slice_definitions_service's slice_id regex. The id
# flattens into metric keys like ``behavioral.<test_id>.pass_rate``;
# the catalog's ``is_behavioral_metric_id`` regex relies on the
# leading-letter-lowercase-ASCII constraint.
_TEST_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class BehavioralTestValidationError(ValueError):
    """Raised on the first malformed entry. Carries a stable code
    prefix so the API layer can map to specific 400 messages and the
    editor can highlight the offending row."""


# ────────────────────────────────────────────────────────────────────────
# Perturbation validation
# ────────────────────────────────────────────────────────────────────────


def _validate_typo(test_id: str, perturbation: dict[str, Any]) -> dict[str, Any]:
    raw_intensity = perturbation.get("intensity")
    if raw_intensity is None:
        intensity = 0.05
    else:
        if not isinstance(raw_intensity, (int, float)) or isinstance(raw_intensity, bool):
            raise BehavioralTestValidationError(
                f"perturbation_intensity_invalid:{test_id}"
            )
        intensity = float(raw_intensity)
    if not (0.0 < intensity <= 0.5):
        raise BehavioralTestValidationError(
            f"perturbation_intensity_out_of_range:{test_id}"
        )
    return {
        "kind": "typo",
        "name": str(perturbation.get("name") or "typo"),
        "intensity": intensity,
        "seed": perturbation.get("seed"),
    }


def _validate_insert_token(
    test_id: str, perturbation: dict[str, Any]
) -> dict[str, Any]:
    params = perturbation.get("params")
    if not isinstance(params, dict):
        raise BehavioralTestValidationError(
            f"perturbation_params_invalid:{test_id}"
        )
    token = params.get("token")
    if not isinstance(token, str) or not token:
        raise BehavioralTestValidationError(
            f"perturbation_token_required:{test_id}"
        )
    if len(token) > MAX_TOKEN_INSERT_CHARS:
        raise BehavioralTestValidationError(
            f"perturbation_token_too_long:{test_id}"
        )
    position = params.get("position", 0)
    if not isinstance(position, int) or isinstance(position, bool):
        raise BehavioralTestValidationError(
            f"perturbation_position_invalid:{test_id}"
        )
    # ``position`` is interpreted by the slice-2 runner: 0 prepend,
    # -1 append, positive int → absolute char position. We validate
    # the type here, not the semantics — large positions are clamped
    # at run time.
    return {
        "kind": "insert_token",
        "name": str(perturbation.get("name") or f"insert_{token.strip()[:16] or 'token'}"),
        "params": {"token": token, "position": position},
        "seed": perturbation.get("seed"),
    }


_CASE_CHANGE_OPTIONS = frozenset({"lower", "upper", "title"})


def _validate_case_change(
    test_id: str, perturbation: dict[str, Any]
) -> dict[str, Any]:
    params = perturbation.get("params")
    if not isinstance(params, dict):
        raise BehavioralTestValidationError(
            f"perturbation_params_invalid:{test_id}"
        )
    case_op = str(params.get("case") or "").strip().lower()
    if case_op not in _CASE_CHANGE_OPTIONS:
        raise BehavioralTestValidationError(
            f"perturbation_case_invalid:{test_id}"
        )
    return {
        "kind": "case_change",
        "name": str(perturbation.get("name") or f"case_{case_op}"),
        "params": {"case": case_op},
        "seed": perturbation.get("seed"),
    }


def _validate_whitespace_jitter(
    test_id: str, perturbation: dict[str, Any]
) -> dict[str, Any]:
    # No required params; intensity controls collapse vs expand ratio.
    raw_intensity = perturbation.get("intensity")
    if raw_intensity is None:
        intensity = 0.20
    else:
        if not isinstance(raw_intensity, (int, float)) or isinstance(raw_intensity, bool):
            raise BehavioralTestValidationError(
                f"perturbation_intensity_invalid:{test_id}"
            )
        intensity = float(raw_intensity)
    if not (0.0 < intensity <= 1.0):
        raise BehavioralTestValidationError(
            f"perturbation_intensity_out_of_range:{test_id}"
        )
    return {
        "kind": "whitespace_jitter",
        "name": str(perturbation.get("name") or "whitespace_jitter"),
        "intensity": intensity,
        "seed": perturbation.get("seed"),
    }


_PERTURBATION_VALIDATORS = {
    "typo": _validate_typo,
    "insert_token": _validate_insert_token,
    "case_change": _validate_case_change,
    "whitespace_jitter": _validate_whitespace_jitter,
}


def _validate_perturbation(
    test_id: str, raw: Any
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BehavioralTestValidationError(
            f"perturbation_shape_invalid:{test_id}"
        )
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in PERTURBATION_KINDS:
        raise BehavioralTestValidationError(
            f"unknown_perturbation_kind:{test_id}:{kind}"
        )
    return _PERTURBATION_VALIDATORS[kind](test_id, raw)


# ────────────────────────────────────────────────────────────────────────
# Per-kind shape validation
# ────────────────────────────────────────────────────────────────────────


def _validate_seed_examples(
    test_id: str, raw: Any, *, max_examples: int = MAX_SEED_EXAMPLES_PER_TEST,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise BehavioralTestValidationError(
            f"seed_examples_required:{test_id}"
        )
    if len(raw) > max_examples:
        raise BehavioralTestValidationError(
            f"seed_examples_too_many:{test_id}"
        )
    cleaned: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise BehavioralTestValidationError(
                f"seed_example_shape_invalid:{test_id}:{idx}"
            )
        text = entry.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BehavioralTestValidationError(
                f"seed_example_input_required:{test_id}:{idx}"
            )
        if len(text) > MAX_INPUT_CHARS:
            raise BehavioralTestValidationError(
                f"seed_example_input_too_long:{test_id}:{idx}"
            )
        given = entry.get("given_label")
        if given is not None and not isinstance(given, str):
            raise BehavioralTestValidationError(
                f"seed_example_label_invalid:{test_id}:{idx}"
            )
        cleaned.append({
            "input": text.strip(),
            "given_label": given.strip() if isinstance(given, str) else None,
        })
    return cleaned


def _validate_inv_test(test_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    seed_examples = _validate_seed_examples(test_id, raw.get("seed_examples"))
    perturbations = _validate_perturbation_list(test_id, raw.get("perturbations"))
    expectation = raw.get("expectation") or {"kind": "same_label"}
    if not isinstance(expectation, dict):
        raise BehavioralTestValidationError(
            f"expectation_shape_invalid:{test_id}"
        )
    if str(expectation.get("kind") or "").strip().lower() != "same_label":
        # INV's only legal expectation is same_label — anything else is
        # operator confusion (probably wanted DIR). Surface explicitly.
        raise BehavioralTestValidationError(
            f"inv_expectation_must_be_same_label:{test_id}"
        )
    return {
        "seed_examples": seed_examples,
        "perturbations": perturbations,
        "expectation": {"kind": "same_label"},
        "n_perturbations_per_seed": _validate_per_seed_count(
            test_id, raw.get("n_perturbations_per_seed"),
        ),
    }


def _validate_dir_test(test_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    seed_examples = _validate_seed_examples(test_id, raw.get("seed_examples"))
    perturbations = _validate_perturbation_list(test_id, raw.get("perturbations"))
    expectation = raw.get("expectation")
    if not isinstance(expectation, dict):
        raise BehavioralTestValidationError(
            f"expectation_shape_invalid:{test_id}"
        )
    kind = str(expectation.get("kind") or "").strip().lower()
    if kind not in DIR_EXPECTATION_KINDS:
        raise BehavioralTestValidationError(
            f"unknown_dir_expectation_kind:{test_id}:{kind}"
        )
    cleaned_expectation: dict[str, Any] = {"kind": kind}
    if kind == "must_change_to":
        target = expectation.get("target_label")
        if not isinstance(target, str) or not target.strip():
            raise BehavioralTestValidationError(
                f"dir_target_label_required:{test_id}"
            )
        cleaned_expectation["target_label"] = target.strip()
    elif kind == "must_change_to_one_of":
        targets = expectation.get("target_labels")
        if not isinstance(targets, list) or not targets:
            raise BehavioralTestValidationError(
                f"dir_target_labels_required:{test_id}"
            )
        cleaned_targets: list[str] = []
        for label in targets:
            if not isinstance(label, str) or not label.strip():
                raise BehavioralTestValidationError(
                    f"dir_target_label_invalid:{test_id}"
                )
            cleaned_targets.append(label.strip())
        cleaned_expectation["target_labels"] = cleaned_targets
    return {
        "seed_examples": seed_examples,
        "perturbations": perturbations,
        "expectation": cleaned_expectation,
        "n_perturbations_per_seed": _validate_per_seed_count(
            test_id, raw.get("n_perturbations_per_seed"),
        ),
    }


def _validate_mft_test(test_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    examples_raw = raw.get("examples")
    if not isinstance(examples_raw, list) or not examples_raw:
        raise BehavioralTestValidationError(
            f"mft_examples_required:{test_id}"
        )
    if len(examples_raw) > MAX_MFT_EXAMPLES_PER_TEST:
        raise BehavioralTestValidationError(
            f"mft_examples_too_many:{test_id}"
        )
    cleaned: list[dict[str, Any]] = []
    for idx, entry in enumerate(examples_raw):
        if not isinstance(entry, dict):
            raise BehavioralTestValidationError(
                f"mft_example_shape_invalid:{test_id}:{idx}"
            )
        text = entry.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BehavioralTestValidationError(
                f"mft_example_input_required:{test_id}:{idx}"
            )
        if len(text) > MAX_INPUT_CHARS:
            raise BehavioralTestValidationError(
                f"mft_example_input_too_long:{test_id}:{idx}"
            )
        expected = entry.get("expected_label")
        if not isinstance(expected, str) or not expected.strip():
            raise BehavioralTestValidationError(
                f"mft_expected_label_required:{test_id}:{idx}"
            )
        cleaned.append({
            "input": text.strip(),
            "expected_label": expected.strip(),
        })
    return {"examples": cleaned}


def _validate_perturbation_list(
    test_id: str, raw: Any
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise BehavioralTestValidationError(
            f"perturbations_required:{test_id}"
        )
    if len(raw) > MAX_PERTURBATIONS_PER_TEST:
        raise BehavioralTestValidationError(
            f"perturbations_too_many:{test_id}"
        )
    return [_validate_perturbation(test_id, p) for p in raw]


def _validate_per_seed_count(test_id: str, raw: Any) -> int:
    if raw is None:
        return 1
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise BehavioralTestValidationError(
            f"n_perturbations_per_seed_invalid:{test_id}"
        )
    if raw < 1 or raw > 50:
        raise BehavioralTestValidationError(
            f"n_perturbations_per_seed_out_of_range:{test_id}"
        )
    return raw


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────


def validate_behavioral_tests(payload: Any) -> list[dict[str, Any]]:
    """Validate + normalize the ``behavioral_tests`` block on a task_spec.

    Returns the cleaned list ready to persist back into the pack
    contract. Raises ``BehavioralTestValidationError`` with a stable
    colon-delimited code on the first malformed entry so the editor
    can highlight the bad row.

    Empty / None / ``[]`` is a valid "no behavioral tests configured"
    state; the runner (slice 2) silently skips emitting any
    ``behavioral.*`` metrics in that case.
    """
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise BehavioralTestValidationError("behavioral_tests_shape_invalid")
    if len(payload) > MAX_TESTS_PER_PACK:
        raise BehavioralTestValidationError(
            f"behavioral_tests_too_many:{len(payload)}"
        )

    seen_ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise BehavioralTestValidationError(
                f"behavioral_test_shape_invalid:{idx}"
            )

        test_id_raw = raw.get("test_id")
        if not isinstance(test_id_raw, str) or not test_id_raw.strip():
            raise BehavioralTestValidationError(
                f"behavioral_test_id_required:{idx}"
            )
        test_id = test_id_raw.strip()
        if not _TEST_ID_RE.match(test_id):
            raise BehavioralTestValidationError(
                f"behavioral_test_id_invalid:{test_id}"
            )
        if test_id in seen_ids:
            raise BehavioralTestValidationError(
                f"duplicate_behavioral_test_id:{test_id}"
            )
        seen_ids.add(test_id)

        kind = str(raw.get("kind") or "").strip().upper()
        if kind not in BEHAVIORAL_TEST_KINDS:
            raise BehavioralTestValidationError(
                f"unknown_behavioral_test_kind:{test_id}:{kind}"
            )

        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise BehavioralTestValidationError(
                f"behavioral_test_description_invalid:{test_id}"
            )
        description = description.strip() if isinstance(description, str) else ""
        if description and len(description) > 512:
            raise BehavioralTestValidationError(
                f"behavioral_test_description_too_long:{test_id}"
            )

        # pass_rate_floor is metadata-only — the gate threshold in
        # task_specs.gates[] is authoritative. We validate the type +
        # range here so the Coach nudge (slice 3) can read it as a
        # default-threshold hint without a try/except.
        raw_floor = raw.get("pass_rate_floor")
        if raw_floor is None:
            pass_rate_floor = 0.85 if kind != "MFT" else 1.0
        else:
            if not isinstance(raw_floor, (int, float)) or isinstance(raw_floor, bool):
                raise BehavioralTestValidationError(
                    f"behavioral_test_pass_rate_floor_invalid:{test_id}"
                )
            pass_rate_floor = float(raw_floor)
            if not (0.0 <= pass_rate_floor <= 1.0):
                raise BehavioralTestValidationError(
                    f"behavioral_test_pass_rate_floor_out_of_range:{test_id}"
                )

        # Per-kind shape validation.
        if kind == "INV":
            kind_block = _validate_inv_test(test_id, raw)
        elif kind == "DIR":
            kind_block = _validate_dir_test(test_id, raw)
        else:  # MFT
            kind_block = _validate_mft_test(test_id, raw)

        cleaned.append({
            "test_id": test_id,
            "kind": kind,
            "description": description,
            "pass_rate_floor": pass_rate_floor,
            **kind_block,
        })
    return cleaned
