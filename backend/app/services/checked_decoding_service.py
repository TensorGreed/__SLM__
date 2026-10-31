"""Checked decoding: verify-and-retry at eval time (opt-in, Epic Y1).

After the model answers a held-out row, cheap deterministic *verifiers* check
the answer using ONLY what the model saw (the prompt and context fields that
are rendered into it) and the answer itself. When one fails, the row is
retried up to ``max_attempts`` times:

* ``resample`` - the same formatted prompt, sampled at ``sample_temperature``
  with a seed fixed per (row, attempt, seed);
* ``repair``  - one follow-up user turn through the model's chat template,
  listing the failed checks, with the previous answer as the assistant turn.

The first passing attempt wins; when none passes, attempt 1 is kept (never an
answer picked by a score the verifiers did not compute from prompt + answer).
Attempt 1 is exactly the normal generation, so with the setting off (the
default) nothing changes, and with it on a row that passes is unchanged too.

Setting: ``project.runtime_config["inference"]["checked_decoding"]`` =
``{enabled, mode, max_attempts, verifiers, sample_temperature, seed}``
(``DEFAULTS``); ``POST .../evaluation/run-heldout`` and the lift check take a
per-eval override. The applied policy is recorded in the eval's decoding
record (``metrics.inference.decoding.checked_decoding``) so Compare and the
lift check's base-model cache can tell a checked eval from an unchecked one.

**The verifier interface is the safety rule.** A verifier receives a
``VerifierInput`` (built by ``build_verifier_input`` from an explicit
allow-list of prompt-visible fields) plus the raw answer. It never receives
the reference answer, the answer key, or any judge output; nothing in the
row dict reaches it except the allow-listed fields. A label set is passed
only when every label is visible in the prompt text.

Verifiers (by answer shape; none names a domain):

* ``code_checks``       - code answers (a code-example row with a declared
  language): the code-check registry's runnable checks for that language;
* ``context_grounding`` - the same code against the prompt text; opt-in even
  in auto mode: only when its id is in ``CODE_CHECK_ALLOWED``, in the project's
  code-check selection, or named in the policy's ``verifiers``
  (``with_grounding_opt_in``);
* ``json_schema``       - JSON answers when the eval declares an output
  schema (field types, null allowed for every field, the schema's own
  ``required`` keys present - none when it declares none);
* ``label_set``         - label answers: the whole answer (trimmed, casefolded,
  surrounding quotes / punctuation stripped) is one of the labels the prompt
  explicitly enumerates (the classification handler's candidate list);
* ``citations``         - every passage id the answer cites is in the prompt.

A row with no applicable verifier is not retried (``skipped: "no_verifiers"``).
A retry generation that raises (CUDA out of memory, say) stops the row's
retries and keeps attempt 1 (``attempts_detail`` names the error).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Mapping

CONFIG_KEY = "checked_decoding"
INFERENCE_KEY = "inference"
MODES = ("resample", "repair")
VERIFIER_IDS = ("code_checks", "context_grounding", "json_schema", "label_set", "citations")
GROUNDING_CHECK_ID = "context_grounding"
# Set on a resolved policy (never saved / validated) by ``with_grounding_opt_in``.
GROUNDING_OPT_IN_KEY = "grounding_opt_in"
MAX_ATTEMPTS_LIMIT = 5
MAX_MESSAGES_PER_FAILURE = 3
MAX_MESSAGE_CHARS = 300

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "resample",
    "max_attempts": 3,
    "verifiers": None,  # None = every verifier the answer shape allows
    "sample_temperature": 0.7,
    "seed": 0,
}

REPAIR_TEMPLATE = (
    "Your previous answer failed these automatic checks:\n{failures}\n"
    "Return a corrected answer in the same format."
)


class PolicyError(ValueError):
    """An invalid checked-decoding setting (the API answers 422)."""


# ── policy ────────────────────────────────────────────────────────────────


def normalize_policy(raw: Any) -> dict[str, Any]:
    """The full policy dict (defaults filled in); ``PolicyError`` on a bad value.
    ``None`` / ``{}`` = defaults (off)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise PolicyError("checked_decoding must be an object")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise PolicyError(f"checked_decoding: unknown field(s) {', '.join(unknown)}")
    out = dict(DEFAULTS)
    if "enabled" in raw and raw["enabled"] is not None:
        if not isinstance(raw["enabled"], bool):
            raise PolicyError("checked_decoding.enabled must be true or false")
        out["enabled"] = raw["enabled"]
    if raw.get("mode") is not None:
        if raw["mode"] not in MODES:
            raise PolicyError(f"checked_decoding.mode must be one of: {', '.join(MODES)}")
        out["mode"] = raw["mode"]
    if raw.get("max_attempts") is not None:
        value = raw["max_attempts"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_ATTEMPTS_LIMIT:
            raise PolicyError(f"checked_decoding.max_attempts must be an integer from 1 to {MAX_ATTEMPTS_LIMIT}")
        out["max_attempts"] = value
    if raw.get("verifiers") is not None:
        value = raw["verifiers"]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise PolicyError("checked_decoding.verifiers must be a list of verifier ids or null")
        bad = sorted({v for v in value if v not in VERIFIER_IDS})
        if bad:
            raise PolicyError(
                f"checked_decoding.verifiers: unknown id(s) {', '.join(bad)}; known: {', '.join(VERIFIER_IDS)}"
            )
        out["verifiers"] = [v for v in VERIFIER_IDS if v in value]  # canonical order, no duplicates
    if raw.get("sample_temperature") is not None:
        value = raw["sample_temperature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 2:
            raise PolicyError("checked_decoding.sample_temperature must be above 0 and at most 2")
        out["sample_temperature"] = float(value)
    if raw.get("seed") is not None:
        value = raw["seed"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PolicyError("checked_decoding.seed must be a non-negative integer")
        out["seed"] = value
    return out


def project_policy(runtime_config: Any) -> dict[str, Any]:
    """The project's saved setting (defaults when unset or unreadable)."""
    rc = runtime_config if isinstance(runtime_config, Mapping) else {}
    inference = rc.get(INFERENCE_KEY) if isinstance(rc.get(INFERENCE_KEY), Mapping) else {}
    try:
        return normalize_policy(inference.get(CONFIG_KEY))
    except PolicyError:
        return dict(DEFAULTS)


def with_project_policy(runtime_config: Any, raw: Any) -> dict[str, Any]:
    """A copy of ``runtime_config`` with the validated policy saved (``PolicyError`` on a bad one)."""
    policy = normalize_policy(raw)
    rc = dict(runtime_config) if isinstance(runtime_config, Mapping) else {}
    inference = dict(rc.get(INFERENCE_KEY)) if isinstance(rc.get(INFERENCE_KEY), Mapping) else {}
    inference[CONFIG_KEY] = policy
    rc[INFERENCE_KEY] = inference
    return rc


def merged_project_policy(runtime_config: Any, partial: Any) -> dict[str, Any]:
    """The project's saved setting with ``partial`` merged over it (the fields it names
    replace the saved ones), validated (``PolicyError`` on a bad merged value)."""
    if not isinstance(partial, Mapping):
        raise PolicyError("checked_decoding must be an object")
    return normalize_policy({**project_policy(runtime_config), **dict(partial)})


def resolve_policy(runtime_config: Any, override: Any = None) -> dict[str, Any] | None:
    """The policy an eval runs under: the per-eval ``override`` when given
    (it replaces the project setting), else the project setting. None = off."""
    policy = normalize_policy(override) if override is not None else project_policy(runtime_config)
    return policy if policy["enabled"] else None


def lift_policy_override(runtime_config: Any, override: Any = None) -> dict[str, Any]:
    """The explicit policy a lift check passes to EVERY eval it runs (base and
    fine-tuned): resolved once, so a setting saved mid-check can't split them.
    Always a full dict (``{"enabled": False, ...}`` when off)."""
    policy = resolve_policy(runtime_config, override)
    return policy if policy is not None else dict(DEFAULTS)


def policy_record(policy: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the decoding record carries (None = off): every knob that can
    change an answer. The sampling knobs only matter for resample."""
    if not policy or not policy.get("enabled"):
        return None
    rec: dict[str, Any] = {
        "mode": policy["mode"],
        "max_attempts": int(policy["max_attempts"]),
        "verifiers": list(policy["verifiers"]) if policy.get("verifiers") is not None else "auto",
    }
    if policy.get("verifiers") is None and policy.get(GROUNDING_OPT_IN_KEY):
        rec["auto_includes_grounding"] = True  # auto mode with context_grounding opted in
    if policy["mode"] == "resample":
        rec["sample_temperature"] = float(policy["sample_temperature"])
        rec["seed"] = int(policy["seed"])
    return rec


def policy_label(record: Mapping[str, Any] | None) -> str:
    """"Checked decoding · up to 3 tries · resample" / "no checked decoding"."""
    if not record:
        return "no checked decoding"
    return f"Checked decoding · up to {record.get('max_attempts')} tries · {record.get('mode')}"


def attempt_seed(seed: int, row_index: Any, attempt: int) -> int:
    """Deterministic sampling seed per (row, attempt, policy seed)."""
    digest = hashlib.sha256(f"{int(seed)}:{row_index}:{int(attempt)}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


# ── verifier input (the allow-list) ──────────────────────────────────────


@dataclass(frozen=True)
class VerifierInput:
    """Everything a verifier may see besides the answer. Built only by
    ``build_verifier_input``; there is deliberately no field for a reference,
    an answer key or a judge verdict."""

    prompt: str
    code_language: str | None = None
    passage_ids: tuple[str, ...] = ()
    label_set: tuple[str, ...] = ()
    json_schema: Mapping[str, Any] | None = field(default=None, hash=False)


def _prompt_text(pair: Mapping[str, Any]) -> str:
    """The prompt string plus any chat history the model sees (roles + content)."""
    parts: list[str] = []
    history = pair.get("history")
    if isinstance(history, list):
        for m in history:
            if isinstance(m, Mapping) and isinstance(m.get("content"), str):
                parts.append(m["content"])
    prompt = pair.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)
    return "\n".join(parts)


def _passage_ids(block: Any) -> tuple[str, ...]:
    if not isinstance(block, Mapping):
        return ()
    from app.services.code_remediation_prompt import normalize_passages

    return tuple(p["id"] for p in normalize_passages(block.get("passages")))


def _prompt_enumerates(prompt: Any, candidates: list[str]) -> bool:
    """True when ``prompt`` carries the classification handler's rendering of exactly these
    candidates (``candidate_list_line``)."""
    from app.services.eval_task_handler_service import ClassificationHandler

    line = ClassificationHandler.candidate_list_line(candidates)
    return bool(line) and isinstance(prompt, str) and line in prompt


def build_verifier_input(pair: Mapping[str, Any], eval_spec: Mapping[str, Any] | None = None) -> VerifierInput:
    """The ``VerifierInput`` for one held-out pair. Reads ONLY these keys:

    * ``prompt`` / ``history`` - the text the model is given;
    * ``code_example_row.language`` / ``.passages`` - rendered into a code-example prompt;
    * ``plan_row.passages`` - rendered into a plan prompt (``acceptable`` is NOT read);
    * ``classification_candidates`` - kept only when the prompt explicitly enumerates them
      (the classification handler's candidate-list line; never when there are more
      candidates than the prompt lists, or the labels merely appear in the input text);
    * ``eval_spec["json_schema"]`` - the eval's declared output schema (project config).

    ``reference`` and every other key are never read."""
    prompt = _prompt_text(pair)
    code_language: str | None = None
    passage_ids: tuple[str, ...] = ()
    ce = pair.get("code_example_row")
    if isinstance(ce, Mapping):
        lang = ce.get("language")
        code_language = lang.strip() if isinstance(lang, str) and lang.strip() else None
        passage_ids = _passage_ids(ce)
    if not passage_ids:
        passage_ids = _passage_ids(pair.get("plan_row"))
    labels: tuple[str, ...] = ()
    cands = pair.get("classification_candidates")
    if isinstance(cands, list):
        clean = [str(c) for c in cands if isinstance(c, str) and c.strip()]
        if clean and _prompt_enumerates(pair.get("prompt"), clean):
            labels = tuple(clean)
    schema = (eval_spec or {}).get("json_schema")
    return VerifierInput(
        prompt=prompt,
        code_language=code_language,
        passage_ids=passage_ids,
        label_set=labels,
        json_schema=schema if isinstance(schema, Mapping) else None,
    )


_STANDARD_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}


def output_json_schema(manifest: Any) -> dict[str, Any] | None:
    """A JSON Schema from the eval's DECLARED output schema (``manifest.output_schema``
    with ``properties``; plain field extraction only), or None. Structural: each
    field's declared standard type (null always allowed: an absent value is not a
    format error) and the keys the schema itself marks ``required`` (none when it declares
    none). Never derived from reference answers."""
    schema = (manifest or {}).get("output_schema") if isinstance(manifest, Mapping) else None
    if not isinstance(schema, Mapping):
        return None
    mode = str(schema.get("scoring_mode") or "").strip().lower()
    if mode not in ("", "field_match"):
        return None
    props = schema.get("properties")
    if not isinstance(props, Mapping) or not props:
        return None
    out_props: dict[str, Any] = {}
    for name, spec in props.items():
        entry: dict[str, Any] = {}
        declared = spec.get("type") if isinstance(spec, Mapping) else None
        types = [declared] if isinstance(declared, str) else declared if isinstance(declared, list) else []
        types = [t for t in types if t in _STANDARD_TYPES]
        if types:
            entry["type"] = sorted(set(types) | {"null"})
        out_props[str(name)] = entry
    required = schema.get("required")
    # The schema's own ``required`` only; none declared = nothing required.
    req = [str(r) for r in required if str(r) in out_props] if isinstance(required, list) else []
    return {"type": "object", "properties": out_props, "required": req}


# ── verifiers ─────────────────────────────────────────────────────────────


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Lenient JSON-object parse (fences stripped, first balanced ``{...}``) — the
    same reading the structured scorers use."""
    from app.services.eval_task_handler_service import StructuredExtractionHandler

    parsed = StructuredExtractionHandler()._parse_json_safely(text)
    return parsed if isinstance(parsed, dict) else None


def _result(verifier_id: str, ok: bool | None, messages: list[str] | None = None) -> dict[str, Any]:
    return {"verifier_id": verifier_id, "ok": ok, "messages": [str(m) for m in (messages or [])]}


def _code_of(answer: str) -> str | None:
    parsed = parse_json_object(answer)
    code = parsed.get("code") if parsed else None
    return code if isinstance(code, str) else None


def _check_failures(report: Mapping[str, Any]) -> list[str]:
    msgs: list[str] = []
    for r in report.get("results") or []:
        if r.get("ok") is False:
            for m in r.get("messages") or [f"{r.get('check_id')} failed"]:
                msgs.append(f"{r.get('check_id')}: {m}")
    return msgs


def _code_check_ids(language: str) -> list[str]:
    from app.services import code_check_service as ccs

    return [c.id for c in ccs.checks_for_language(language) if c.id != GROUNDING_CHECK_ID]


def verify_code_checks(vin: VerifierInput, answer: str) -> dict[str, Any]:
    from app.services import code_check_service as ccs

    code = _code_of(answer)
    if code is None:
        return _result("code_checks", False, ['the answer is not a JSON object with a "code" string'])
    ids = _code_check_ids(vin.code_language or "")
    if not ids:
        return _result("code_checks", None, ["no runnable check for this language"])
    report = ccs.run_code_checks(
        code, language=vin.code_language, check_ids=ids, context={"path": None, "grounding_text": vin.prompt}
    )
    return _result("code_checks", report.get("ok"), _check_failures(report))


def verify_context_grounding(vin: VerifierInput, answer: str) -> dict[str, Any]:
    from app.services import code_check_service as ccs

    code = _code_of(answer)
    if code is None:
        return _result(GROUNDING_CHECK_ID, None, ["no code to check"])
    report = ccs.run_code_checks(
        code, language=vin.code_language, check_ids=[GROUNDING_CHECK_ID],
        context={"path": None, "grounding_text": vin.prompt},
    )
    return _result(GROUNDING_CHECK_ID, report.get("ok"), _check_failures(report))


def verify_json_schema(vin: VerifierInput, answer: str) -> dict[str, Any]:
    from app.services import code_check_service as ccs

    parsed = parse_json_object(answer)
    if parsed is None:
        return _result("json_schema", False, ["the answer is not a JSON object"])
    report = ccs.run_code_checks(
        json.dumps(parsed), language="json", check_ids=["json_schema"], context={"json_schema": dict(vin.json_schema or {})}
    )
    return _result("json_schema", report.get("ok"), _check_failures(report))


_LABEL_EDGE = " \t\r\n\"'`“”‘’.,;:!?()[]{}<>*"


def _normalize_label(text: Any) -> str:
    """Trimmed, casefolded, surrounding quotes / punctuation stripped."""
    return str(text or "").strip().strip(_LABEL_EDGE).casefold()


def verify_label_set(vin: VerifierInput, answer: str) -> dict[str, Any]:
    """Passes when the WHOLE normalised answer is one of the labels (``"Alpha."`` and
    ``'"alpha"'`` pass; ``"alpha or beta"`` and ``"not alpha"`` do not)."""
    text = _normalize_label(answer)
    if text and any(text == _normalize_label(label) for label in vin.label_set):
        return _result("label_set", True)
    shown = ", ".join(vin.label_set[:12]) + (" ..." if len(vin.label_set) > 12 else "")
    return _result("label_set", False, [f"the answer is not exactly one of the allowed labels: {shown}"])


def verify_citations(vin: VerifierInput, answer: str) -> dict[str, Any]:
    parsed = parse_json_object(answer)
    if parsed is None or "citations" not in parsed:
        return _result("citations", None)
    cited = parsed.get("citations")
    if not isinstance(cited, list):
        return _result("citations", False, ['"citations" must be a list of passage ids'])
    unknown = [str(c) for c in cited if str(c) not in vin.passage_ids]
    if unknown:
        return _result(
            "citations", False,
            [f"cited id(s) not among the passages ({', '.join(vin.passage_ids)}): {', '.join(unknown[:5])}"],
        )
    return _result("citations", True)


Verifier = Callable[[VerifierInput, str], dict[str, Any]]

_VERIFIERS: dict[str, Verifier] = {
    "code_checks": verify_code_checks,
    "context_grounding": verify_context_grounding,
    "json_schema": verify_json_schema,
    "label_set": verify_label_set,
    "citations": verify_citations,
}


def _grounding_registered() -> bool:
    from app.services import code_check_service as ccs

    return ccs.get_code_check(GROUNDING_CHECK_ID) is not None


def grounding_opted_in(policy: Mapping[str, Any], *, runtime_config: Any = None, manifest: Any = None) -> bool:
    """``context_grounding`` is opt-in even in auto mode: True only when its id is named in
    the policy's ``verifiers``, listed in ``settings.CODE_CHECK_ALLOWED``, or in the project's
    code-check selection (``runtime_config["remediation"]["checks"]``, or the eval manifest's
    ``output_schema.remediation.checks`` the live selection is stamped into)."""
    wanted = policy.get("verifiers") if isinstance(policy, Mapping) else None
    if isinstance(wanted, list) and GROUNDING_CHECK_ID in wanted:
        return True
    from app.services import code_check_service as ccs

    if GROUNDING_CHECK_ID in ccs.allowed_check_ids():
        return True

    def _checks(block: Any) -> Any:
        rem = block.get("remediation") if isinstance(block, Mapping) else None
        return rem.get("checks") if isinstance(rem, Mapping) else None

    schema = manifest.get("output_schema") if isinstance(manifest, Mapping) else None
    for checks in (_checks(runtime_config), _checks(schema)):
        if isinstance(checks, (list, tuple)) and GROUNDING_CHECK_ID in checks:
            return True
    return False


def with_grounding_opt_in(
    policy: Mapping[str, Any] | None, *, runtime_config: Any = None, manifest: Any = None
) -> dict[str, Any] | None:
    """A copy of the resolved ``policy`` carrying whether ``context_grounding`` is opted in
    (``grounding_opted_in``); None stays None."""
    if policy is None:
        return None
    return {**policy, GROUNDING_OPT_IN_KEY: grounding_opted_in(policy, runtime_config=runtime_config, manifest=manifest)}


def applicable_verifiers(vin: VerifierInput | None, policy: Mapping[str, Any]) -> list[str]:
    """Verifier ids that apply to this row's answer shape, filtered by the policy's list.
    ``context_grounding`` joins auto mode only when opted in (``with_grounding_opt_in``)."""
    if vin is None:
        return []
    wanted = policy.get("verifiers")
    ids: list[str] = []
    if vin.code_language:
        ids.append("code_checks")
        opted_in = policy.get(GROUNDING_OPT_IN_KEY) is True if wanted is None else GROUNDING_CHECK_ID in wanted
        if opted_in and _grounding_registered():
            ids.append(GROUNDING_CHECK_ID)
    if vin.json_schema:
        ids.append("json_schema")
    if vin.label_set:
        ids.append("label_set")
    if vin.passage_ids:
        ids.append("citations")
    return ids if wanted is None else [v for v in ids if v in wanted]


def run_verifiers(vin: VerifierInput, answer: str, verifier_ids: list[str]) -> list[dict[str, Any]]:
    """Each verifier's result; a crashing verifier counts as skipped (``ok`` None)."""
    out: list[dict[str, Any]] = []
    for vid in verifier_ids:
        fn = _VERIFIERS.get(vid)
        if fn is None:
            continue
        try:
            res = fn(vin, answer)
        except Exception as exc:  # noqa: BLE001 - a broken verifier never fails the eval
            res = _result(vid, None, [f"verifier failed to run: {type(exc).__name__}"])
        out.append(res)
    return out


def _failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "verifier_id": r["verifier_id"],
            "messages": [str(m)[:MAX_MESSAGE_CHARS] for m in (r.get("messages") or [])][:MAX_MESSAGES_PER_FAILURE],
        }
        for r in results
        if r.get("ok") is False
    ]


def repair_message(failures: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for f in failures:
        msgs = f.get("messages") or ["failed"]
        for m in msgs:
            lines.append(f"- {f['verifier_id']}: {m}")
    return REPAIR_TEMPLATE.format(failures="\n".join(lines))


# ── the retry loop ────────────────────────────────────────────────────────

# ``regenerate(attempt_number, previous_attempt, previous_failures)`` returns the
# next attempt; an attempt is ``{"text", "tokens", "seconds"}`` (+ optional "mode").
# When it raises, the row's retries stop and attempt 1 is kept.
Regenerate = Callable[[int, dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def run_checked_row(
    *,
    policy: Mapping[str, Any],
    vin: VerifierInput | None,
    first: dict[str, Any],
    regenerate: Regenerate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(chosen attempt, row record)``. ``first`` is attempt 1 (the normal generation).

    A ``regenerate`` that raises ends the retries: ``{"attempt": n, "error": <type name>}`` goes
    into ``attempts_detail`` and attempt 1 is kept (no attempt passed, since the loop stops at the
    first pass); a CUDA out-of-memory error also frees the CUDA cache. When a later attempt is
    chosen, its ``generated_tokens`` / ``latency_ms`` are ``record["chosen_stats"]`` (the row's
    top-level fields stay attempt 1's)."""
    verifier_ids = applicable_verifiers(vin, policy)
    if not verifier_ids:
        return first, {
            "attempts": 1, "chosen_attempt": 1, "attempts_detail": [], "verifiers": [], "skipped": "no_verifiers",
        }
    assert vin is not None
    detail: list[dict[str, Any]] = []
    attempts = [first]
    verify_seconds = 0.0
    chosen_index: int | None = None
    max_attempts = int(policy.get("max_attempts") or 1)
    modes_used: list[str] = []
    current = first
    retry_error: str | None = None
    for n in range(1, max_attempts + 1):
        if n > 1:
            try:
                current = regenerate(n, attempts[-1], detail[-1]["failed"])
            except Exception as exc:  # noqa: BLE001 - a failed retry never fails the eval
                _free_cuda_cache_after(exc)
                retry_error = type(exc).__name__
                detail.append({"attempt": n, "error": retry_error})
                break
            attempts.append(current)
            modes_used.append(str(current.get("mode") or policy.get("mode")))
        started = perf_counter()
        results = run_verifiers(vin, str(current.get("text") or ""), verifier_ids)
        verify_seconds += perf_counter() - started
        failed = _failures(results)
        detail.append({"attempt": n, "failed": failed})
        if not failed:
            chosen_index = n
            break
    record: dict[str, Any] = {
        "attempts": len(attempts),
        "chosen_attempt": chosen_index or 1,
        "attempts_detail": detail,
        "verifiers": verifier_ids,
        "passed": chosen_index is not None,
        "extra_generation_seconds": round(sum(float(a.get("seconds") or 0.0) for a in attempts[1:]), 4),
        "extra_generated_tokens": int(sum(int(a.get("tokens") or 0) for a in attempts[1:])),
        "verify_seconds": round(verify_seconds, 4),
    }
    if modes_used and any(m != policy.get("mode") for m in modes_used):
        record["modes_used"] = modes_used
    if retry_error is not None:
        record["retry_error"] = retry_error
    chosen = attempts[(chosen_index or 1) - 1]
    if chosen_index is not None and chosen_index > 1:
        record["chosen_stats"] = {
            "generated_tokens": int(chosen.get("tokens") or 0),
            "latency_ms": round(float(chosen.get("seconds") or 0.0) * 1000, 3),
        }
    return chosen, record


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _free_cuda_cache_after(exc: BaseException) -> None:
    """After a CUDA out-of-memory error: release the cached blocks (guarded)."""
    if not _is_cuda_oom(exc):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def summarize(records: list[dict[str, Any] | None], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Eval-level metrics for the rows a policy ran over."""
    rows = [r for r in records if isinstance(r, dict)]
    checked = [r for r in rows if not r.get("skipped")]
    retried = [r for r in checked if r["attempts"] > 1]
    first_failed = [r for r in checked if r["attempts_detail"] and r["attempts_detail"][0]["failed"]]
    fixed = [r for r in first_failed if r.get("passed") and r["chosen_attempt"] > 1]
    still = [r for r in first_failed if not r.get("passed")]
    rec = policy_record(policy)
    return {
        "policy": rec,
        "label": policy_label(rec),
        "rows": len(rows),
        "rows_checked": len(checked),
        "rows_without_verifiers": len(rows) - len(checked),
        "rows_failed_first_try": len(first_failed),
        "rows_retried": len(retried),
        "rows_fixed_by_retry": len(fixed),
        "rows_still_failing": len(still),
        "mean_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 4) if rows else 0.0,
        "extra_generation_seconds": round(sum(float(r.get("extra_generation_seconds") or 0.0) for r in rows), 3),
        "extra_generated_tokens": int(sum(int(r.get("extra_generated_tokens") or 0) for r in rows)),
        "verify_seconds": round(sum(float(r.get("verify_seconds") or 0.0) for r in rows), 3),
    }


def checked_decoding_summary(metrics: Any) -> dict[str, Any] | None:
    """The short view of an eval's ``metrics["checked_decoding"]`` for cards (summary, lift,
    Compare): label + row counts; None when the eval ran without checked decoding."""
    block = metrics.get("checked_decoding") if isinstance(metrics, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    keys = ("policy", "label", "skipped", "rows", "rows_checked", "rows_retried", "rows_fixed_by_retry",
            "rows_still_failing", "mean_attempts", "extra_generation_seconds")
    return {k: block[k] for k in keys if k in block}


def with_prompt(vin: VerifierInput | None, prompt_seen: str) -> VerifierInput | None:
    """The input with the exact prompt text the model saw (chat template applied)."""
    if vin is None or not prompt_seen:
        return vin
    return replace(vin, prompt=prompt_seen)


__all__ = [
    "DEFAULTS", "MODES", "VERIFIER_IDS", "PolicyError", "VerifierInput", "applicable_verifiers", "attempt_seed",
    "build_verifier_input", "grounding_opted_in", "merged_project_policy", "normalize_policy", "output_json_schema", "policy_label", "policy_record",
    "project_policy", "repair_message", "resolve_policy", "run_checked_row", "run_verifiers", "summarize",
    "with_grounding_opt_in", "with_project_policy", "with_prompt",
]
