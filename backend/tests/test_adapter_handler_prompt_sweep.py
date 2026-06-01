"""Permanent regression guard for β-shape prompt-format gaps
(closes the cross-task audit thread).

A β-shape gap is the failure mode β/ζ/η/θ/ι/κ each closed for one
adapter: the data adapter writes ``source_text`` in one format,
but the eval handler (whose ``wraps_own_prompt() == True``) builds
a different prompt at eval. The model never sees the eval-time
scaffold; held-out metrics come in artificially low.

This sweep is dynamic over the registry, pinning two invariants:

  1. **Canonical-pair byte compatibility.** For each
     (adapter, canonical handler) pair the audit closed, the
     adapter's ``source_text`` must contain at least one of the
     handler's ``expected_prompt_prefixes()`` across every
     supported subtask. If a future change makes the adapter
     drop the wrap, the assertion fails.

  2. **Audit-coverage parity.** The set of wrap-own-prompt
     handler classes registered in
     ``eval_task_handler_service`` must equal the set the
     ``_CANONICAL_ADAPTER_HANDLERS`` map covers — exactly. Any
     new wrapping handler added later that doesn't have a
     canonical adapter pinned here makes this fail loudly,
     forcing the audit to extend (new adapter wrap + fixture)
     before the handler ships.

A diagnostic test ALSO records (without failing) any adapter
whose declared ``task_profiles`` resolve to a wrapping handler
OTHER than its canonical one — a different class of gap
("cross-pair drift") the audit didn't address, surfaced for
follow-up investigation.

If a new adapter or handler is added later that drifts into a
β-shape gap, this test fails at CI time — no future audit
needed.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.services.data_adapter_service import BUILTIN_ADAPTERS
from app.services.dataset_service import _ADAPTER_SUBTASK_SPECS
from app.services.eval_task_handler_service import (
    AudioTranscriptHandler,
    ClassificationHandler,
    RAGHandler,
    Seq2SeqHandler,
    StructuredExtractionHandler,
    VisionLanguageHandler,
    resolve_task_handler,
)


# ── Canonical pairs the audit closed ─────────────────────────────────
#
# Each entry pins the adapter's natural handler — the handler the
# β/ζ/η/θ/ι/κ commits each wrapped ``source_text`` against. A new
# wrapping adapter+handler must add an entry here AND a fixture
# below, otherwise ``test_wrapping_handler_universe_matches_audit_closure``
# fails loud (forcing the audit to extend before the new pair
# ships).
_CANONICAL_ADAPTER_HANDLERS: dict[str, type] = {
    "classification-label": ClassificationHandler,
    "structured-extraction": StructuredExtractionHandler,
    "rag-grounded": RAGHandler,
    "seq2seq-pair": Seq2SeqHandler,
    "vision-language-pair": VisionLanguageHandler,
    "audio-transcript": AudioTranscriptHandler,
}


# Per-adapter base fixture. Subtask-aware adapters get the same
# fixture for every subtask (the wrap differs, not the row shape)
# — the adapter_config picks the branch.
_ADAPTER_FIXTURES: dict[str, dict[str, Any]] = {
    "classification-label": {"text": "this is a test", "label": "positive"},
    "structured-extraction": {
        "text": "John works at Acme.",
        "structured_output": {"name": "John", "company": "Acme"},
    },
    "rag-grounded": {
        "question": "When was Acme founded?",
        "context": "Acme was founded in 1999.",
        "answer": "1999",
    },
    "seq2seq-pair": {
        "source": "Long article body here.",
        "target": "Short summary.",
    },
    "vision-language-pair": {
        "image_path": "imgs/sample.jpg",
        "caption": "A cat on a mat.",
        # Question for VQA branch; ignored by captioning.
        "question": "What animal is shown?",
        "answer": "A cat.",
    },
    "audio-transcript": {
        "audio_path": "audio/sample.wav",
        "transcript": "hello world",
        "question": "Who is speaking?",
        "answer": "A child.",
    },
}


_WRAPPING_HANDLER_TYPES: frozenset[type] = frozenset(
    _CANONICAL_ADAPTER_HANDLERS.values()
)


class AdapterHandlerPromptSweepTests(unittest.TestCase):
    """The audit-as-test. Canonical-pair byte-compat + audit
    coverage parity, dynamic over the registry."""

    def _assert_source_text_carries_prefix(
        self,
        *,
        adapter_id: str,
        handler: Any,
        adapter_config: dict[str, Any],
        subtask_label: str,
    ) -> None:
        fixture = dict(_ADAPTER_FIXTURES[adapter_id])
        adapter_entry = BUILTIN_ADAPTERS[adapter_id]
        map_row = adapter_entry["map_row"]
        out = map_row(fixture, adapter_config)
        self.assertIsNotNone(
            out,
            f"adapter {adapter_id!r} couldn't map fixture "
            f"({subtask_label}); fixture={fixture!r}",
        )
        source_text = out.get("source_text", "")
        prefixes = handler.expected_prompt_prefixes()
        carries = any(p in source_text for p in prefixes)
        self.assertTrue(
            carries,
            (
                f"β-shape gap detected: adapter {adapter_id!r} "
                f"({subtask_label}) wrote a source_text that doesn't "
                f"carry any of {handler.__class__.__name__}'s expected "
                f"prefixes {prefixes!r}. Eval-time prompt won't match "
                f"trainer-time prompt; held-out metrics will collapse. "
                f"Fix shape: have the adapter wrap source_text with "
                f"the same instruction prompt the handler builds at "
                f"eval (see β/ζ/η/θ/ι/κ commits for the template). "
                f"source_text={source_text!r}"
            ),
        )

    def test_every_canonical_pair_byte_compatible_across_all_subtasks(self):
        """Sweep the canonical pairs. For each, assert the adapter's
        ``source_text`` carries the canonical handler's prefix. For
        subtask-aware adapters (vision / audio / seq2seq), check
        every supported subtask — a regression that breaks only
        one branch still fails the sweep."""
        for adapter_id, handler_cls in _CANONICAL_ADAPTER_HANDLERS.items():
            self.assertIn(
                adapter_id, BUILTIN_ADAPTERS,
                f"canonical adapter {adapter_id!r} not in "
                f"BUILTIN_ADAPTERS — registry drift?",
            )
            self.assertIn(
                adapter_id, _ADAPTER_FIXTURES,
                f"canonical adapter {adapter_id!r} has no fixture — "
                f"add one so the sweep can pin this pair.",
            )
            handler = handler_cls()
            subtask_spec = _ADAPTER_SUBTASK_SPECS.get(adapter_id)
            if subtask_spec is None:
                self._assert_source_text_carries_prefix(
                    adapter_id=adapter_id,
                    handler=handler,
                    adapter_config={},
                    subtask_label="default",
                )
            else:
                for subtask in sorted(subtask_spec["allowed"]):
                    self._assert_source_text_carries_prefix(
                        adapter_id=adapter_id,
                        handler=handler,
                        adapter_config={"subtask": subtask},
                        subtask_label=f"subtask={subtask}",
                    )

    def test_subtask_aware_adapters_exercise_every_supported_subtask(self):
        """Regression guard for the sweep itself: every adapter in
        ``_ADAPTER_SUBTASK_SPECS`` must have a fixture sufficient
        for every supported subtask. If a new subtask is added to
        a handler and the adapter doesn't gain support, the sweep
        above wouldn't cover the new branch — this test fails
        loudly to force the addition."""
        for adapter_id, spec in _ADAPTER_SUBTASK_SPECS.items():
            self.assertIn(
                adapter_id, _ADAPTER_FIXTURES,
                f"adapter {adapter_id!r} in _ADAPTER_SUBTASK_SPECS "
                f"has no fixture; sweep can't cover its subtasks.",
            )
            self.assertIn(
                adapter_id, _CANONICAL_ADAPTER_HANDLERS,
                f"adapter {adapter_id!r} in _ADAPTER_SUBTASK_SPECS "
                f"has no canonical handler pinned — drift.",
            )
            adapter_entry = BUILTIN_ADAPTERS.get(adapter_id)
            self.assertIsNotNone(
                adapter_entry,
                f"adapter {adapter_id!r} in _ADAPTER_SUBTASK_SPECS "
                f"not in BUILTIN_ADAPTERS — stale entry?",
            )
            map_row = adapter_entry["map_row"]
            for subtask in sorted(spec["allowed"]):
                out = map_row(
                    dict(_ADAPTER_FIXTURES[adapter_id]), {"subtask": subtask},
                )
                self.assertIsNotNone(
                    out,
                    f"adapter {adapter_id!r} returned None for "
                    f"subtask={subtask!r} — fixture insufficient.",
                )
                self.assertTrue(
                    isinstance(out.get("source_text"), str) and out["source_text"],
                    f"adapter {adapter_id!r} subtask={subtask!r} did "
                    f"not produce a non-empty source_text. Fixture "
                    f"missing a field?",
                )

    def test_wrapping_handler_universe_matches_audit_closure(self):
        """Audit-coverage parity. The set of handler classes that
        declare ``wraps_own_prompt() == True`` and are reachable
        through the live registry must equal the set
        ``_CANONICAL_ADAPTER_HANDLERS`` pins. A new handler that
        wraps its own prompt MUST add (a) a canonical adapter
        below + (b) a fixture, OR this test fails — the audit
        must extend before the new handler ships."""
        from app.services.eval_task_handler_service import _HANDLER_FACTORIES
        wrap_classes: set[type] = set()
        for factory in _HANDLER_FACTORIES.values():
            try:
                handler = factory()
            except Exception:
                continue
            if (
                hasattr(handler, "wraps_own_prompt")
                and bool(handler.wraps_own_prompt())
            ):
                wrap_classes.add(handler.__class__)
        self.assertEqual(
            wrap_classes, set(_WRAPPING_HANDLER_TYPES),
            (
                "Set of wrap-own-prompt handler classes drifted. "
                f"Found={sorted(c.__name__ for c in wrap_classes)} "
                f"Expected={sorted(c.__name__ for c in _WRAPPING_HANDLER_TYPES)}. "
                "A new wrapping handler needs (a) a canonical adapter "
                "in ``_CANONICAL_ADAPTER_HANDLERS``, (b) a fixture in "
                "``_ADAPTER_FIXTURES``, and (c) byte-for-byte wrap "
                "in the adapter's ``_map_*`` against the handler's "
                "prompt format. Otherwise the sweep can't enforce "
                "byte-compat for the new pair and a β-shape gap "
                "can silently re-emerge."
            ),
        )

    def test_cross_pair_drift_is_documented_or_absent(self):
        """**Diagnostic** — surface (without failing) any adapter
        whose declared ``task_profiles`` resolve to a wrapping
        handler OTHER than its canonical one. The audit closed
        the canonical pairs; cross-pair gaps are a separate
        class of bug ("adapter X declares profile Y but X's
        source_text isn't shaped for handler Y") that this
        sweep can detect but didn't fix.

        We assert against a known-acknowledged list — if a NEW
        cross-pair drift appears, the test fails and the list
        needs to be updated (either by acknowledging the gap or
        by removing the offending profile from the adapter's
        declaration). This keeps the cross-pair surface visible
        without locking us into fixing every instance today.
        """
        # Known cross-pair drift the canonical audit didn't close.
        # Each entry is (adapter_id, foreign_handler_class).
        # Reason: the adapter declares a task_profile that
        # routes to a wrapping handler other than its canonical
        # one. A project picking that task_profile would hit a
        # β-shape-equivalent gap. Either (a) remove the profile
        # from the adapter's declaration, or (b) fix the adapter
        # to wrap conditionally. Both are out-of-scope for this
        # sweep — listed here as acknowledged debt.
        known_cross_pair_drift: set[tuple[str, str]] = {
            # Each entry: adapter declares a task_profile that
            # routes to a wrapping handler OTHER than its canonical
            # one. Out of scope for the β-shape audit; tracked as
            # acknowledged debt. Fix shape (for each entry) is one
            # of: (a) remove the foreign profile from the adapter's
            # declaration, or (b) extend the adapter to wrap
            # conditionally based on which handler will run.
            #
            # ``qa-pair → Seq2SeqHandler`` was resolved by removing
            # ``seq2seq`` from qa-pair's ``task_profiles`` declaration
            # (the legitimate "QA data trained as seq2seq" path is
            # already covered by ``seq2seq-pair`` adapter, whose
            # source_aliases include ``question`` and ``answer``).
            #
            # rag-grounded → Seq2SeqHandler: rag-grounded writes
            # the η-fixed ``Answer the question using only the
            # context…\nContext: …\nQuestion: …\nAnswer:``;
            # Seq2SeqHandler expects ``Translate/Summarize/
            # Paraphrase``. Any rag-grounded project picking
            # ``seq2seq`` profile hits this.
            ("rag-grounded", "Seq2SeqHandler"),
            # structured-extraction → ClassificationHandler: the
            # ζ-fixed ``Extract the following fields as JSON…``
            # wrap doesn't carry ``Classify the following text``.
            ("structured-extraction", "ClassificationHandler"),
            # structured-extraction → Seq2SeqHandler: same wrap;
            # doesn't carry seq2seq prefixes.
            ("structured-extraction", "Seq2SeqHandler"),
            # vision-language-pair → Seq2SeqHandler: ι-fixed
            # ``Describe the image…`` / ``Question:…Image:…Answer:``
            # doesn't carry seq2seq prefixes.
            ("vision-language-pair", "Seq2SeqHandler"),
            # audio-transcript → Seq2SeqHandler: κ-fixed
            # ``Transcribe the audio…`` / ``Question:…Audio:…Answer:``
            # doesn't carry seq2seq prefixes.
            ("audio-transcript", "Seq2SeqHandler"),
        }
        observed: set[tuple[str, str]] = set()
        for adapter_id, entry in BUILTIN_ADAPTERS.items():
            canonical_cls = _CANONICAL_ADAPTER_HANDLERS.get(adapter_id)
            task_profiles = entry.get("task_profiles") or []
            for profile in task_profiles:
                handler = resolve_task_handler(profile)
                if not (
                    hasattr(handler, "wraps_own_prompt")
                    and bool(handler.wraps_own_prompt())
                ):
                    continue
                if (
                    canonical_cls is not None
                    and isinstance(handler, canonical_cls)
                ):
                    continue
                observed.add((adapter_id, handler.__class__.__name__))
        new_drift = observed - known_cross_pair_drift
        resolved_drift = known_cross_pair_drift - observed
        msg_parts: list[str] = []
        if new_drift:
            msg_parts.append(
                "NEW cross-pair drift discovered (adapters that "
                "declare a task_profile resolving to a foreign "
                "wrapping handler): "
                f"{sorted(new_drift)}. Either remove the offending "
                "profile from the adapter's task_profiles or add "
                "the pair to ``known_cross_pair_drift`` as "
                "acknowledged debt."
            )
        if resolved_drift:
            msg_parts.append(
                "Acknowledged cross-pair drift is no longer "
                f"observed: {sorted(resolved_drift)}. Remove from "
                "``known_cross_pair_drift`` to keep the test tight."
            )
        if msg_parts:
            self.fail(" | ".join(msg_parts))


if __name__ == "__main__":
    unittest.main()
