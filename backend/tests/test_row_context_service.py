"""Per-row task context (``row_context_service``): normalisation + grouped reject reasons, readers, the
shared renderers, project flags applied to a row's own tool list, and -- the genericity check --
train/eval prompt parity for a row's own system message on QA, classification and tool calling.

pytest-only. Pure functions: no DB, no model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _row_context_golden import CASES, FakeTokenizer, NoTemplateTokenizer, _train_module, train_rows, training_render

from app.services import row_context_service as rcs
from app.services import tool_catalog_service as tcat


def tool(name: str, props: dict | None = None, **extra) -> dict:
    return {"name": name, "description": f"{name} tool",
            "inputSchema": {"type": "object", "properties": props or {"id": {"type": "string"}}}, **extra}


class NormalizeTests(unittest.TestCase):
    def test_no_context_is_none(self):
        self.assertEqual(rcs.normalize(None), (None, None))
        self.assertEqual(rcs.normalize({}), (None, None))
        self.assertEqual(rcs.validate_row({"question": "q"}), (None, None))

    def test_each_kind_normalises(self):
        out, reason = rcs.normalize({
            "system": "  Be terse.  ", "tools": [tool("b"), tool("a")], "output_schema": {"type": "object"},
            "labels": ["x", "y", "x"], "schema_text": "CREATE TABLE t (id int)",
        })
        self.assertIsNone(reason)
        self.assertEqual(out["system"], "Be terse.")
        self.assertEqual([t["name"] for t in out["tools"]], ["b", "a"])  # catalog form, in the row's order
        self.assertIn("input_schema", out["tools"][0])
        self.assertEqual(out["labels"], ["x", "y"])

    def test_empty_tool_list_is_kept(self):
        out, reason = rcs.normalize({"tools": []})
        self.assertIsNone(reason)
        self.assertEqual(out, {"tools": []})
        self.assertEqual(rcs.render_row_tool_block([]), "Available tools (call only these):\n- (no tools are allowed)")

    def test_reject_reasons(self):
        cases = {
            "row_context_invalid": "not an object",
            "row_context_unknown_kind": {"sytem": "typo"},
            "row_context_system_invalid": {"system": 3},
            "row_context_tools_invalid": {"tools": [tool("a"), tool("a")]},
            "row_context_schema_invalid": {"output_schema": {"type": 12}},
            "row_context_labels_invalid": {"labels": ["", 1]},
            "row_context_schema_text_invalid": {"schema_text": ["x"]},
        }
        for reason, raw in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(rcs.normalize(raw), (None, reason))
                self.assertIn(reason, rcs.REJECT_REASONS)
        bad_name, _ = rcs.normalize({"tools": [{"name": "has space"}]})
        self.assertIsNone(bad_name)


class ReaderTests(unittest.TestCase):
    def test_system_precedence(self):
        self.assertEqual(rcs.row_system({"row_context": {"system": "A"}, "system_prompt": "B"}), ("A", "row_context.system"))
        self.assertEqual(rcs.row_system({"system_prompt": "B"}), ("B", "system_prompt"))
        self.assertEqual(rcs.row_system({"question": "q"}), (None, None))

    def test_prepared_row_exposes_system_prompt_only_with_context(self):
        row = {"question": "q"}
        rcs.apply_to_prepared_row(row, None)
        self.assertEqual(row, {"question": "q"})
        rcs.apply_to_prepared_row(row, {"system": "S"})
        self.assertEqual(row["system_prompt"], "S")

    def test_consumers(self):
        self.assertTrue(rcs.consumes("system", "classification"))
        self.assertTrue(rcs.consumes("tools", "tool_calling"))
        self.assertFalse(rcs.consumes("tools", "qa"))
        self.assertFalse(rcs.consumes("labels", "classification"))  # not consumed yet: shown as unused

    def test_digest_is_stable(self):
        self.assertEqual(rcs.digest({"a": 1, "b": [1, 2]}), rcs.digest({"b": [1, 2], "a": 1}))


class ToolRenderTests(unittest.TestCase):
    def test_row_block_renders_like_the_project_block(self):
        tools = [tool("get_item"), tool("list_items", {})]
        normalized = rcs.normalize_tools(tools)
        state = tcat._new_state(normalized, {"kind": "manual"}, None, {}, "")
        self.assertEqual(rcs.render_row_tool_block(normalized), tcat.render_tool_block({"tool_catalog": state}))

    def test_project_flags_apply_by_name(self):
        project_tools = rcs.normalize_tools([tool("get_item"), tool("drop_item", annotations={"readOnlyHint": False})])
        state = tcat._new_state(project_tools, {"kind": "manual"}, None, {"get_item": {"allowed": False}}, "")
        overrides = rcs.project_tool_overrides({"tool_catalog": state})
        self.assertEqual(overrides["get_item"], {"allowed": False})
        self.assertTrue(overrides["drop_item"]["destructive"])
        row_tools = rcs.normalize_tools([tool("get_item"), tool("drop_item"), tool("other")])
        block = rcs.render_row_tool_block(row_tools, overrides)
        self.assertNotIn("get_item(", block)  # not allowed by the project: never offered
        self.assertIn("drop_item(id?: string) — drop_item tool [destructive]", block)  # stays destructive
        self.assertIn("other(", block)  # a tool the project does not know: the row's own
        view = rcs.tool_view_config(row_tools, overrides)
        allowed = {t["name"] for t in tcat.allowed_tools(view)}
        self.assertEqual(allowed, {"drop_item", "other"})

    def test_eval_view(self):
        row = {"row_context": {"system": "S", "tools": [tool("a")], "labels": ["x"]}}
        view = rcs.eval_view(row, profile="tool_calling")
        self.assertEqual(view["system"], "S")
        self.assertEqual(view["tools_count"], 1)
        self.assertEqual(view["tools_source"], "row")
        self.assertEqual(view["unused_kinds"], ["labels"])
        qa = rcs.eval_view(row, profile="qa")
        self.assertNotIn("tools", qa)
        self.assertEqual(qa["unused_kinds"], ["tools", "labels"])
        self.assertIsNone(rcs.eval_view({"question": "q"}, profile="qa"))

    def test_metrics_block_only_with_context(self):
        self.assertIsNone(rcs.eval_metrics_block([{"prediction": "x"}], profile="qa", project_has_catalog=False))
        block = rcs.eval_metrics_block(
            [{"row_context_view": {"system": "S", "digest": "d1"}}, {"prediction": "x"},
             {"row_context_view": {"tools_source": "row", "digest": "d2"}}],
            profile="tool_calling", project_has_catalog=True,
        )
        self.assertEqual(block["rows_with_context"], 2)
        self.assertEqual(block["kinds"]["system"], {"row": 1, "none": 2})
        self.assertEqual(block["kinds"]["tools"], {"row": 1, "project": 2, "none": 0})

    def test_row_list_renders_in_the_rows_order(self):
        tools = rcs.normalize_tools([tool("zeta"), tool("alpha"), tool("mid")])
        self.assertEqual([t["name"] for t in tools], ["zeta", "alpha", "mid"])
        block = rcs.render_row_tool_block(tools)
        self.assertEqual([ln.split("(")[0] for ln in block.splitlines()[1:]], ["- zeta", "- alpha", "- mid"])
        # "name": how splits prepared before the row order was kept rendered them
        legacy = rcs.render_row_tool_block(tools, order="name")
        self.assertEqual([ln.split("(")[0] for ln in legacy.splitlines()[1:]], ["- alpha", "- mid", "- zeta"])
        self.assertEqual(sorted(block.splitlines()), sorted(legacy.splitlines()))
        # one distinct list whatever the order
        self.assertEqual(rcs.tool_list_digest(tools), rcs.tool_list_digest(list(reversed(tools))))

    def test_project_catalog_still_sorts_by_name(self):
        project_tools, _ = tcat.normalize_tools([tool("zeta"), tool("alpha")], strict=True)
        self.assertEqual([t["name"] for t in project_tools], ["alpha", "zeta"])
        state = tcat._new_state(list(reversed(project_tools)), {"kind": "manual"}, None, {}, "")
        self.assertEqual(
            tcat.render_tool_block({"tool_catalog": state}),
            "Available tools (call only these):\n- alpha(id?: string) — alpha tool [mutating]\n"
            "- zeta(id?: string) — zeta tool [mutating]",
        )

    def test_prepared_row_tool_order(self):
        with tempfile.TemporaryDirectory() as d:
            prep = Path(d)
            self.assertEqual(rcs.prepared_row_tool_order({}, prep), "kept")  # nothing prepared
            (prep / "train.jsonl").write_text(json.dumps({"question": "q", "row_context": {"tools": [tool("a")]}}) + "\n")
            # per-row lists prepared before the order was kept (no stamp): they were rendered sorted
            self.assertEqual(rcs.prepared_row_tool_order({}, prep), "name")
            manifest: dict = {}
            rcs.stamp_row_tool_order(manifest, prep)
            self.assertEqual(manifest, {"row_tool_order": "kept"})
            self.assertEqual(rcs.prepared_row_tool_order(manifest, prep), "kept")
            # a split stamped "row" the same day (key order not yet kept) stays legacy
            self.assertEqual(rcs.prepared_row_tool_order({"row_tool_order": "row"}, prep), "row")
            (prep / "train.jsonl").write_text(json.dumps({"question": "q"}) + "\n")
            plain: dict = {}
            rcs.stamp_row_tool_order(plain, prep)
            self.assertEqual(plain, {})  # manifests without per-row lists are unchanged

    def test_fingerprint_marks_row_order_only(self):
        from app.services import tool_call_scoring as tcs

        before = tcs.tool_call_fingerprint({}, row_tool_lists=True)
        prints = {o: tcs.tool_call_fingerprint({}, row_tool_lists=True, row_tool_order=o) for o in ("kept", "row", "name")}
        self.assertNotIn(before, prints.values())  # every order marks the parameter-order fix
        self.assertEqual(len(set(prints.values())), 3)
        self.assertEqual(tcs.tool_call_fingerprint({}, row_tool_order="row"), tcs.tool_call_fingerprint({}))

    def test_split_has_row_tools(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text(json.dumps({"question": "q"}) + "\n")
            self.assertFalse(rcs.split_has_row_tools(p))
            p.write_text(json.dumps({"question": "q", "row_context": {"tools": [tool("a")]}}) + "\n")
            self.assertTrue(rcs.split_has_row_tools(p))
            # a prepared row stores its context as JSON text (escaped quotes)
            p.write_text(json.dumps({"question": "q", "row_context": rcs.encode({"tools": [tool("a")]})}) + "\n")
            self.assertTrue(rcs.split_has_row_tools(p))
        self.assertFalse(rcs.split_has_row_tools("/nonexistent/x.jsonl"))


class FormatVersionTests(unittest.TestCase):
    def test_lift_cache_never_reuses_another_prompt_format(self):
        from types import SimpleNamespace

        from app.services.post_training_eval_service import _baseline_scoring_matches

        v2 = SimpleNamespace(eval_type="exact_match", metrics={"prompt_format_version": rcs.EVAL_PROMPT_FORMAT_VERSION})
        v1 = SimpleNamespace(eval_type="exact_match", metrics={})
        self.assertTrue(_baseline_scoring_matches(v2, None))
        self.assertFalse(_baseline_scoring_matches(v1, None))
        perplexity = SimpleNamespace(eval_type="perplexity", metrics={})
        self.assertTrue(_baseline_scoring_matches(perplexity, None))

    def test_compare_blocks_only_when_system_rows_changed(self):
        from app.services.compare_row_evidence_service import prompt_format_difference

        self.assertIsNone(prompt_format_difference({"prompt_format_version": 2}, {"prompt_format_version": 2}))
        kind, msg = prompt_format_difference({"prompt_format_version": 1}, {"prompt_format_version": 2, "system_rows": 0})
        self.assertEqual(kind, "note")
        kind, msg = prompt_format_difference({"prompt_format_version": 1}, {"prompt_format_version": 2, "system_rows": 3})
        self.assertEqual(kind, "block")
        self.assertIn("evaluate A again", msg)


class TrainEvalParityTests(unittest.TestCase):
    """The row's own system message renders identically in training and eval, for QA (chat-wrapped),
    classification (a handler that builds its own prompt) and tool calling (with its own tool list)."""

    def test_train_py_copy_matches(self):
        train = _train_module()
        self.assertEqual(train._render_system_plain("S", "P"), rcs.render_system_plain("S", "P"))

    def _parity(self, name: str, rows: list[dict]) -> list[str]:
        from _row_context_golden import eval_render

        case = {**CASES[name], "rows": rows}
        prepared = training_render(case)
        self.assertEqual(len(prepared), len(rows))
        with tempfile.TemporaryDirectory() as d:
            pairs, texts = eval_render(case, prepared, Path(d))
        self.assertEqual(len(pairs), len(rows))
        trained_by_tok = [train_rows(prepared, FakeTokenizer()), train_rows(prepared, NoTemplateTokenizer())]
        for i in range(len(prepared)):
            for j in range(2):
                trained = trained_by_tok[j][i]
                eval_text = texts[i * 2 + j]
                if j == 0:
                    self.assertEqual(trained["source_text"], eval_text, f"{name} row {i} (chat template)")
                else:
                    # no chat template: training appends a newline after the prompt (as for every row)
                    self.assertEqual(trained["source_text"].rstrip("\n"), eval_text, f"{name} row {i} (no template)")
        return texts

    def test_qa_system_message(self):
        texts = self._parity("qa", [{"question": "Capital of France?", "answer": "Paris",
                                     "row_context": {"system": "Answer in one word."}}])
        self.assertIn("<system>Answer in one word.</system><user>Capital of France?</user>", texts[0])
        self.assertTrue(texts[1].startswith("Answer in one word.\n\n"))

    def test_legacy_system_prompt_field_is_rendered_in_eval_too(self):
        texts = self._parity("qa", [{"question": "Name a prime.", "answer": "7", "system_prompt": "Passages: 7."}])
        self.assertIn("<system>Passages: 7.</system>", texts[0])

    def test_classification_system_message(self):
        texts = self._parity("classification", [
            {"text": "I loved it", "label": "pos", "row_context": {"system": "You label reviews."}},
            {"text": "Terrible", "label": "neg"},  # a row without one in the same split: unchanged
        ])
        self.assertTrue(texts[2].startswith("Classify the following text"))
        self.assertTrue(texts[0].startswith("You label reviews.\n\nClassify the following text"))

    def test_tool_calling_system_and_own_tools(self):
        rows = [{"prompt": "Get item 7", "tool_name": "get_item", "tool_args": {"id": "7"}, "result": "ok",
                 "row_context": {"system": "You call tools.", "tools": [tool("get_item"), tool("zap")]}}]
        texts = self._parity("tool_calling", rows)
        self.assertIn("<system>You call tools.</system><user>Available tools (call only these):", texts[0])
        self.assertIn("- zap(", texts[0])  # the row's own list, not the project block
        self.assertNotIn("list_items", texts[0])
        prepared = training_render({**CASES["tool_calling"], "rows": rows})
        self.assertEqual(prepared[0]["user_prompt"], "Get item 7")
        self.assertLess(texts[0].index("- get_item("), texts[0].index("- zap("))
        self.assertEqual(prepared[0]["system_prompt"], "You call tools.")
        self.assertEqual(len(rcs.decode(prepared[0]["row_context"])["tools"]), 2)

    def test_tool_calling_own_tools_keep_the_rows_order(self):
        rows = [{"prompt": "Get item 7", "tool_name": "get_item", "tool_args": {"id": "7"}, "result": "ok",
                 "row_context": {"tools": [tool("zap"), tool("get_item"), tool("alpha")]}}]
        texts = self._parity("tool_calling", rows)
        order = [texts[0].index(f"- {n}(") for n in ("zap", "get_item", "alpha")]
        self.assertEqual(order, sorted(order))
        prepared = training_render({**CASES["tool_calling"], "rows": rows})
        self.assertEqual([t["name"] for t in rcs.decode(prepared[0]["row_context"])["tools"]], ["zap", "get_item", "alpha"])

    def test_train_py_leading_system_copy_matches(self):
        train = _train_module()
        for msgs in ([{"role": "user", "content": "u"}],
                     [{"role": "system", "content": "old"}, {"role": "user", "content": "u"}]):
            self.assertEqual(train._with_leading_system(msgs, "S"), rcs.with_leading_system(msgs, "S"))

    def _single_exchange(self, extra: dict) -> list[dict]:
        return [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}], **extra}]

    def test_single_exchange_chat_row_keeps_its_system_in_eval(self):
        # A chat row with one user->assistant exchange trains on the prompt/completion path WITH its
        # system turn; eval must render the same ``[system, user]`` prompt (it used to drop it).
        rows = self._single_exchange({"row_context": {"system": "Be brief."}})
        texts = self._parity("chat", rows)
        self.assertEqual(texts[0], "<system>Be brief.</system><user>Hi</user><assistant>")
        with tempfile.TemporaryDirectory() as d:
            from _row_context_golden import eval_render

            pairs, _ = eval_render(CASES["chat"], training_render({**CASES["chat"], "rows": rows}), Path(d))
        self.assertEqual(pairs[0]["row_system"], "Be brief.")
        self.assertEqual(pairs[0]["row_context_view"]["system_source"], "row_context.system")

    def test_single_exchange_chat_row_with_legacy_system_prompt(self):
        texts = self._parity("chat", self._single_exchange({"system_prompt": "Passages: x."}))
        self.assertTrue(texts[0].startswith("<system>Passages: x.</system><user>Hi</user>"))

    def _multi_turn_parity(self, rows: list[dict]) -> list[dict]:
        """Training's chat path (``chat_messages``) vs the eval pair's history, message for message and
        rendered with the chat template."""
        from _row_context_golden import eval_render

        train = _train_module()
        case = {**CASES["chat"], "rows": rows}
        prepared = training_render(case)
        contract = train._build_data_adapter_contract("causal_lm", "llama3")
        with tempfile.TemporaryDirectory() as d:
            pairs, texts = eval_render(case, prepared, Path(d))
        self.assertEqual(len(pairs), len(rows))
        for i, row in enumerate(prepared):
            mapped = train._adapt_record_to_text(dict(row), contract, "llama3")
            trained = json.loads(mapped[train.CHAT_MESSAGES_COLUMN])
            self.assertEqual(trained[-1]["role"], "assistant")
            self.assertEqual(trained[:-1], pairs[i]["history"], f"row {i}")
            rendered = FakeTokenizer().apply_chat_template(trained[:-1], add_generation_prompt=True)
            self.assertEqual(rendered, texts[i * 2], f"row {i}")
        return pairs

    def test_multi_turn_chat_row_context_system_is_the_leading_turn(self):
        turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                 {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Bye."}]
        pairs = self._multi_turn_parity([
            {"messages": turns, "row_context": {"system": "Be brief."}},
            # the row's own system turn is replaced: row_context.system wins
            {"messages": [{"role": "system", "content": "Old."}, *turns], "row_context": {"system": "Be brief."}},
            {"messages": turns},  # no row_context: unchanged
        ])
        self.assertEqual(pairs[0]["history"][0], {"role": "system", "content": "Be brief."})
        self.assertEqual(pairs[1]["history"][0], {"role": "system", "content": "Be brief."})
        self.assertEqual(sum(m["role"] == "system" for m in pairs[1]["history"]), 1)
        self.assertEqual(pairs[0]["dialogue_messages"][0]["content"], "Be brief.")
        self.assertEqual(pairs[0]["row_context_view"]["system_source"], "row_context.system")
        self.assertEqual(pairs[2]["history"][0]["role"], "user")
        self.assertNotIn("row_context_view", pairs[2])

    def test_multi_turn_chat_legacy_system_prompt_is_not_claimed(self):
        # Training's chat path never rendered the legacy field; eval does not, and says nothing about it.
        turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                 {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Bye."}]
        pairs = self._multi_turn_parity([{"messages": turns, "system_prompt": "Passages."}])
        self.assertEqual(pairs[0]["history"][0]["role"], "user")
        self.assertNotIn("row_system", pairs[0])
        self.assertNotIn("row_context_view", pairs[0])

    def test_legacy_system_prompt_on_a_handler_built_prompt_is_not_rendered(self):
        # Training passes these rows through (as before row_context); eval matches and claims nothing.
        from _row_context_golden import eval_render

        case = CASES["classification_legacy_system"]
        prepared = training_render(case)
        trained = train_rows(prepared, FakeTokenizer())
        with tempfile.TemporaryDirectory() as d:
            pairs, texts = eval_render(case, prepared, Path(d))
        self.assertTrue(trained[0]["source_text"].startswith("Classify the following text"))
        self.assertEqual(trained[0]["source_text"], texts[0])
        self.assertNotIn("row_system", pairs[0])
        self.assertNotIn("row_context_view", pairs[0])

    def test_rows_without_context_get_no_system(self):
        prepared = training_render(CASES["qa"])
        self.assertTrue(all("system_prompt" not in r and "row_context" not in r for r in prepared))


if __name__ == "__main__":
    unittest.main()
