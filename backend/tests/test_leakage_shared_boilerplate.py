"""Split-leakage / dedup matcher with shared boilerplate: when most rows carry the same long block (a
tool list, a system prompt), that block is ignored, so distinct requests are no longer flagged as
near-duplicates of each other, while real duplicates and near-duplicates of the request are still
caught. Short shared headers and small corpora compare exactly as before.

Pure functions; no app, no DB."""

from __future__ import annotations

import unittest

from app.services import data_health_service as dh
from app.services.dataset_service import _dedup_entries_for_split

# A catalog large enough that its tokens dominate every row (as a real tool list does).
BLOCK = "Available tools (call only these):\n" + "\n".join(
    f"- tool_{i}(id: string, scope{i}?: string, filter{i}?: string) [read-only] Looks up kind{i} records by area{i}"
    for i in range(40)
)
REQUESTS = [
    "Show me the open invoices for the northern warehouse",
    "Count how many tickets were closed yesterday afternoon",
    "List every user who signed in from a new device",
    "Find the shipment that left the depot on Friday",
    "Which printer reported a paper jam this morning",
    "Give me the latest temperature reading from sensor nine",
    "Fetch the onboarding checklist for the design team",
    "What is the current balance on the travel budget",
    "Pull the release notes for version four point two",
    "Who approved the purchase order for new laptops",
    "Summarise the error rates for the payments service",
    "Get the contact details for the facilities manager",
    "List the meetings booked in room twelve next week",
    "How many seats are left on the evening training",
    "Find the incident report about the network outage",
    "Show the backlog items tagged as high priority",
    "Which servers are scheduled for patching tonight",
    "Retrieve the signed contract for the cleaning vendor",
    "What were the top search queries on the help site",
    "Look up the warranty status for asset number seven",
    "List the courses a new hire has not finished yet",
    "Find the budget line for the spring marketing push",
    "Show the overtime hours logged by the night shift",
    "Which repositories have failing builds right now",
]


def row(request: str) -> dict:
    return {"question": f"{BLOCK}\n\n{request}", "answer": '{"tool": "tool_1", "args": {}}'}


class SharedBoilerplateTests(unittest.TestCase):
    def test_distinct_requests_with_a_shared_tool_block_do_not_leak(self):
        train = [row(r) for r in REQUESTS[:16]]
        test = [row(r) for r in REQUESTS[16:]]
        # Without the stripping every test row is a "near-duplicate" of some train row.
        index_plain = dh._build_leakage_index(train)
        plain_hits = sum(1 for r in test if dh._match_row_against_index(r["question"] + " " + r["answer"], index_plain)[0])
        self.assertEqual(plain_hits, len(test))

        scan = dh._scan_prepared_split_pairs({"train": train, "val": [], "test": test})
        self.assertEqual(scan["total_leaked"], 0, scan["examples"][:2])
        self.assertEqual(scan["severity"], "ok")
        self.assertGreaterEqual(scan["boilerplate_ignored"]["lines"], 12)

    def test_real_duplicates_and_near_duplicates_are_still_caught(self):
        train = [row(r) for r in REQUESTS[:20]]
        test = [
            row(REQUESTS[0]),                                                    # exact duplicate
            row("Show me the open invoices for the northern warehouse please"),  # near-duplicate (wording)
            row(REQUESTS[21]),                                                   # distinct
            row(REQUESTS[22]),                                                   # distinct
        ]
        scan = dh._scan_prepared_split_pairs({"train": train, "val": [], "test": test})
        self.assertEqual(scan["per_pair"]["test_in_train"]["leaked"], 2)
        kinds = sorted(ex["match_kind"] for ex in scan["examples"])
        self.assertEqual(kinds, ["exact", "near_duplicate"])
        # the drill-down excerpt shows the request, not the shared block
        self.assertTrue(all(not ex["excerpt"].startswith("Available tools") for ex in scan["examples"]))

    def test_gold_scan_ignores_the_shared_block_too(self):
        train = [row(r) for r in REQUESTS[:18]]
        gold = {"gold_dev": [], "gold_test": [row(REQUESTS[18]), row(REQUESTS[1])]}
        scan = dh._scan_leakage_rows(train, gold)
        self.assertEqual(scan["total_leaked"], 1)  # only the real duplicate
        self.assertEqual(scan["test_leaked"], 1)

    def test_dedup_before_a_split_keeps_distinct_rows_and_drops_real_duplicates(self):
        entries = [row(r) for r in REQUESTS] + [row(REQUESTS[3]), row(REQUESTS[5] + " please")]
        kept, dropped = _dedup_entries_for_split(entries)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(kept), len(REQUESTS))

    def test_short_shared_headers_and_small_corpora_change_nothing(self):
        headed = [f"Question:\n{r}" for r in REQUESTS]
        self.assertEqual(dh.shared_boilerplate_lines(headed), frozenset())  # under the char floor
        self.assertEqual(dh.shared_boilerplate_lines([f"{BLOCK}\n\n{r}" for r in REQUESTS[:5]]), frozenset())  # too few rows
        self.assertEqual(dh.strip_boilerplate("abc", frozenset()), "abc")
        # a row that is all boilerplate is compared whole
        bp = dh.shared_boilerplate_lines([f"{BLOCK}\n\n{r}" for r in REQUESTS])
        self.assertEqual(dh.strip_boilerplate(BLOCK, bp), BLOCK)
        self.assertEqual(dh.strip_boilerplate(f"{BLOCK}\n\nhello there", bp).strip(), "hello there")

    def test_a_plain_corpus_scans_exactly_as_before(self):
        train = [{"question": r, "answer": "x"} for r in REQUESTS[:16]]
        test = [{"question": REQUESTS[2], "answer": "x"}, {"question": REQUESTS[20], "answer": "x"}]
        scan = dh._scan_prepared_split_pairs({"train": train, "val": [], "test": test})
        self.assertIsNone(scan["boilerplate_ignored"])
        self.assertEqual(scan["total_leaked"], 1)


if __name__ == "__main__":
    unittest.main()
