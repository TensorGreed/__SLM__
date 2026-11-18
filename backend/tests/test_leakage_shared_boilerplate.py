"""Split-leakage / dedup matcher with shared lines.

* Tool-calling rows (the ``tool-call-json`` shape: tool list + blank line + user request): the tool list
  is always ignored, so distinct requests are not near-duplicates of each other; real copies and
  reworded copies of the request are still caught.
* Every other dataset matches exactly as before unless the project opts in
  (``runtime_config.data_health.ignore_shared_lines``); then the shared-line set is computed ONCE from
  the whole corpus before the dedup, and the rescan uses that same set (the reviewer's scenario:
  5,000 rows without a preamble first, then 15,000 with a 267-char preamble, copies included).
"""

from __future__ import annotations

import random
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services import data_health_service as dh
from app.services.dataset_service import _dedup_entries_for_split
from app.services.trainability_forecast_service import _row_to_text

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


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
PREAMBLE = ("You are a careful assistant for the operations desk. Answer only from the records you are given, "
            "keep every answer short and factual, never invent identifiers, say so plainly when a record is "
            "missing, and use the units and spelling that the records use in your reply.")


def tool_row(request: str) -> dict:
    """A row as the ``tool-call-json`` adapter writes it with a tool catalog."""
    return {"question": f"{BLOCK}\n\n{request}", "user_prompt": request, "tool_name": "tool_1",
            "answer": '{"tool": "tool_1", "args": {}}'}


def preamble_row(request: str) -> dict:
    return {"question": f"{PREAMBLE}\n{request}", "answer": "ok"}


OPT_IN = {"data_health": {"ignore_shared_lines": True}}


class ToolRowsTests(unittest.TestCase):
    def scan(self, splits):
        rows = [r for s in splits.values() for r in s]
        return dh._scan_prepared_split_pairs(splits, dh.leakage_boilerplate(rows, {}))

    def test_distinct_requests_behind_a_tool_list_do_not_leak(self):
        train = [tool_row(r) for r in REQUESTS[:16]]
        test = [tool_row(r) for r in REQUESTS[16:]]
        plain = dh._scan_prepared_split_pairs({"train": train, "val": [], "test": test})
        self.assertEqual(plain["total_leaked"], len(test))  # the false positive, without the stripping
        scan = self.scan({"train": train, "val": [], "test": test})
        self.assertEqual(scan["total_leaked"], 0, scan["examples"][:2])
        self.assertGreaterEqual(scan["boilerplate_ignored"]["lines"], 40)

    def test_real_duplicates_and_near_duplicates_are_still_caught(self):
        train = [tool_row(r) for r in REQUESTS[:20]]
        test = [tool_row(REQUESTS[0]),
                tool_row("Show me the open invoices for the northern warehouse please"),
                tool_row(REQUESTS[21]), tool_row(REQUESTS[22])]
        scan = self.scan({"train": train, "val": [], "test": test})
        self.assertEqual(scan["per_pair"]["test_in_train"]["leaked"], 2)
        self.assertEqual(sorted(ex["match_kind"] for ex in scan["examples"]), ["exact", "near_duplicate"])
        self.assertTrue(all(not ex["excerpt"].startswith("Available tools") for ex in scan["examples"]))

    def test_gold_scan_and_dedup_use_the_same_lines(self):
        train = [tool_row(r) for r in REQUESTS[:18]]
        gold = {"gold_dev": [], "gold_test": [tool_row(REQUESTS[18]), tool_row(REQUESTS[1])]}
        bp = dh.leakage_boilerplate(train + gold["gold_test"], {})
        self.assertEqual(dh._scan_leakage_rows(train, gold, bp)["total_leaked"], 1)
        entries = [tool_row(r) for r in REQUESTS] + [tool_row(REQUESTS[3]), tool_row(REQUESTS[5] + " please")]
        kept, dropped = _dedup_entries_for_split(entries, dh.leakage_boilerplate(entries, {}))
        self.assertEqual((len(kept), dropped), (len(REQUESTS), 2))


class OtherDatasetsUnchangedTests(unittest.TestCase):
    """Without the tool shape and without the opt-in, nothing is stripped: the matcher runs the exact
    code path of main (empty set), including its flags on a long shared preamble."""

    def test_default_is_identical_to_no_stripping(self):
        train = [preamble_row(r) for r in REQUESTS[:16]]
        test = [preamble_row(r) for r in REQUESTS[16:]]
        rows = train + test
        bp = dh.leakage_boilerplate(rows, {})
        self.assertEqual(bp, frozenset())
        splits = {"train": train, "val": [], "test": test}
        self.assertEqual(dh._scan_prepared_split_pairs(splits, bp), dh._scan_prepared_split_pairs(splits))
        self.assertNotIn("boilerplate_ignored", dh._scan_prepared_split_pairs(splits, bp))
        self.assertEqual(_dedup_entries_for_split(rows, bp), _dedup_entries_for_split(rows))
        gold = {"gold_dev": [], "gold_test": test}
        self.assertEqual(dh._scan_leakage_rows(train, gold, bp), dh._scan_leakage_rows(train, gold))

    def test_short_headers_and_small_corpora_change_nothing_even_opted_in(self):
        self.assertEqual(dh.shared_boilerplate_lines([f"Question:\n{r}" for r in REQUESTS]), frozenset())
        self.assertEqual(dh.shared_boilerplate_lines([f"{PREAMBLE}\n{r}" for r in REQUESTS[:5]]), frozenset())
        bp = dh.shared_boilerplate_lines([f"{PREAMBLE}\n{r}" for r in REQUESTS])
        self.assertEqual(dh.strip_boilerplate(PREAMBLE, bp), PREAMBLE)  # all-boilerplate row compares whole

    def test_opt_in_ignores_the_shared_preamble(self):
        train = [preamble_row(r) for r in REQUESTS[:16]]
        test = [preamble_row(r) for r in REQUESTS[16:]] + [preamble_row(REQUESTS[2])]
        bp = dh.leakage_boilerplate(train + test, OPT_IN)
        self.assertIn(PREAMBLE, bp)
        scan = dh._scan_prepared_split_pairs({"train": train, "val": [], "test": test}, bp)
        self.assertEqual(scan["total_leaked"], 1)  # only the real copy


class ReviewerScenarioTests(unittest.TestCase):
    """Opted in: 5,000 rows without the preamble first, then 15,000 with a 267-char preamble, plus
    copies. The set is computed once from the whole corpus BEFORE the dedup and reused by the rescan:
    after "Re-split with dedup" nothing is flagged."""

    @staticmethod
    def request(i: int) -> str:
        return (f"please find the open record item{i} in zone{i % 97} for team{i % 53} "
                "by tomorrow morning")

    def test_dedup_then_rescan_flags_nothing(self):
        self.assertEqual(len(PREAMBLE), 267)
        entries = [{"question": self.request(i), "answer": f"ok{i}"} for i in range(5000)]
        entries += [{"question": f"{PREAMBLE}\n{self.request(i)}", "answer": f"ok{i}"} for i in range(5000, 20000)]
        copies = [dict(entries[i]) for i in range(6000, 6200)]                                   # exact
        near = [{"question": f"{PREAMBLE}\n{self.request(i)} now", "answer": f"ok{i}"} for i in range(7000, 7100)]
        entries += copies + near

        # The old per-sample choice (first 5,000 rows) sees no preamble at all.
        self.assertEqual(dh.shared_boilerplate_lines([_row_to_text(e) for e in entries[:5000]]), frozenset())
        started = time.monotonic()
        shared = dh.shared_boilerplate_lines([_row_to_text(e) for e in entries])  # what split_dataset stores
        self.assertEqual(shared, frozenset({PREAMBLE}))
        bp = dh.leakage_boilerplate(entries, OPT_IN, stored=shared)
        kept, dropped = _dedup_entries_for_split(entries, bp)
        self.assertEqual(dropped, 300)

        rng = random.Random(7)
        rng.shuffle(kept)
        n = len(kept)
        splits = {"train": kept[: int(n * 0.8)], "val": kept[int(n * 0.8): int(n * 0.9)], "test": kept[int(n * 0.9):]}
        rescan_bp = dh.leakage_boilerplate([r for s in splits.values() for r in s], OPT_IN, stored=shared)
        self.assertEqual(rescan_bp, bp)
        scan = dh._scan_prepared_split_pairs(splits, rescan_bp)
        self.assertEqual(scan["total_leaked"], 0, scan["examples"][:2])
        self.assertLess(time.monotonic() - started, 120)

    def test_stored_set_round_trips(self):
        pid = 990_000 + (time.time_ns() % 1000)
        dh.write_stored_shared_lines(pid, frozenset({PREAMBLE}), rows=3)
        self.assertEqual(dh.load_stored_shared_lines(pid), frozenset({PREAMBLE}))
        dh.write_stored_shared_lines(pid, None)
        self.assertIsNone(dh.load_stored_shared_lines(pid))


class SettingsApiTests(unittest.TestCase):
    def test_opt_in_round_trips_and_shows_in_the_report(self):
        pid = client.post("/api/projects", json={"name": f"dh-settings-{time.time_ns()}", "description": "x"}).json()["id"]
        self.assertFalse(client.get(f"/api/projects/{pid}/data-health").json()["settings"]["ignore_shared_lines"])
        on = client.put(f"/api/projects/{pid}/data-health/settings", json={"ignore_shared_lines": True})
        self.assertEqual(on.status_code, 200, on.text)
        self.assertTrue(client.get(f"/api/projects/{pid}/data-health").json()["settings"]["ignore_shared_lines"])
        client.put(f"/api/projects/{pid}/data-health/settings", json={"ignore_shared_lines": False})
        self.assertFalse(client.get(f"/api/projects/{pid}/data-health").json()["settings"]["ignore_shared_lines"])


if __name__ == "__main__":
    unittest.main()
