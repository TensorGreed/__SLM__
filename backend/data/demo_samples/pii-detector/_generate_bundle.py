"""Generator for the PII/PCI demo dataset.

Hand-counted character offsets are bug-prone. This script composes each
row from a list of segments (literal strings or entity tuples) and
computes start/end offsets on the fly while building the text, so the
gold answers and the source text can never drift out of sync.

Run from the bundle directory:

    cd backend/data/demo_samples/pii-detector
    python _generate_bundle.py

Writes:
  - pii_records.csv  (training source — 60 rows: text + entities_json)
  - gold.jsonl       (locked gold set — 25 rows in the {key, input,
                       expected, rationale} shape Phase 4.1's seeder
                       consumes)

All PII values are synthetic. Names are placeholder-ish; phone numbers
are 555-prefixed; credit cards are well-known test PANs (Stripe /
Visa "0000" placeholders); SSNs use the 000-XX-XXXX range that's
reserved-never-issued; emails are @example.{com,net,org}.

Re-run is idempotent — same input templates produce the same files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class E:
    """One entity segment in a template."""

    type: str
    text: str


# A "template" is a list of either plain strings or E entries.
Segment = str | E


def render(parts: list[Segment]) -> tuple[str, list[dict]]:
    """Walk a template, return (text, entities) with correct offsets."""
    buf: list[str] = []
    entities: list[dict] = []
    cursor = 0
    for part in parts:
        if isinstance(part, E):
            start = cursor
            buf.append(part.text)
            cursor += len(part.text)
            entities.append(
                {"type": part.type, "start": start, "end": cursor, "text": part.text}
            )
        else:
            buf.append(part)
            cursor += len(part)
    return "".join(buf), entities


# ── Training set (60 rows) ────────────────────────────────────────────

TRAINING_TEMPLATES: list[list[Segment]] = [
    # — Customer support style —
    [
        "Hi support team, I'm ",
        E("person_name", "Jane Doe"),
        " and I can't log in. Email me at ",
        E("email", "jane.doe@example.com"),
        " or call ",
        E("phone", "555-0173"),
        ".",
    ],
    [
        "Please reset the password for account ",
        E("email", "marcus.lee@example.net"),
        ". Backup phone ",
        E("phone", "(555) 014-2381"),
        ".",
    ],
    [
        "Ticket from ",
        E("person_name", "Priya Raman"),
        " (DOB ",
        E("date_of_birth", "1989-03-14"),
        ") — billing question.",
    ],
    [
        "Customer ",
        E("person_name", "Olivia Carter"),
        " reports a duplicate charge on card ending ",
        E("credit_card", "4242424242424242"),
        ".",
    ],
    [
        "Caller ID ",
        E("phone", "555-0166"),
        " left a voicemail asking about invoice 88421. They also gave ",
        E("email", "support+caller@example.org"),
        ".",
    ],
    [
        "Refund request from ",
        E("person_name", "Daniel Kim"),
        ", reachable at ",
        E("email", "dkim@example.com"),
        ". Account number ",
        E("bank_account", "GB29 NWBK 6016 1331 9268 19"),
        ".",
    ],
    [
        "Update mailing address for ",
        E("person_name", "Aisha Khan"),
        " to ",
        E("street_address", "742 Evergreen Terrace, Springfield, IL 62704"),
        ".",
    ],
    [
        "Lost wallet — please freeze card ",
        E("credit_card", "5555555555554444"),
        " owned by ",
        E("person_name", "Henrik Olsen"),
        ".",
    ],
    [
        "Verification call to ",
        E("phone", "+1-555-0188"),
        ". SSN on file: ",
        E("ssn", "000-12-3456"),
        ".",
    ],
    [
        "Identity proof received for ",
        E("person_name", "Mei Tanaka"),
        ", date of birth ",
        E("date_of_birth", "1995-11-22"),
        ".",
    ],
    # — System / log lines —
    [
        "Login from ",
        E("ip_address", "192.0.2.55"),
        " user=",
        E("email", "ops@example.com"),
        " at 03:14 UTC.",
    ],
    [
        "Suspicious request: src=",
        E("ip_address", "203.0.113.42"),
        " auth=",
        E("api_key", "sk_live_4eC39HqLyjWDarjtT1zdp7dc"),
        " rejected.",
    ],
    [
        "Cron job heartbeat from ",
        E("ip_address", "198.51.100.7"),
        " — ok.",
    ],
    [
        "Token issued to ",
        E("email", "service-acct@example.com"),
        ": ",
        E("api_key", "ghp_16C7e42F292c6912E7710c838347Ae178B4a"),
        ".",
    ],
    [
        "Probe failure for client ",
        E("person_name", "T. Nguyen"),
        " (",
        E("ip_address", "10.0.0.84"),
        ") — retry queued.",
    ],
    # — Form / data dump style —
    [
        "Name: ",
        E("person_name", "Sofia Martinez"),
        " | Email: ",
        E("email", "sofia.m@example.com"),
        " | Phone: ",
        E("phone", "555-0199"),
        " | DOB: ",
        E("date_of_birth", "1992-07-04"),
        ".",
    ],
    [
        "Applicant ",
        E("person_name", "Yuki Sato"),
        " applied for the credit line. SSN: ",
        E("ssn", "000-22-1188"),
        ", address: ",
        E("street_address", "1313 Mockingbird Lane, Buffalo, NY 14201"),
        ".",
    ],
    [
        "Wire transfer initiated by ",
        E("person_name", "Carlos Rivera"),
        " to account ",
        E("bank_account", "DE89 3704 0044 0532 0130 00"),
        " — pending review.",
    ],
    [
        "Direct deposit set up: ",
        E("person_name", "Emily Chen"),
        ", account ",
        E("bank_account", "021000021 123456789"),
        ".",
    ],
    [
        "New employee onboarded: ",
        E("person_name", "Liam O'Connor"),
        ", phone ",
        E("phone", "(555) 014-7733"),
        ", emergency contact ",
        E("phone", "555-0124"),
        ".",
    ],
    # — Chat / DM style —
    [
        "Hey, you can reach me on ",
        E("email", "alex@example.net"),
        " — I'm out of office until next Tuesday.",
    ],
    [
        "btw my new number is ",
        E("phone", "555-0142"),
        ", same name.",
    ],
    [
        "I sent the invoice to ",
        E("email", "billing@example.com"),
        " yesterday — let me know if it bounced.",
    ],
    [
        "Sharing my address: ",
        E("street_address", "221B Baker Street, London NW1 6XE"),
        ". Package can come tomorrow.",
    ],
    [
        "If anyone asks, the on-call number is ",
        E("phone", "555-0107"),
        " through end of week.",
    ],
    # — Mixed / fraud-flagged scenarios —
    [
        "Possible card fraud — duplicate transactions on ",
        E("credit_card", "378282246310005"),
        " issued to ",
        E("person_name", "Greta Larsson"),
        ".",
    ],
    [
        "Card-on-file ",
        E("credit_card", "6011111111111117"),
        " expired. Owner: ",
        E("email", "greta.l@example.com"),
        ".",
    ],
    [
        "Beneficiary ",
        E("person_name", "Noor Hassan"),
        " (SSN ",
        E("ssn", "000-44-9911"),
        ") flagged on watchlist — manual review.",
    ],
    [
        "API key leaked in commit by ",
        E("email", "dev@example.com"),
        " — rotate ",
        E("api_key", "AKIAIOSFODNN7EXAMPLE"),
        " immediately.",
    ],
    [
        "Slack DM from ",
        E("person_name", "Ravi Patel"),
        ": 'my new corp card is ",
        E("credit_card", "5105105105105100"),
        ", use that for the team dinner.'",
    ],
    # — Edge cases / multiple entities of same type —
    [
        "CC list: ",
        E("email", "alice@example.com"),
        ", ",
        E("email", "bob@example.net"),
        ", ",
        E("email", "carol@example.org"),
        ".",
    ],
    [
        "Roommates ",
        E("person_name", "Maya Singh"),
        " and ",
        E("person_name", "Theo Müller"),
        " share the address ",
        E("street_address", "88 Linden Avenue, Cambridge, MA 02139"),
        ".",
    ],
    [
        "Phones changed: old ",
        E("phone", "555-0101"),
        ", new ",
        E("phone", "555-0210"),
        " — please update CRM.",
    ],
    [
        "Three IPs hammering the API: ",
        E("ip_address", "192.0.2.10"),
        ", ",
        E("ip_address", "192.0.2.11"),
        ", ",
        E("ip_address", "192.0.2.12"),
        ".",
    ],
    [
        "Two SSNs on the same form — ",
        E("ssn", "000-55-7711"),
        " (applicant) and ",
        E("ssn", "000-55-7712"),
        " (spouse).",
    ],
    # — Single-entity rows (variety) —
    [
        "My email is ",
        E("email", "robin@example.com"),
        ".",
    ],
    [
        "Call ",
        E("phone", "555-0150"),
        " for after-hours support.",
    ],
    [
        "SSN: ",
        E("ssn", "000-33-2211"),
        ".",
    ],
    [
        "Card: ",
        E("credit_card", "4111111111111111"),
        ".",
    ],
    [
        "Name on file: ",
        E("person_name", "Ingrid Lindberg"),
        ".",
    ],
    [
        "Address: ",
        E("street_address", "500 Terry Francois Blvd, San Francisco, CA 94158"),
        ".",
    ],
    [
        "DOB: ",
        E("date_of_birth", "1978-02-19"),
        ".",
    ],
    [
        "Origin IP: ",
        E("ip_address", "172.16.254.1"),
        ".",
    ],
    [
        "API key: ",
        E("api_key", "xoxb-12345-67890-abcdefghij"),
        ".",
    ],
    [
        "IBAN: ",
        E("bank_account", "FR76 3000 6000 0112 3456 7890 189"),
        ".",
    ],
    # — Clean rows (no PII) so the model learns when to emit []
    ["Please find attached the Q3 product roadmap for review."],
    ["The release is scheduled for Friday afternoon."],
    ["Re-enable the feature flag once the canary clears."],
    ["Migration ran successfully — no rows skipped."],
    ["Standup notes: blocked on infra, no PII to redact today."],
    # — Long context / paragraph rows —
    [
        "Hi team — looping in ",
        E("person_name", "Hannah Becker"),
        " on the escalation. You can reach her at ",
        E("email", "hannah.becker@example.com"),
        " or ",
        E("phone", "(555) 014-9920"),
        " during EU hours. She's based at ",
        E("street_address", "Friedrichstraße 100, 10117 Berlin"),
        ".",
    ],
    [
        "Welcome packet for ",
        E("person_name", "Diego Fernández"),
        " — phone ",
        E("phone", "555-0188"),
        ", DOB ",
        E("date_of_birth", "1985-09-30"),
        ", direct deposit account ",
        E("bank_account", "ES91 2100 0418 4502 0005 1332"),
        ", primary email ",
        E("email", "diego.f@example.com"),
        ".",
    ],
    [
        "Security alert: SSO token ",
        E("api_key", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"),
        " was rotated for ",
        E("email", "admin@example.net"),
        " from ",
        E("ip_address", "203.0.113.99"),
        ". No further action needed.",
    ],
    [
        "Vendor record for ",
        E("person_name", "Sven Andersson"),
        ": billing email ",
        E("email", "billing@example.org"),
        ", remit-to ",
        E("street_address", "Storgatan 1, 111 51 Stockholm"),
        ", bank ",
        E("bank_account", "SE45 5000 0000 0583 9825 7466"),
        ".",
    ],
    [
        "Card declined for purchase by ",
        E("person_name", "Beatrice Romano"),
        " on card ",
        E("credit_card", "5454545454545454"),
        " — please retry or use a different card.",
    ],
    [
        "Account locked after 5 failed logins from ",
        E("ip_address", "198.51.100.55"),
        ". Owner: ",
        E("email", "user42@example.com"),
        ".",
    ],
    [
        "Background check pulled for ",
        E("person_name", "Kenji Aoki"),
        ", DOB ",
        E("date_of_birth", "1990-12-01"),
        ", SSN ",
        E("ssn", "000-67-2233"),
        ".",
    ],
    [
        "Reissue physical card for ",
        E("person_name", "Lucía Hernández"),
        " — old ",
        E("credit_card", "4222222222222"),
        ", deliver to ",
        E("street_address", "10 Downing Street, London SW1A 2AA"),
        ".",
    ],
    [
        "GDPR data export requested by ",
        E("email", "subject@example.com"),
        " — name on file ",
        E("person_name", "Oluwaseun Adebayo"),
        ", account number ",
        E("bank_account", "NG34 GTBI 0581 7321 5476"),
        ".",
    ],
    [
        "Phishing report — sender claimed to be ",
        E("person_name", "Eva Schmidt"),
        " but the reply-to was ",
        E("email", "no-reply@suspicious-example.tld"),
        ".",
    ],
    [
        "Final reminder before suspension. Account ",
        E("email", "delinquent@example.net"),
        " | DOB ",
        E("date_of_birth", "1972-04-17"),
        ".",
    ],
]


# ── Gold set (25 rows, separate templates so it's not a leak) ─────────

GOLD_TEMPLATES: list[list[Segment]] = [
    [
        "Customer ",
        E("person_name", "Audrey Liu"),
        " reports two unauthorised charges. Card on file: ",
        E("credit_card", "4012888888881881"),
        ".",
    ],
    [
        "Resetting MFA for ",
        E("email", "qa-bot@example.com"),
        " — confirmation sent to ",
        E("phone", "555-0102"),
        ".",
    ],
    [
        "Background-check vendor needs SSN ",
        E("ssn", "000-78-4321"),
        " for ",
        E("person_name", "Maria Esposito"),
        ".",
    ],
    [
        "DM from ",
        E("person_name", "Hiroshi Tanaka"),
        ": 'rotate ",
        E("api_key", "sk_test_BQokikJOvBiI2HlWgH4olfQ2"),
        " — it leaked in a screenshot.'",
    ],
    [
        "Mail merge field — name: ",
        E("person_name", "Anna Kowalski"),
        ", address: ",
        E("street_address", "ulica Marszałkowska 1, 00-001 Warszawa"),
        ".",
    ],
    [
        "Direct deposit edit for ",
        E("person_name", "Tobias Berg"),
        ", account ",
        E("bank_account", "NL91 ABNA 0417 1643 00"),
        ".",
    ],
    [
        "Suspicious login from ",
        E("ip_address", "192.0.2.250"),
        " — owner: ",
        E("email", "owner@example.org"),
        ".",
    ],
    [
        "Onboarding form — DOB ",
        E("date_of_birth", "2001-06-08"),
        ", phone ",
        E("phone", "(555) 014-3030"),
        ".",
    ],
    [
        "Two cards on file: ",
        E("credit_card", "30569309025904"),
        " and ",
        E("credit_card", "3530111333300000"),
        ".",
    ],
    [
        "Compliance ping — please redact ",
        E("ssn", "000-99-1122"),
        " and ",
        E("ssn", "000-99-1123"),
        " from the export.",
    ],
    [
        "Logs show traffic from ",
        E("ip_address", "10.10.10.10"),
        ", ",
        E("ip_address", "10.10.10.11"),
        ", and ",
        E("ip_address", "10.10.10.12"),
        " during the incident window.",
    ],
    [
        "Welcome ",
        E("person_name", "Selene Park"),
        "! Your email is ",
        E("email", "selene.p@example.com"),
        ".",
    ],
    [
        "Wire to ",
        E("bank_account", "AT61 1904 3002 3457 3201"),
        " owned by ",
        E("person_name", "Klaus Wagner"),
        " — flagged for AML review.",
    ],
    [
        "PHI scrub needed — DOB ",
        E("date_of_birth", "1958-11-03"),
        " plus phone ",
        E("phone", "555-0188"),
        ".",
    ],
    [
        "Slack export — secret in plain text: ",
        E("api_key", "ghp_AAAA0000BBBB1111CCCC2222DDDD3333EEEE"),
        ".",
    ],
    [
        "Reminder — ",
        E("person_name", "Naledi Mokoena"),
        " is at ",
        E("street_address", "32 Loop Street, Cape Town 8001"),
        " for the rest of the month.",
    ],
    [
        "Sweep of ",
        E("email", "all-hands@example.com"),
        " mentioned ",
        E("phone", "555-0173"),
        " — double-check before publishing.",
    ],
    [
        "Card declined — ",
        E("credit_card", "6011000990139424"),
        " is past expiry.",
    ],
    [
        "Standup note: nothing sensitive to redact this morning.",
    ],
    [
        "Internal memo — release notes are clean of customer PII.",
    ],
    [
        "Outage RCA mentions ",
        E("ip_address", "172.31.0.15"),
        " — confirm before publishing externally.",
    ],
    [
        "Form A12 from ",
        E("person_name", "Wanjiku Mwangi"),
        ": SSN ",
        E("ssn", "000-12-7788"),
        ", DOB ",
        E("date_of_birth", "1982-08-15"),
        ".",
    ],
    [
        "Update billing email for ",
        E("person_name", "Pavel Novak"),
        " to ",
        E("email", "pavel.n@example.com"),
        ".",
    ],
    [
        "Bank account on file: ",
        E("bank_account", "CH9300762011623852957"),
        " — owner ",
        E("person_name", "Léa Dubois"),
        ".",
    ],
    [
        "Two API keys leaked in the same commit: ",
        E("api_key", "AKIA1111EXAMPLEKEY11"),
        " and ",
        E("api_key", "AKIA2222EXAMPLEKEY22"),
        ".",
    ],
]


def main() -> None:
    here = Path(__file__).resolve().parent

    # Training CSV ------------------------------------------------------
    train_path = here / "pii_records.csv"
    with train_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "entities_json"])
        for parts in TRAINING_TEMPLATES:
            text, entities = render(parts)
            payload = {"entities": entities}
            writer.writerow([text, json.dumps(payload, ensure_ascii=False)])
    print(f"wrote {len(TRAINING_TEMPLATES)} training rows → {train_path.name}")

    # Gold JSONL --------------------------------------------------------
    gold_path = here / "gold.jsonl"
    with gold_path.open("w", encoding="utf-8") as fh:
        for idx, parts in enumerate(GOLD_TEMPLATES, start=1):
            text, entities = render(parts)
            row = {
                "key": f"g{idx:02d}",
                "input": {"text": text},
                "expected": {"entities": entities},
                "rationale": _rationale_for(entities),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(GOLD_TEMPLATES)} gold rows → {gold_path.name}")


def _rationale_for(entities: list[dict]) -> str:
    if not entities:
        return "Clean text — model should emit an empty entities list."
    by_type: dict[str, int] = {}
    for e in entities:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    parts = [f"{c} {t}" for t, c in sorted(by_type.items())]
    return "Must detect: " + ", ".join(parts) + "."


if __name__ == "__main__":
    main()
