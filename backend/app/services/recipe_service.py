"""Task-shape recipe registry.

A Recipe answers the question "what kind of model do I want to
train?" — it bundles task profile, adapter id, scoring mode, gold-
set template, suggested base model, and seed eval prompts for a
specific task shape (Q&A, classification, span extraction, etc.).

Orthogonal to StarterPack, which answers "what's my data about?"
(legal, medical, ...). A new project picks one of each: a recipe
chooses the model SHAPE; a starter pack chooses the domain
context. The composition is the answer to "what defaults should
I land on for this brand-new project?"

The recipes shipped here are the 6 in ROADMAP-NEXT.md Theme 2:
qa-sft, classification, span-extraction, summarization,
code-review, generic-sft.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field

RECIPE_CATALOG_VERSION = "recipes.builtin/v1"


class ShapeColumn(BaseModel):
    """A column the shape sniffer should look for in a CSV/JSONL.

    `name_patterns` is a list of case-insensitive substrings; any
    match counts. The shape sniffer scores a file by counting how
    many recipe columns find a match in the file's headers.
    """

    name_patterns: list[str]
    column_role: str  # "input" | "output" | "label" | "rationale" | "auxiliary"
    required: bool = True


class ShapeSignature(BaseModel):
    """A pattern the shape sniffer can match against a file.

    A recipe can carry multiple signatures (e.g. "input/output" and
    "prompt/response" for the same Q&A task). Confidence is the
    base score; the sniffer may apply discounts for partial matches.
    """

    columns: list[ShapeColumn]
    base_confidence: float = 0.9


class GoldFieldSpec(BaseModel):
    name: str
    required: bool = True
    description: str = ""


class GoldTemplate(BaseModel):
    """How a user should structure their gold-set rows for this recipe."""

    shape_label: str  # e.g. "question_expected_rationale"
    min_rows_recommended: int = 50
    fields: list[GoldFieldSpec]
    example_row: dict[str, Any] = Field(default_factory=dict)


class Recipe(BaseModel):
    id: str
    name: str
    headline: str
    description: str
    icon: str = "🧪"

    # Pipeline-level defaults
    task_profile: str
    adapter_id: str
    scoring_mode: str  # "field_match" | "span_set"

    # Data shape defaults
    default_input_column: str
    default_output_column: str

    # Base model choices
    suggested_base_model: str
    alt_base_models: list[str] = Field(default_factory=list)

    # Pipeline routing
    target_profile: str = "vllm_server"
    training_plan_profile: str = "balanced"
    eval_pack_id: str = "evalpack.general.default"

    # Gold-set guidance
    gold_template: GoldTemplate

    # Seed prompts the user can run in the Playground after training
    sample_eval_prompts: list[str] = Field(default_factory=list)

    # Where users typically find data for this shape
    data_acquisition_hints: list[str] = Field(default_factory=list)

    # How to recognize a file that fits this recipe
    shape_signatures: list[ShapeSignature] = Field(default_factory=list)

    # Provenance
    catalog_source: str = "builtin"
    catalog_version: str = "builtin-v1"
    is_builtin: bool = True


# ─────────────────────────────────────────────────────────────────────
# Six built-in recipes covering the task shapes ROADMAP-NEXT calls out.
# ─────────────────────────────────────────────────────────────────────


def _qa_sft_recipe() -> Recipe:
    return Recipe(
        id="qa-sft",
        name="Question & Answer Assistant",
        headline="Train a model to answer questions like a domain expert.",
        description=(
            "For datasets where each row is a question paired with the "
            "answer you wish the model would give. The canonical SFT shape — "
            "support tickets, FAQ entries, documentation Q&A, internal "
            "knowledge-base lookups all fit here."
        ),
        icon="💬",
        task_profile="instruction_sft",
        adapter_id="qa-pair",
        scoring_mode="field_match",
        default_input_column="question",
        default_output_column="answer",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="question_expected_rationale",
            min_rows_recommended=50,
            fields=[
                GoldFieldSpec(name="question", description="The question the model should answer."),
                GoldFieldSpec(name="expected", description="The reference answer."),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional: why this is the right answer (used by some eval handlers).",
                ),
            ],
            example_row={
                "question": "How do I reset my password?",
                "expected": "Visit Settings → Security → Reset password, then check your email for a confirmation link.",
                "rationale": "Standard SaaS password-reset flow; the answer references the exact UI path.",
            },
        ),
        sample_eval_prompts=[
            "How do I reset my password?",
            "What is your refund policy?",
            "Can I export my data?",
        ],
        data_acquisition_hints=[
            "Export resolved tickets from Zendesk/Intercom/Salesforce as a CSV.",
            "Convert FAQ pages: question column + answer column from the page's Q&A pairs.",
            "Generate synthetic Q&A from documentation using the Synthetic tab (requires a teacher model).",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["question", "prompt", "query", "input", "q"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["answer", "response", "output", "a", "reply"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.92,
            ),
        ],
    )


def _classification_recipe() -> Recipe:
    return Recipe(
        id="classification",
        name="Text Classifier",
        headline="Train a model to assign each input to one of a fixed set of labels.",
        description=(
            "For datasets where each row is a text snippet paired with a "
            "single discrete label. Sentiment, intent, spam/not-spam, topic "
            "categorization, multi-class triage — all classification tasks "
            "fit here. The label column should have a small fixed vocabulary."
        ),
        icon="🏷️",
        task_profile="classification",
        adapter_id="classification-label",
        scoring_mode="field_match",
        default_input_column="text",
        default_output_column="label",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "distilbert-base-uncased",
        ],
        target_profile="mobile_cpu",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.classification.default",
        gold_template=GoldTemplate(
            shape_label="text_label_rationale",
            min_rows_recommended=100,
            fields=[
                GoldFieldSpec(name="text", description="The input text to classify."),
                GoldFieldSpec(name="label", description="The correct class label (must match training labels exactly)."),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional: why this label is correct.",
                ),
            ],
            example_row={
                "text": "This product is amazing — it changed my life!",
                "label": "positive",
                "rationale": "Strong positive sentiment, no qualifiers.",
            },
        ),
        sample_eval_prompts=[
            "This is the worst experience I've ever had.",
            "It's fine, nothing special.",
            "Absolutely love it, would recommend!",
        ],
        data_acquisition_hints=[
            "Existing labeled CSVs from product analytics or support routing.",
            "Public benchmark datasets on Hugging Face Hub (search 'sentiment', 'classification').",
            "Kaggle datasets — most classification competitions ship in this shape.",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["text", "content", "input", "message", "review", "tweet", "post"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["label", "class", "category", "sentiment", "tag"],
                        column_role="label",
                    ),
                ],
                base_confidence=0.90,
            ),
        ],
    )


def _span_extraction_recipe() -> Recipe:
    return Recipe(
        id="span-extraction",
        name="Structured Span Extractor",
        headline="Train a model to find and label entities or spans inside text.",
        description=(
            "For datasets where each row is text paired with a structured "
            "JSON list of every entity to extract — type, start offset, end "
            "offset, raw text. PII detection, NER, invoice field extraction, "
            "structured key-value pulling from documents, span-level labeling "
            "all fit here. The output is JSON, not a single label."
        ),
        icon="🎯",
        task_profile="structured_extraction",
        adapter_id="default-canonical",
        scoring_mode="span_set",
        default_input_column="text",
        default_output_column="entities_json",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="text_entities_rationale",
            min_rows_recommended=150,
            fields=[
                GoldFieldSpec(name="text", description="The source text to extract spans from."),
                GoldFieldSpec(
                    name="entities",
                    description="A JSON array of {type, start, end, text} objects naming every span.",
                ),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional: notes about edge cases this row exercises.",
                ),
            ],
            example_row={
                "text": "Contact me at jane@example.com or +1-555-867-5309.",
                "entities": [
                    {"type": "email", "start": 14, "end": 30, "text": "jane@example.com"},
                    {"type": "phone", "start": 34, "end": 50, "text": "+1-555-867-5309"},
                ],
                "rationale": "Tests two-entity coverage with offsets after a 'Contact me at' prefix.",
            },
        ),
        sample_eval_prompts=[
            "Send the invoice to billing@acme.co by 555-123-9876.",
            "John Doe lives at 1600 Pennsylvania Ave, Washington DC.",
            "My SSN is 123-45-6789 — please do not share.",
        ],
        data_acquisition_hints=[
            "Convert annotated NER datasets (CoNLL, OntoNotes) to this JSON shape via a small script.",
            "Use spaCy / Prodigy / Doccano to label your own corpus, then export JSONL.",
            "Synthesize examples from regex matches + a teacher model (Synthetic tab covers this).",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["text", "content", "snippet", "input"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["entities", "entities_json", "spans", "annotations", "labels_json"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.93,
            ),
        ],
    )


def _summarization_recipe() -> Recipe:
    return Recipe(
        id="summarization",
        name="Summarizer",
        headline="Train a model to write a short summary of a longer text.",
        description=(
            "For datasets where each row is a long-form document paired "
            "with a target summary. Meeting notes → executive summary, "
            "article → TL;DR, ticket history → resolution recap. The "
            "output should generally be much shorter than the input."
        ),
        icon="📝",
        task_profile="summarization",
        adapter_id="qa-pair",
        scoring_mode="field_match",
        default_input_column="document",
        default_output_column="summary",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "facebook/bart-large-cnn",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="document_summary_rationale",
            min_rows_recommended=75,
            fields=[
                GoldFieldSpec(name="document", description="The long-form source text."),
                GoldFieldSpec(name="summary", description="The reference summary (typically 1-5 sentences)."),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional: what makes this summary 'right' (key points covered, tone, length).",
                ),
            ],
            example_row={
                "document": "The board meeting on March 14 covered three topics ...",
                "summary": "Board approved Q1 budget, deferred hiring freeze, scheduled all-hands for April.",
                "rationale": "Captures the three concrete outcomes in one sentence.",
            },
        ),
        sample_eval_prompts=[
            "Long meeting transcript here — produce a 3-sentence executive summary.",
            "Multi-page support ticket thread — write a 1-sentence resolution.",
        ],
        data_acquisition_hints=[
            "CNN/DailyMail, XSum, SAMSum and similar public summarization datasets on Hugging Face.",
            "Internal documents paired with their existing executive summaries (most teams have these).",
            "Article → headline pairs from any news archive.",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["document", "article", "text", "source", "transcript", "body"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["summary", "tldr", "abstract", "headline", "recap"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.88,
            ),
        ],
    )


def _code_review_recipe() -> Recipe:
    return Recipe(
        id="code-review",
        name="Code Review Nitpicker",
        headline="Train a model to comment on code diffs the way your senior engineers do.",
        description=(
            "For datasets where each row is a code diff (or full file) paired "
            "with a review comment. Style nits, bug suggestions, "
            "convention reminders. Works on any language; the input is text "
            "and the output is a comment string, so existing GitHub PR data "
            "fits cleanly."
        ),
        icon="🧐",
        task_profile="instruction_sft",
        adapter_id="qa-pair",
        scoring_mode="field_match",
        default_input_column="diff",
        default_output_column="review",
        suggested_base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="diff_review_rationale",
            min_rows_recommended=100,
            fields=[
                GoldFieldSpec(name="diff", description="A code diff or full file the reviewer is commenting on."),
                GoldFieldSpec(name="review", description="The review comment — what a senior engineer would say."),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional: severity tag (nit / suggestion / blocker) or rule category.",
                ),
            ],
            example_row={
                "diff": "+ if (user.id == null) {\n+     return;\n+ }",
                "review": "Use `===` for null check, or `if (user.id != null)` to also catch undefined.",
                "rationale": "JS equality nit; the loose-equality form coerces both null and undefined.",
            },
        ),
        sample_eval_prompts=[
            "[ a Python diff that mutates a default argument ]",
            "[ a React component that uses useState inside a conditional ]",
            "[ a SQL query missing a LIMIT clause ]",
        ],
        data_acquisition_hints=[
            "Scrape your own team's accepted PR comments via the GitHub API (gh api repos/foo/pulls/<n>/comments).",
            "Use public datasets like 'cr-comments' on Hugging Face Hub.",
            "Manually curate examples from internal review guidelines — quality > quantity here.",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["diff", "patch", "code", "file", "snippet"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["review", "comment", "feedback", "suggestion"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.85,
            ),
        ],
    )


def _generic_sft_recipe() -> Recipe:
    return Recipe(
        id="generic-sft",
        name="Generic Instruction SFT",
        headline="The catch-all: any input → output text pair, no special structure.",
        description=(
            "When your task doesn't cleanly fit one of the more specific "
            "recipes — pick this and the platform applies sensible "
            "defaults. Free-form input, free-form output, exact-match "
            "evaluation. Easiest to start with; switch to a more specific "
            "recipe once your data shape is settled."
        ),
        icon="🧰",
        task_profile="instruction_sft",
        adapter_id="default-canonical",
        scoring_mode="field_match",
        default_input_column="input",
        default_output_column="output",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="input_output_rationale",
            min_rows_recommended=50,
            fields=[
                GoldFieldSpec(name="input", description="The prompt the model receives."),
                GoldFieldSpec(name="output", description="The reference answer the model should produce."),
                GoldFieldSpec(
                    name="rationale",
                    required=False,
                    description="Optional explanation field; ignored by basic eval handlers.",
                ),
            ],
            example_row={
                "input": "Translate to French: 'Hello, how are you?'",
                "output": "Bonjour, comment ça va ?",
                "rationale": "Standard informal greeting.",
            },
        ),
        sample_eval_prompts=[
            "Sample prompt 1 — replace with your own.",
            "Sample prompt 2 — replace with your own.",
            "Sample prompt 3 — replace with your own.",
        ],
        data_acquisition_hints=[
            "If you're not sure where to start, the other recipes' guidance probably applies — switch to one of them.",
            "Public 'instruction-tuning' datasets on Hugging Face: Alpaca, Dolly, OpenAssistant.",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["input", "prompt"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["output", "response", "completion"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.70,
            ),
        ],
    )


def _rag_protocol_recipe() -> Recipe:
    """Arc R-1 — protocol-aware RAG fine-tune.

    Trains a model on (context, question, answer-with-citation) tuples
    so the resulting model knows how to USE a RAG index correctly:
    cite the chunk it pulled, refuse when context is insufficient,
    and hew to a consistent response format. Domain-agnostic — the
    same recipe works for ecommerce FAQ, legal QA, support, and
    internal knowledge bases. Stage 2 (the customer's actual data)
    bolts on via the existing auto-RAG / RAG-first runtime; the
    recipe owns the *protocol*, not the *facts*.

    Pairs with the ``rag-grounded`` adapter (which already maps
    context/question/answer to RAGHandler) and the ``rag_qa`` task
    profile (which scores SQuAD EM/F1 + faithfulness +
    context_recall).

    Curated playbooks generate three signal types from gold seeds:
      - POSITIVES_PARAPHRASE → citation drills (answers must
        reference the chunk id)
      - REFUSALS             → context-insufficient examples with
        templated refusal copy
      - FORMAT_ROBUSTNESS    → varied question phrasings demanding
        identical answer formatting
    """
    return Recipe(
        id="rag-protocol",
        name="Protocol-aware RAG Assistant",
        headline="Train a model to USE a retrieval index correctly — cite, refuse, format consistently.",
        description=(
            "For projects that deploy with retrieval (auto-RAG / RAG-first). "
            "The recipe trains the model on RAG-shaped triples (context, "
            "question, answer-with-citation) so it learns to cite the "
            "chunk it used, refuse when context is insufficient, and "
            "produce consistently-formatted answers. The customer-"
            "specific facts come from their own BM25 index at inference "
            "time — the recipe owns the protocol, not the domain."
        ),
        icon="📚",
        task_profile="rag_qa",
        adapter_id="rag-grounded",
        scoring_mode="field_match",
        default_input_column="question",
        default_output_column="answer",
        suggested_base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        alt_base_models=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ],
        target_profile="vllm_server",
        training_plan_profile="balanced",
        eval_pack_id="evalpack.general.default",
        gold_template=GoldTemplate(
            shape_label="context_question_answer_with_citation",
            min_rows_recommended=60,
            fields=[
                GoldFieldSpec(
                    name="context",
                    description=(
                        "The retrieved chunk(s) the model should ground its "
                        "answer in. Multi-chunk context can be concatenated "
                        "with [#1] / [#2] markers."
                    ),
                ),
                GoldFieldSpec(
                    name="question",
                    description="The user-facing question.",
                ),
                GoldFieldSpec(
                    name="answer",
                    description=(
                        "The grounded answer. Include the chunk citation "
                        "(e.g. [#1]) when the answer pulls from a specific "
                        "passage. For refusal cases, use the templated "
                        "phrase \"I don't have enough context to answer "
                        "that.\""
                    ),
                ),
            ],
            example_row={
                "context": "[#1] Our refund policy allows returns within 30 days of delivery for unused items in original packaging.",
                "question": "How long do I have to return an item?",
                "answer": "You have 30 days from delivery to return unused items in their original packaging [#1].",
            },
        ),
        sample_eval_prompts=[
            "Context: [#1] Free standard shipping on orders over $50. → Question: When does free shipping kick in?",
            "Context: [#1] Premium tier includes 24/7 phone support. → Question: What's the refund window?",
            "Context: (none) → Question: What's your return policy?",
        ],
        data_acquisition_hints=[
            "Pair existing FAQ entries with their source paragraph (the context) — most knowledge bases already have this implicit linkage.",
            "Generate refusal examples synthetically: for each context-irrelevant question, the answer should be the templated 'not enough context' phrase.",
            "Mine support transcripts: agent answers cite specific KB articles — that linkage is your context-question-answer triple.",
        ],
        shape_signatures=[
            ShapeSignature(
                columns=[
                    ShapeColumn(
                        name_patterns=["context", "passage", "chunk", "source", "evidence"],
                        column_role="auxiliary",
                    ),
                    ShapeColumn(
                        name_patterns=["question", "query", "prompt", "q"],
                        column_role="input",
                    ),
                    ShapeColumn(
                        name_patterns=["answer", "response", "output", "a"],
                        column_role="output",
                    ),
                ],
                base_confidence=0.86,
            ),
        ],
    )


_BUILTIN_RECIPE_FACTORIES = [
    _qa_sft_recipe,
    _classification_recipe,
    _span_extraction_recipe,
    _summarization_recipe,
    _code_review_recipe,
    _generic_sft_recipe,
    _rag_protocol_recipe,
]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def _build_registry() -> dict[str, Recipe]:
    """Build the recipe registry. Each call rebuilds — recipes are
    cheap to construct and the registry is small, so we avoid the
    cache-invalidation headache of memoizing this."""
    registry: dict[str, Recipe] = {}
    for factory in _BUILTIN_RECIPE_FACTORIES:
        recipe = factory()
        registry[recipe.id] = recipe
    return registry


def list_recipes() -> list[Recipe]:
    """List all built-in recipes in catalog order."""
    return [factory() for factory in _BUILTIN_RECIPE_FACTORIES]


def get_recipe(recipe_id: str) -> Recipe | None:
    """Fetch a single recipe by id. Returns None for unknown ids."""
    registry = _build_registry()
    return registry.get(recipe_id)


def list_recipe_catalog() -> dict[str, Any]:
    """Catalog payload for API responses — wraps the recipe list
    with metadata about the catalog itself (version, source)."""
    return {
        "catalog_version": RECIPE_CATALOG_VERSION,
        "catalog_source": "builtin",
        "recipe_count": len(_BUILTIN_RECIPE_FACTORIES),
        "recipes": [copy.deepcopy(r.model_dump()) for r in list_recipes()],
    }


def list_supported_task_profiles_for_recipes() -> list[str]:
    """Task profiles referenced by at least one built-in recipe.
    Used by validation to make sure recipes only reference known
    task profiles."""
    return sorted({r.task_profile for r in list_recipes()})


# ─────────────────────────────────────────────────────────────────────
# task_family / task_profile → catalog recipe defaults
#
# Used by the brief-driven `POST /api/projects` and `/magic-create`
# code paths to auto-apply a task-shape recipe at creation time so
# `Project.selected_recipe` is never NULL on a freshly-created
# non-templated project. Downstream consumers (synth playbook
# runner, auto-RAG comparison, post-eval reroute analyzer, several
# Coach Mode signals) read `recipe_id` off the selected_recipe
# snapshot and hard-fail when it's missing.
#
# The brief analyzer (`domain_blueprint_service._infer_task_family`)
# emits a `task_family` token; magic-create's recommendation carries
# a `task_profile` token. The two vocabularies overlap but aren't
# identical — keep two helpers so each call site uses the field it
# already has on hand instead of remapping on the way in.
#
# Unknown / "instruction_sft" / catch-all → "generic-sft" rather
# than NULL: the generic SFT recipe is the safest fallback and the
# user can override via the DatasetImportWizard's recipe picker.
# ─────────────────────────────────────────────────────────────────────


_TASK_FAMILY_TO_RECIPE_ID: dict[str, str] = {
    "qa": "qa-sft",
    "rag_qa": "qa-sft",
    "classification": "classification",
    "structured_extraction": "span-extraction",
    "summarization": "summarization",
    "instruction_sft": "generic-sft",
}


_TASK_PROFILE_TO_RECIPE_ID: dict[str, str] = {
    # Subset of the magic-create vocabulary that maps to a real
    # catalog recipe. `tool_calling`, `preference`, `seq2seq`, and
    # `chat_sft` don't have dedicated catalog recipes yet, so they
    # all fall back to generic-sft below.
    "qa": "qa-sft",
    "rag_qa": "qa-sft",
    "classification": "classification",
    "structured_extraction": "span-extraction",
    "summarization": "summarization",
    "instruction_sft": "generic-sft",
}


_FALLBACK_RECIPE_ID = "generic-sft"


def default_recipe_for_task_family(task_family: str | None) -> str:
    """Map a brief-analyzer `task_family` token to a catalog recipe
    id. Returns ``"generic-sft"`` for unknown / empty inputs so the
    caller never has to special-case a NULL recipe.

    The result is guaranteed to be a key in the recipe catalog —
    any future renames to the catalog should update this map in
    lockstep (see ``test_default_recipe_helpers`` for the guard)."""
    token = (task_family or "").strip().lower()
    return _TASK_FAMILY_TO_RECIPE_ID.get(token, _FALLBACK_RECIPE_ID)


def default_recipe_for_task_profile(task_profile: str | None) -> str:
    """Map a magic-create `task_profile` token to a catalog recipe
    id. Mirrors ``default_recipe_for_task_family`` for the second
    non-templated create path which carries `task_profile` strings
    instead of `task_family`."""
    token = (task_profile or "").strip().lower()
    return _TASK_PROFILE_TO_RECIPE_ID.get(token, _FALLBACK_RECIPE_ID)


# ─────────────────────────────────────────────────────────────────────
# Shape sniffer
#
# Header-based recipe suggester. Given a list of column headers from a
# fresh CSV / JSONL / Parquet upload, score each recipe by how many of
# its shape signatures' columns find a name-pattern match in the file
# headers. Returns a ranked list of (recipe_id, confidence) so the UI
# can land on a recommended recipe at file-pick time.
#
# This is intentionally header-only — it does NOT inspect cell content.
# Content sniffing is the job of the existing dataset_import
# introspector (which runs after the user confirms a recipe). The
# split keeps recipe selection fast and offline-friendly: a user can
# audit the suggestion before any rows are read.
# ─────────────────────────────────────────────────────────────────────


def _normalize_header(header: str) -> str:
    """Lowercase + strip + collapse non-alphanumerics for matching."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in header).strip("_")


def _column_matches_header(column: ShapeColumn, normalized_headers: list[str]) -> str | None:
    """Return the header that matched, or None if no pattern hit."""
    for header in normalized_headers:
        for pattern in column.name_patterns:
            needle = _normalize_header(pattern)
            if not needle:
                continue
            if header == needle or header.startswith(needle + "_") or header.endswith("_" + needle) or needle in header.split("_"):
                return header
    return None


def _score_signature(
    signature: ShapeSignature,
    normalized_headers: list[str],
) -> tuple[float, dict[str, str | None]]:
    """Score one signature against a list of normalized headers.

    Returns (confidence, matched_columns) where matched_columns maps
    role -> matched header (or None). All required columns must have
    a match for the signature to score above zero.
    """
    matches: dict[str, str | None] = {}
    required_hits = 0
    required_total = 0
    optional_hits = 0
    optional_total = 0

    for column in signature.columns:
        matched_header = _column_matches_header(column, normalized_headers)
        matches[column.column_role] = matched_header
        if column.required:
            required_total += 1
            if matched_header:
                required_hits += 1
        else:
            optional_total += 1
            if matched_header:
                optional_hits += 1

    if required_total > 0 and required_hits < required_total:
        return 0.0, matches

    # All required hit. Confidence = base * (1 - 0.1 * missing-optionals).
    optional_penalty = 0.0
    if optional_total:
        optional_penalty = 0.1 * (optional_total - optional_hits) / optional_total
    confidence = signature.base_confidence * (1.0 - optional_penalty)
    return round(confidence, 4), matches


def sniff_recipe_from_headers(headers: list[str]) -> list[dict[str, Any]]:
    """Rank recipes by how well their shape signatures match a list
    of column headers.

    Returns a list of dicts (ordered best-first):
    `{recipe_id, recipe_name, confidence, matched_columns, signature_index}`.
    Entries with confidence 0 (required columns missing) are dropped.

    The list always ends with the `generic-sft` recipe at a small
    floor confidence so the UI never lands on "no suggestion" — even
    if nothing matches cleanly, the generic recipe is a sensible
    last resort.
    """
    normalized = [_normalize_header(h) for h in headers if isinstance(h, str) and h.strip()]
    suggestions: list[dict[str, Any]] = []

    for recipe in list_recipes():
        best: tuple[float, dict[str, str | None], int] | None = None
        for idx, signature in enumerate(recipe.shape_signatures):
            confidence, matches = _score_signature(signature, normalized)
            if confidence <= 0:
                continue
            if best is None or confidence > best[0]:
                best = (confidence, matches, idx)
        if best is None:
            continue
        suggestions.append(
            {
                "recipe_id": recipe.id,
                "recipe_name": recipe.name,
                "icon": recipe.icon,
                "confidence": best[0],
                "matched_columns": best[1],
                "signature_index": best[2],
            }
        )

    suggestions.sort(key=lambda s: -s["confidence"])

    # Always include generic-sft as a floor fallback if it wasn't
    # already in the ranked results.
    if not any(s["recipe_id"] == "generic-sft" for s in suggestions):
        generic = get_recipe("generic-sft")
        if generic is not None:
            suggestions.append(
                {
                    "recipe_id": generic.id,
                    "recipe_name": generic.name,
                    "icon": generic.icon,
                    "confidence": 0.30,
                    "matched_columns": {},
                    "signature_index": 0,
                    "fallback": True,
                }
            )

    return suggestions
