/**
 * Concept-ID → beginner-friendly term mapping.
 *
 * The popover body is sourced from the backend glossary (GET /domain-blueprints/glossary/help)
 * whenever available; the entries below provide the canonical concept IDs, the beginner label
 * shown in UI, and a fallback plain-language description so that the UI remains usable if the
 * backend glossary has not been loaded yet.
 *
 * Concept IDs intentionally use snake_case so they line up with glossary keys from the backend
 * (both BUILTIN_GLOSSARY in domain_blueprint_service.py and project-scoped blueprint glossaries).
 */

export interface TermDefinition {
    id: string;
    /** The jargon label (what engineers say). */
    advancedLabel: string;
    /** The beginner-friendly label used when rendering. */
    beginnerLabel: string;
    /** Backend glossary term this definition is indexed by (case-insensitive match). */
    glossaryKey: string;
    /** Category hint (mirrors backend GlossaryEntry.category). */
    category: string;
    /** Plain-language fallback, used until the backend glossary is loaded. */
    fallback: string;
    /**
     * Arc G — deep-link to the matching BrewSLM Academy lesson on
     * brewslm.com. Rendered as a "Learn more →" footer in the
     * popover when present. Keep the path relative to the
     * brewslm.com root so the same constant works in dev, prod, and
     * any future preview deploy.
     *
     * Slugs must stay in sync with files under
     * ``__SLM__website/academy/<track>/<lesson>.html``. The Academy
     * playlist on YouTube covers each track end-to-end; per-lesson
     * pages on the website are the deeper read.
     */
    academyUrl?: string;
}

const ACADEMY_ROOT = 'https://brewslm.com/academy';

export const TERM_DEFINITIONS: Record<string, TermDefinition> = {
    domain_pack: {
        id: 'domain_pack',
        advancedLabel: 'Domain Pack',
        beginnerLabel: 'Domain Kit',
        glossaryKey: 'domain pack',
        category: 'domain',
        fallback: 'A bundle of defaults and policy overlays for a domain use case.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/recipes-and-handlers.html`,
    },
    domain_profile: {
        id: 'domain_profile',
        advancedLabel: 'Domain Profile',
        beginnerLabel: 'Domain Settings',
        glossaryKey: 'domain profile',
        category: 'domain',
        fallback: 'A typed configuration profile for runtime and evaluation behavior.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/recipes-and-handlers.html`,
    },
    pack: {
        id: 'pack',
        advancedLabel: 'Pack',
        beginnerLabel: 'Kit',
        glossaryKey: 'pack',
        category: 'domain',
        fallback: 'A reusable bundle of domain defaults, prompts, and guardrails.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/recipes-and-handlers.html`,
    },
    recipe: {
        id: 'recipe',
        advancedLabel: 'Recipe',
        beginnerLabel: 'Training Plan',
        glossaryKey: 'recipe',
        category: 'training',
        fallback: 'A saved, reusable training plan — base model, adapter, data, and eval settings bundled together.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/recipes-and-handlers.html`,
    },
    adapter: {
        id: 'adapter',
        advancedLabel: 'Adapter',
        beginnerLabel: 'Data Mapper',
        glossaryKey: 'adapter',
        category: 'data',
        fallback: 'A mapping layer that converts source data into training-ready fields.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/ingest-and-map.html`,
    },
    runtime: {
        id: 'runtime',
        advancedLabel: 'Runtime',
        beginnerLabel: 'Training Backend',
        glossaryKey: 'runtime',
        category: 'deployment',
        fallback: 'The backend that actually runs training or inference.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/training-config-reference.html`,
    },
    gate: {
        id: 'gate',
        advancedLabel: 'Gate',
        beginnerLabel: 'Pass/Fail Check',
        glossaryKey: 'gate',
        category: 'evaluation',
        fallback: 'A pass/fail threshold on an evaluation metric — if the gate fails, the model is not promoted.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/eval-packs-and-failure-clusters.html`,
    },
    gold_set: {
        id: 'gold_set',
        advancedLabel: 'Gold Set',
        beginnerLabel: 'Reference Set',
        glossaryKey: 'gold set',
        category: 'evaluation',
        fallback: 'A trusted reference dataset used for evaluation and regression checks.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/gold-sets.html`,
    },
    hallucination: {
        id: 'hallucination',
        advancedLabel: 'Hallucination',
        beginnerLabel: 'Made-up Answer',
        glossaryKey: 'hallucination',
        category: 'safety',
        fallback: 'A response that sounds plausible but is not supported by source data.',
        academyUrl: `${ACADEMY_ROOT}/foundations/how-language-models-work.html`,
    },
    blueprint: {
        id: 'blueprint',
        advancedLabel: 'Blueprint',
        beginnerLabel: 'Domain Plan',
        glossaryKey: 'blueprint',
        category: 'domain',
        fallback: 'A normalized domain plan generated from your brief.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/from-script-to-platform.html`,
    },
    autopilot: {
        id: 'autopilot',
        advancedLabel: 'Autopilot',
        beginnerLabel: 'Autopilot',
        glossaryKey: 'autopilot',
        category: 'training',
        fallback: 'A guided mode that proposes and runs a safe training plan for you, with every decision explained.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/from-script-to-platform.html`,
    },
    preflight: {
        id: 'preflight',
        advancedLabel: 'Preflight',
        beginnerLabel: 'Pre-launch Check',
        glossaryKey: 'preflight',
        category: 'operations',
        fallback: 'A validation step that checks compatibility before training or deployment.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/preflight-and-trainability.html`,
    },
    // ── Coach Mode terms (USER-SUCCESS Epic 4 Phase 4) ─────────────
    // These get auto-wrapped in CoachSuggestion bodies via a regex
    // dictionary scan; ``glossaryKey`` is kept lower-case + space-
    // separated so the backend glossary endpoint can resolve them.
    f1: {
        id: 'f1',
        advancedLabel: 'F1',
        beginnerLabel: 'F1',
        glossaryKey: 'f1',
        category: 'evaluation',
        fallback: 'A single score that combines precision (how often you\'re right when you say yes) and recall (how many real yeses you catch). 0.70 is good for narrow tasks, 0.90+ is excellent. Below 0.50 usually means structural data issues.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/evaluation-fundamentals.html`,
    },
    pass_rate: {
        id: 'pass_rate',
        advancedLabel: 'Pass Rate',
        beginnerLabel: 'Pass Rate',
        glossaryKey: 'pass rate',
        category: 'evaluation',
        fallback: 'The share of eval rows the model got right. 0.90 is the healthy zone for most narrow tasks; below 0.60 means the model is struggling enough that more data or a stronger base model is needed.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/eval-packs-and-failure-clusters.html`,
    },
    shannon_entropy: {
        id: 'shannon_entropy',
        advancedLabel: 'Shannon Entropy',
        beginnerLabel: 'Shannon Entropy',
        glossaryKey: 'shannon entropy',
        category: 'evaluation',
        fallback: 'A number from 0 upward that measures how evenly classes are spread. 0 means everything is one class; higher numbers mean more even spread. Below 1.0 is "skewed enough to worry about"; below 0.5 is severe class imbalance.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/data-quality-splits.html`,
    },
    jaccard_similarity: {
        id: 'jaccard_similarity',
        advancedLabel: 'Jaccard Similarity',
        beginnerLabel: 'Jaccard Similarity',
        glossaryKey: 'jaccard similarity',
        category: 'evaluation',
        fallback: 'A 0-1 score for how much two texts share. 0 = no overlap, 1 = identical. Coach uses the *average* across your gold set\'s rows: high means your examples all look the same, which leaves the model nothing to learn from.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/gold-sets.html`,
    },
    class_imbalance: {
        id: 'class_imbalance',
        advancedLabel: 'Class Imbalance',
        beginnerLabel: 'Class Imbalance',
        glossaryKey: 'class imbalance',
        category: 'evaluation',
        fallback: 'When one label dominates your data, the model learns to over-predict it. Coach flags this when any class falls below 15% of the total. Fix by generating more examples for the rare classes.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/data-quality-splits.html`,
    },
    failure_cluster: {
        id: 'failure_cluster',
        advancedLabel: 'Failure Cluster',
        beginnerLabel: 'Failure Cluster',
        glossaryKey: 'failure cluster',
        category: 'evaluation',
        fallback: 'A group of eval failures that share a pattern (e.g. "the model hallucinates dates" or "wrong output shape"). Targeting the biggest cluster with synthetic data lifts the most failed rows for the same generation budget.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/eval-packs-and-failure-clusters.html`,
    },
    predicted_f1_confidence: {
        id: 'predicted_f1_confidence',
        advancedLabel: 'Predicted Pass Probability',
        beginnerLabel: 'Predicted Pass Probability',
        glossaryKey: 'predicted pass probability',
        category: 'evaluation',
        fallback: 'BrewSLM\'s forecast of how likely your run is to pass the eval gate, computed from row count, class balance, gold-set diversity, and base-model size. Below 40% = likely fail; above 65% = likely pass; in between = borderline.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/preflight-and-trainability.html`,
    },
    // ── Arc G new entries (close visible gaps in Data Studio / Coach) ──
    lora: {
        id: 'lora',
        advancedLabel: 'LoRA',
        beginnerLabel: 'LoRA',
        glossaryKey: 'lora',
        category: 'training',
        fallback: 'Low-Rank Adaptation: a parameter-efficient training method that adds a small trainable adapter on top of a frozen base model. Fast, cheap to host, and avoids overwriting the base — the BrewSLM default.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/full-finetuning-vs-lora.html`,
    },
    chat_template: {
        id: 'chat_template',
        advancedLabel: 'Chat Template',
        beginnerLabel: 'Chat Template',
        glossaryKey: 'chat template',
        category: 'training',
        fallback: 'The exact wrapping format (special tokens + role markers) the tokenizer expects for chat input. A mismatch here is the single most common cause of low eval F1 — the model literally never saw the eval prompt format during training.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/chat-templates.html`,
    },
    auto_rag: {
        id: 'auto_rag',
        advancedLabel: 'Auto-RAG',
        beginnerLabel: 'Auto-RAG',
        glossaryKey: 'auto rag',
        category: 'evaluation',
        fallback: 'BrewSLM builds a BM25 index from your training data after each run and prepends the top-K retrievals to playground turns. Useful for QA shapes where the model would otherwise have to memorise every fact.',
        academyUrl: `${ACADEMY_ROOT}/with-brewslm/auto-rag-and-reroute.html`,
    },
    task_shape: {
        id: 'task_shape',
        advancedLabel: 'Task Shape',
        beginnerLabel: 'Task Shape',
        glossaryKey: 'task shape',
        category: 'data',
        fallback: 'The structural pattern of inputs and outputs your task needs: classification, QA-pair, structured extraction, span tagging, RAG-grounded, seq2seq, etc. BrewSLM uses task shape (not the domain) to pick the right adapter + handler.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/task-shapes.html`,
    },
    cross_entropy_loss: {
        id: 'cross_entropy_loss',
        advancedLabel: 'Cross-Entropy Loss',
        beginnerLabel: 'Cross-Entropy Loss',
        glossaryKey: 'cross entropy loss',
        category: 'training',
        fallback: 'The default loss function for language models: penalises the model for assigning low probability to the correct next token. Lower is better; trends matter more than absolute values.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/cross-entropy-loss.html`,
    },
    learning_rate: {
        id: 'learning_rate',
        advancedLabel: 'Learning Rate',
        beginnerLabel: 'Learning Rate',
        glossaryKey: 'learning rate',
        category: 'training',
        fallback: 'How big a step the optimiser takes on each update. Too high = loss explodes; too low = training is glacial. For LoRA on small models 1e-4 to 5e-4 is the usual safe band.',
        academyUrl: `${ACADEMY_ROOT}/sft-fundamentals/learning-rate-and-schedules.html`,
    },
};

export function getTermDefinition(id: string): TermDefinition | null {
    return TERM_DEFINITIONS[id] || null;
}
