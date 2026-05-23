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
}

export const TERM_DEFINITIONS: Record<string, TermDefinition> = {
    domain_pack: {
        id: 'domain_pack',
        advancedLabel: 'Domain Pack',
        beginnerLabel: 'Domain Kit',
        glossaryKey: 'domain pack',
        category: 'domain',
        fallback: 'A bundle of defaults and policy overlays for a domain use case.',
    },
    domain_profile: {
        id: 'domain_profile',
        advancedLabel: 'Domain Profile',
        beginnerLabel: 'Domain Settings',
        glossaryKey: 'domain profile',
        category: 'domain',
        fallback: 'A typed configuration profile for runtime and evaluation behavior.',
    },
    pack: {
        id: 'pack',
        advancedLabel: 'Pack',
        beginnerLabel: 'Kit',
        glossaryKey: 'pack',
        category: 'domain',
        fallback: 'A reusable bundle of domain defaults, prompts, and guardrails.',
    },
    recipe: {
        id: 'recipe',
        advancedLabel: 'Recipe',
        beginnerLabel: 'Training Plan',
        glossaryKey: 'recipe',
        category: 'training',
        fallback: 'A saved, reusable training plan — base model, adapter, data, and eval settings bundled together.',
    },
    adapter: {
        id: 'adapter',
        advancedLabel: 'Adapter',
        beginnerLabel: 'Data Mapper',
        glossaryKey: 'adapter',
        category: 'data',
        fallback: 'A mapping layer that converts source data into training-ready fields.',
    },
    runtime: {
        id: 'runtime',
        advancedLabel: 'Runtime',
        beginnerLabel: 'Training Backend',
        glossaryKey: 'runtime',
        category: 'deployment',
        fallback: 'The backend that actually runs training or inference.',
    },
    gate: {
        id: 'gate',
        advancedLabel: 'Gate',
        beginnerLabel: 'Pass/Fail Check',
        glossaryKey: 'gate',
        category: 'evaluation',
        fallback: 'A pass/fail threshold on an evaluation metric — if the gate fails, the model is not promoted.',
    },
    gold_set: {
        id: 'gold_set',
        advancedLabel: 'Gold Set',
        beginnerLabel: 'Reference Set',
        glossaryKey: 'gold set',
        category: 'evaluation',
        fallback: 'A trusted reference dataset used for evaluation and regression checks.',
    },
    hallucination: {
        id: 'hallucination',
        advancedLabel: 'Hallucination',
        beginnerLabel: 'Made-up Answer',
        glossaryKey: 'hallucination',
        category: 'safety',
        fallback: 'A response that sounds plausible but is not supported by source data.',
    },
    blueprint: {
        id: 'blueprint',
        advancedLabel: 'Blueprint',
        beginnerLabel: 'Domain Plan',
        glossaryKey: 'blueprint',
        category: 'domain',
        fallback: 'A normalized domain plan generated from your brief.',
    },
    autopilot: {
        id: 'autopilot',
        advancedLabel: 'Autopilot',
        beginnerLabel: 'Autopilot',
        glossaryKey: 'autopilot',
        category: 'training',
        fallback: 'A guided mode that proposes and runs a safe training plan for you, with every decision explained.',
    },
    preflight: {
        id: 'preflight',
        advancedLabel: 'Preflight',
        beginnerLabel: 'Pre-launch Check',
        glossaryKey: 'preflight',
        category: 'operations',
        fallback: 'A validation step that checks compatibility before training or deployment.',
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
    },
    pass_rate: {
        id: 'pass_rate',
        advancedLabel: 'Pass Rate',
        beginnerLabel: 'Pass Rate',
        glossaryKey: 'pass rate',
        category: 'evaluation',
        fallback: 'The share of eval rows the model got right. 0.90 is the healthy zone for most narrow tasks; below 0.60 means the model is struggling enough that more data or a stronger base model is needed.',
    },
    shannon_entropy: {
        id: 'shannon_entropy',
        advancedLabel: 'Shannon Entropy',
        beginnerLabel: 'Shannon Entropy',
        glossaryKey: 'shannon entropy',
        category: 'evaluation',
        fallback: 'A number from 0 upward that measures how evenly classes are spread. 0 means everything is one class; higher numbers mean more even spread. Below 1.0 is "skewed enough to worry about"; below 0.5 is severe class imbalance.',
    },
    jaccard_similarity: {
        id: 'jaccard_similarity',
        advancedLabel: 'Jaccard Similarity',
        beginnerLabel: 'Jaccard Similarity',
        glossaryKey: 'jaccard similarity',
        category: 'evaluation',
        fallback: 'A 0-1 score for how much two texts share. 0 = no overlap, 1 = identical. Coach uses the *average* across your gold set\'s rows: high means your examples all look the same, which leaves the model nothing to learn from.',
    },
    class_imbalance: {
        id: 'class_imbalance',
        advancedLabel: 'Class Imbalance',
        beginnerLabel: 'Class Imbalance',
        glossaryKey: 'class imbalance',
        category: 'evaluation',
        fallback: 'When one label dominates your data, the model learns to over-predict it. Coach flags this when any class falls below 15% of the total. Fix by generating more examples for the rare classes.',
    },
    failure_cluster: {
        id: 'failure_cluster',
        advancedLabel: 'Failure Cluster',
        beginnerLabel: 'Failure Cluster',
        glossaryKey: 'failure cluster',
        category: 'evaluation',
        fallback: 'A group of eval failures that share a pattern (e.g. "the model hallucinates dates" or "wrong output shape"). Targeting the biggest cluster with synthetic data lifts the most failed rows for the same generation budget.',
    },
    predicted_f1_confidence: {
        id: 'predicted_f1_confidence',
        advancedLabel: 'Predicted Pass Probability',
        beginnerLabel: 'Predicted Pass Probability',
        glossaryKey: 'predicted pass probability',
        category: 'evaluation',
        fallback: 'BrewSLM\'s forecast of how likely your run is to pass the eval gate, computed from row count, class balance, gold-set diversity, and base-model size. Below 40% = likely fail; above 65% = likely pass; in between = borderline.',
    },
};

export function getTermDefinition(id: string): TermDefinition | null {
    return TERM_DEFINITIONS[id] || null;
}
