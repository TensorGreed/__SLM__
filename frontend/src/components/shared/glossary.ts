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
};

export function getTermDefinition(id: string): TermDefinition | null {
    return TERM_DEFINITIONS[id] || null;
}
