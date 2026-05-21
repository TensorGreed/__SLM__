/**
 * Typed wrapper for the brief-analysis endpoint
 * `POST /api/domain-blueprints/analyze`.
 *
 * Used by the brief-driven create modal to power the
 * decision-engine chip (Theme 7) — "do you even need SFT?" —
 * before the user commits to creating the project.
 */

import api from './client';

export type ApproachKind =
    | 'prompt_only'
    | 'rag_first'
    | 'sft'
    | 'dpo'
    | 'distillation';

export interface ApproachRecommendation {
    approach: ApproachKind;
    confidence: number;
    headline: string;
    rationale: string;
    signals: string[];
    cta_label: string;
}

export interface AnalyzeBriefResponse {
    blueprint?: {
        task_family?: string;
        input_modality?: string;
        confidence_score?: number;
        [key: string]: unknown;
    };
    validation?: {
        ok?: boolean;
        warnings?: Array<{ message?: string }>;
        errors?: Array<{ message?: string }>;
    };
    guidance?: {
        recommended_next_actions?: string[];
        unresolved_questions?: string[];
    };
    recommended_approach?: ApproachRecommendation | null;
}

export interface AnalyzeBriefRequest {
    brief_text: string;
    sample_inputs?: string[];
    sample_outputs?: string[];
    deployment_target?: string;
    /** Skip the LLM enrichment pass — the decision engine is
     * pure-Python and runs regardless. Set false for the in-modal
     * debounced call so chip latency stays in single-digit ms. */
    llm_enrich?: boolean;
}

export async function analyzeBrief(
    payload: AnalyzeBriefRequest,
): Promise<AnalyzeBriefResponse> {
    const res = await api.post<AnalyzeBriefResponse>(
        '/domain-blueprints/analyze',
        {
            llm_enrich: payload.llm_enrich ?? false,
            ...payload,
        },
    );
    return res.data;
}
