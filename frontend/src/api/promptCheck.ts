/**
 * Dry run of the benchmark-value gate: does a seed-documents or
 * span-synthesis prompt carry a value from a benchmark's gold rows?
 * ``POST /projects/{id}/synthetic/prompt-check``. The generating routes run
 * the same check and answer 400 ``PROMPT_CARRIES_BENCHMARK_VALUES`` unless
 * the request sets ``ignore_benchmark_values``.
 */

import api from './client';

export const PROMPT_CONTAMINATION_CODE = 'PROMPT_CARRIES_BENCHMARK_VALUES';

export interface PromptOffendingValue {
    value: string;
    type: string;
    benchmarks: string[];
    /** The prompt fields it occurs in (``kinds``, ``genres``, ``entity_types``, ``definition:<type>``…). */
    fields: string[];
}

export interface PromptCheckReport {
    clean: boolean;
    offending: PromptOffendingValue[];
    offending_total: number;
    values_checked: number;
    benchmarks: string[];
    fields: string[];
}

export async function checkPrompt(
    projectId: number,
    fields: Record<string, string | string[]>,
    includeConventions = false,
): Promise<PromptCheckReport> {
    const res = await api.post<PromptCheckReport>(`/projects/${projectId}/synthetic/prompt-check`, {
        fields,
        include_conventions: includeConventions,
    });
    return res.data;
}

const FIELD_LABEL: Record<string, string> = {
    kinds: 'what the documents must contain',
    genres: 'the genres',
    domain_hint: 'the domain hint',
    entity_types: 'the entity types',
    focus_types: 'the focus types',
};

function fieldLabel(name: string): string {
    if (name.startsWith('definition:')) return `the definition of ${name.slice('definition:'.length)}`;
    if (name.startsWith('pattern:')) return `the pattern note of ${name.slice('pattern:'.length)}`;
    return FIELD_LABEL[name] ?? name;
}

/** One line for the Generate step. */
export function promptCheckText(report: PromptCheckReport): string {
    if (report.benchmarks.length === 0) return 'No benchmarks on this project yet, so there is nothing to collide with.';
    if (report.clean) {
        return `Clean: none of the ${report.values_checked} gold values of ${report.benchmarks.join(', ')} appear in the prompt.`;
    }
    const items = report.offending.slice(0, 6).map((o) => `${o.value} (${o.type}, in ${o.fields.map(fieldLabel).join(' and ')})`);
    const more = report.offending_total > 6 ? ` and ${report.offending_total - 6} more` : '';
    return `${report.offending_total} benchmark value${report.offending_total === 1 ? '' : 's'} in the prompt: ${items.join('; ')}${more}. ` +
        'A generation model copies examples word for word, so describe the shape instead.';
}

/** The offending values an axios 400 carries (the envelope hoists ``metadata``). */
export function offendingValuesFromError(err: unknown): PromptOffendingValue[] | null {
    const data = (err as { response?: { data?: { error_code?: unknown; metadata?: { offending?: unknown } } } })?.response?.data;
    if (!data || data.error_code !== PROMPT_CONTAMINATION_CODE) return null;
    const offending = data.metadata?.offending;
    return Array.isArray(offending) ? (offending as PromptOffendingValue[]) : [];
}
