import api from './client';

/** One model's tokens + list-price estimate over the window. */
export interface LlmSpendModelRow {
    model: string;
    calls: number;
    input_tokens: number;
    output_tokens: number;
    cache_write_tokens: number;
    cache_read_tokens: number;
    total_tokens: number;
    cost_usd: number;
    /** False when the model's price is unknown (its tokens carry no $). */
    priced: boolean;
}

export interface LlmSpendKindRow {
    kind: string;
    operations: number;
    calls: number;
    total_tokens: number;
    cost_usd: number;
}

/** GET /projects/{id}/llm-spend — the project's cloud-LLM spend tally. */
export interface LlmSpend {
    project_id: number;
    days: number;
    since: string;
    totals: {
        operations: number;
        calls: number;
        input_tokens: number;
        output_tokens: number;
        cache_write_tokens: number;
        cache_read_tokens: number;
        total_tokens: number;
        cost_usd: number;
        unpriced_tokens: number;
    };
    by_model: LlmSpendModelRow[];
    by_kind: LlmSpendKindRow[];
    basis: string;
    note: string;
}

export async function fetchLlmSpend(projectId: number, days = 30): Promise<LlmSpend> {
    const res = await api.get<LlmSpend>(`/projects/${projectId}/llm-spend`, { params: { days } });
    return res.data;
}
