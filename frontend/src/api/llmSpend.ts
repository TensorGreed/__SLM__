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
    /** Ledger operations whose tokens were estimated (a stream with no usage event). */
    estimated_operations?: number;
    /** Calendar month to date (UTC), whatever the window. */
    month?: { since: string; operations: number; total_tokens: number; cost_usd: number };
    budget?: LlmBudget | null;
    budget_status?: LlmBudgetStatus | null;
    basis: string;
    note: string;
}

/** runtime_config["llm_budget_usd"] — advisory, never blocks anything. */
export interface LlmBudget {
    monthly: number;
    warn_at: number;
}

export interface LlmBudgetStatus {
    budget: number;
    warn_at: number;
    spent: number;
    share: number;
    status: 'ok' | 'warning' | 'over';
    month_start?: string;
}

/** A Job-start response's ``budget_warning`` (the month is over budget; the Job runs). */
export interface LlmBudgetWarning extends LlmBudgetStatus {
    message: string;
}

export interface LlmBudgetPayload {
    project_id: number;
    budget: LlmBudget | null;
    status: LlmBudgetStatus | null;
}

/** PUT /projects/{id}/llm-budget — ``monthly: null`` clears it. */
export async function saveLlmBudget(
    projectId: number,
    monthly: number | null,
    warnAt?: number,
): Promise<LlmBudgetPayload> {
    const res = await api.put<LlmBudgetPayload>(`/projects/${projectId}/llm-budget`, {
        monthly,
        ...(warnAt !== undefined ? { warn_at: warnAt } : {}),
    });
    return res.data;
}

/** The ``budget_warning`` message on a Job-start response, if any. */
export function budgetWarningMessage(data: unknown): string | null {
    const w = (data as { budget_warning?: LlmBudgetWarning | null } | null)?.budget_warning;
    return w && typeof w.message === 'string' ? w.message : null;
}

export async function fetchLlmSpend(projectId: number, days = 30): Promise<LlmSpend> {
    const res = await api.get<LlmSpend>(`/projects/${projectId}/llm-spend`, { params: { days } });
    return res.data;
}
