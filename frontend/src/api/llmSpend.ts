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
    /** "subscription" rows carry no $ (the CLI plan bills them). */
    billing?: 'metered' | 'subscription';
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
        subscription_calls?: number;
        subscription_tokens?: number;
    };
    by_model: LlmSpendModelRow[];
    by_kind: LlmSpendKindRow[];
    /** Ledger operations whose tokens were estimated (a stream with no usage event). */
    estimated_operations?: number;
    /** Calendar month to date (UTC), whatever the window. */
    month?: { since: string; operations: number; total_tokens: number; cost_usd: number };
    budget?: LlmBudget | null;
    budget_status?: LlmBudgetStatus | null;
    /** Account-wide month-to-date, when the backend reports it. */
    global?: LlmGlobalBudget;
    basis: string;
    note: string;
}

/** runtime_config["llm_budget_usd"] — a hard stop for metered API calls (warn_at stays advisory). */
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
    /** Account-wide cap (LLM_MONTHLY_BUDGET_USD), all projects. */
    global?: LlmGlobalBudget;
    hard_stop?: boolean;
}

export interface LlmGlobalBudget {
    monthly: number | null;
    spent_usd: number;
    status: 'ok' | 'warning' | 'over';
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

/** The message of a 402 ``llm_budget_exceeded`` start error (the hard API
 *  budget), or null for any other error — so a start that does not show the
 *  error envelope inline can toast it. */
export function budgetExceededMessage(err: unknown): string | null {
    const response = (err as { response?: { status?: number; data?: { error_code?: unknown; message?: unknown; detail?: unknown } } })?.response;
    const data = response?.data;
    if (data?.error_code !== 'llm_budget_exceeded') return null;
    const text = typeof data.message === 'string' && data.message ? data.message : typeof data.detail === 'string' ? data.detail : '';
    return text || 'The monthly API budget is reached — switch to a local or subscription backend.';
}

export async function fetchLlmSpend(projectId: number, days = 30): Promise<LlmSpend> {
    const res = await api.get<LlmSpend>(`/projects/${projectId}/llm-spend`, { params: { days } });
    return res.data;
}

export async function fetchLlmBudget(projectId: number): Promise<LlmBudgetPayload> {
    const res = await api.get<LlmBudgetPayload>(`/projects/${projectId}/llm-budget`);
    return res.data;
}
