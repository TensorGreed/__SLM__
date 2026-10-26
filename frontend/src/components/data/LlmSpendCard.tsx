import { useEffect, useState } from 'react';

import { fetchLlmBudget, fetchLlmSpend, saveLlmBudget, type LlmGlobalBudget, type LlmSpend } from '../../api/llmSpend';
import './LlmSpendCard.css';

/** "12k" / "940" / "1.2M" token counts. */
export function formatTokens(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1000) return `${Math.round(n / 1000)}k`;
    return String(n);
}

/** "~$0.42" — or "under $0.01" for a non-zero spend that rounds to nothing. */
export function formatUsd(usd: number): string {
    if (usd > 0 && usd < 0.01) return 'under $0.01';
    return `~$${usd.toFixed(2)}`;
}

const KIND_LABELS: Record<string, string> = {
    span_relabel: 'Relabels / answer keys',
    synth_playbook: 'Playbook synthesis',
    synth_legacy_qa: 'Q&A synthesis',
    synth_legacy_spans: 'Span synthesis',
    synth_legacy_conversations: 'Conversation synthesis',
    seed_documents: 'Sample documents',
    documents_qa_flow: 'Documents → Q&A',
    post_training_lift_eval: 'Lift checks (judge)',
    auto_rag_comparison: 'Retrieval comparisons (judge)',
    gold_llm_generate: 'LLM answer key',
    plan_refinement: 'AI plan strategy',
    probe_judge: 'Built-in checks judge',
    distillation_capture: 'Distillation teacher',
    playground_chat: 'Try it chats (cloud)',
};

/** "This month: ~$4.10 of $5.00 (82%)" — the month-to-date budget line. */
export function budgetLineText(status: { spent: number; budget: number; share: number }): string {
    return `This month: ${formatUsd(status.spent)} of $${status.budget.toFixed(2)} (${Math.round(status.share * 100)}%)`;
}

/** Month-to-date budget line + a small editor (monthly $ + warn-at %). */
function BudgetSection({ projectId, spend, global, onSaved }: { projectId: number; spend: LlmSpend; global: LlmGlobalBudget | null; onSaved: () => void }) {
    const budget = spend.budget ?? null;
    const status = spend.budget_status ?? null;
    const [editing, setEditing] = useState(false);
    const [monthly, setMonthly] = useState(budget ? String(budget.monthly) : '');
    const [warnPct, setWarnPct] = useState(budget ? String(Math.round(budget.warn_at * 100)) : '80');
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    const save = async (clear: boolean) => {
        setError(null);
        let amount: number | null = null;
        let warnAt: number | undefined;
        if (!clear) {
            amount = Number(monthly);
            const pct = Number(warnPct);
            if (!Number.isFinite(amount) || amount <= 0) {
                setError('Enter a monthly amount above $0.');
                return;
            }
            if (!Number.isFinite(pct) || pct <= 0 || pct > 100) {
                setError('Warn at must be between 1 and 100%.');
                return;
            }
            warnAt = pct / 100;
        }
        setSaving(true);
        try {
            await saveLlmBudget(projectId, amount, warnAt);
            setEditing(false);
            onSaved();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Could not save the budget.');
        } finally {
            setSaving(false);
        }
    };

    const monthCost = spend.month?.cost_usd ?? 0;
    return (
        <div className="llm-spend-card__budget" data-testid="llm-spend-budget">
            {status ? (
                <p
                    className={`llm-spend-card__budget-line llm-spend-card__budget-line--${status.status}`}
                    data-testid="llm-spend-budget-line"
                >
                    <strong>{budgetLineText(status)}</strong>
                    {status.status === 'over' && ' — over budget. Cloud API calls are stopped; local models (Ollama) and a subscription CLI still run.'}
                    {status.status === 'warning' && ` — past the ${Math.round(status.warn_at * 100)}% warning mark.`}
                </p>
            ) : (
                <p className="llm-spend-card__muted" data-testid="llm-spend-budget-line">
                    This month: {formatUsd(monthCost)} · no monthly budget set.
                </p>
            )}
            {global && global.monthly != null && (
                <p
                    className={`llm-spend-card__budget-line llm-spend-card__budget-line--${global.status}`}
                    data-testid="llm-spend-global-line"
                >
                    All projects this month: {formatUsd(global.spent_usd)} of ${global.monthly.toFixed(2)} (API credits)
                    {global.status === 'over' && ' — cap reached, cloud API calls are stopped.'}
                </p>
            )}
            {!editing && (
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    data-testid="llm-spend-budget-edit"
                    onClick={() => setEditing(true)}
                >
                    {budget ? 'Edit budget' : 'Set a monthly budget'}
                </button>
            )}
            {editing && (
                <div className="llm-spend-card__budget-form" data-testid="llm-spend-budget-form">
                    <label>
                        Monthly budget ($)
                        <input
                            type="number"
                            min="0"
                            step="0.5"
                            value={monthly}
                            data-testid="llm-spend-budget-monthly"
                            onChange={(e) => setMonthly(e.target.value)}
                        />
                    </label>
                    <label>
                        Warn at (%)
                        <input
                            type="number"
                            min="1"
                            max="100"
                            value={warnPct}
                            data-testid="llm-spend-budget-warn"
                            onChange={(e) => setWarnPct(e.target.value)}
                        />
                    </label>
                    <button type="button" className="btn btn-primary btn-sm" disabled={saving} data-testid="llm-spend-budget-save" onClick={() => void save(false)}>
                        Save
                    </button>
                    {budget && (
                        <button type="button" className="btn btn-secondary btn-sm" disabled={saving} data-testid="llm-spend-budget-clear" onClick={() => void save(true)}>
                            Clear budget
                        </button>
                    )}
                    <button type="button" className="btn btn-secondary btn-sm" disabled={saving} onClick={() => { setEditing(false); setError(null); }}>
                        Cancel
                    </button>
                    <p className="llm-spend-card__muted">
                        Cloud API calls stop at the cap (one call may overshoot by its own cost). The warning mark is advisory. Local models and a subscription CLI are never blocked.
                    </p>
                </div>
            )}
            {error && <p className="llm-spend-card__budget-error" data-testid="llm-spend-budget-error">{error}</p>}
        </div>
    );
}

/** Cloud LLM spend over the last N days: tokens, list-price estimate, by
 *  model and by operation. Hidden-quiet when nothing was spent. */
export default function LlmSpendCard({ projectId, days = 30 }: { projectId: number; days?: number }) {
    const [spend, setSpend] = useState<LlmSpend | null>(null);
    const [failed, setFailed] = useState(false);
    const [reloadKey, setReloadKey] = useState(0);
    const [globalBudget, setGlobalBudget] = useState<LlmGlobalBudget | null>(null);

    useEffect(() => {
        let cancelled = false;
        setSpend(null);
        setFailed(false);
        fetchLlmSpend(projectId, days)
            .then((data) => { if (!cancelled) setSpend(data); })
            .catch(() => { if (!cancelled) setFailed(true); });
        setGlobalBudget(null);
        fetchLlmBudget(projectId)
            .then((data) => { if (!cancelled) setGlobalBudget(data?.global ?? null); })
            .catch(() => undefined);
        return () => { cancelled = true; };
    }, [projectId, days, reloadKey]);

    if (failed) return null;
    const totals = spend?.totals;
    return (
        <div className="card llm-spend-card" data-testid="llm-spend-card">
            <h3 className="llm-spend-card__title">Cloud LLM spend (last {days} days)</h3>
            {!spend && <p className="llm-spend-card__muted">Loading…</p>}
            {spend && <BudgetSection projectId={projectId} spend={spend} global={spend.global ?? globalBudget} onSaved={() => setReloadKey((k) => k + 1)} />}
            {spend && totals && totals.operations === 0 && (
                <p className="llm-spend-card__muted" data-testid="llm-spend-empty">
                    No cloud-model calls in this window. Local models (Ollama) cost nothing and are not counted.
                </p>
            )}
            {spend && totals && totals.operations > 0 && (
                <>
                    <p className="llm-spend-card__headline" data-testid="llm-spend-headline">
                        <strong>{formatTokens(totals.total_tokens)} tokens</strong>
                        {' · '}
                        <strong>{formatUsd(totals.cost_usd)}</strong>
                        <span className="llm-spend-card__muted"> list-price estimate</span>
                        {' · '}
                        {totals.calls} calls in {totals.operations} operations
                        {totals.cache_read_tokens > 0 && ` · ${formatTokens(totals.cache_read_tokens)} read from the prompt cache`}
                    </p>
                    {totals.unpriced_tokens > 0 && (
                        <p className="llm-spend-card__muted">
                            {formatTokens(totals.unpriced_tokens)} tokens are on models with no known price and carry no $ figure.
                        </p>
                    )}
                    {(totals.subscription_calls ?? 0) > 0 && (
                        <p className="llm-spend-card__muted" data-testid="llm-spend-subscription">
                            <strong>Subscription CLI:</strong> {totals.subscription_calls} calls
                            {(totals.subscription_tokens ?? 0) > 0 && `, ${formatTokens(totals.subscription_tokens ?? 0)} tokens`}
                            {' '}— not counted against the API budget.
                        </p>
                    )}
                    <table className="llm-spend-card__table" data-testid="llm-spend-by-model">
                        <thead>
                            <tr><th>Model</th><th>Calls</th><th>Tokens</th><th>~$</th></tr>
                        </thead>
                        <tbody>
                            {spend.by_model.map((row) => (
                                <tr key={row.model}>
                                    <td>{row.model}</td>
                                    <td>{row.calls}</td>
                                    <td>{formatTokens(row.total_tokens)}</td>
                                    <td>{row.billing === 'subscription' ? 'subscription' : row.priced ? formatUsd(row.cost_usd) : 'price unknown'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="llm-spend-card__muted" data-testid="llm-spend-by-kind">
                        By operation: {spend.by_kind.map((k) => `${KIND_LABELS[k.kind] ?? k.kind} ${formatUsd(k.cost_usd)} (${formatTokens(k.total_tokens)} tokens)`).join(' · ')}
                    </p>
                    <p className="llm-spend-card__muted">{spend.note}</p>
                </>
            )}
        </div>
    );
}
