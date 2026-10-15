import { useEffect, useState } from 'react';

import { fetchLlmSpend, type LlmSpend } from '../../api/llmSpend';
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
};

/** Cloud LLM spend over the last N days: tokens, list-price estimate, by
 *  model and by operation. Hidden-quiet when nothing was spent. */
export default function LlmSpendCard({ projectId, days = 30 }: { projectId: number; days?: number }) {
    const [spend, setSpend] = useState<LlmSpend | null>(null);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setSpend(null);
        setFailed(false);
        fetchLlmSpend(projectId, days)
            .then((data) => { if (!cancelled) setSpend(data); })
            .catch(() => { if (!cancelled) setFailed(true); });
        return () => { cancelled = true; };
    }, [projectId, days]);

    if (failed) return null;
    const totals = spend?.totals;
    return (
        <div className="card llm-spend-card" data-testid="llm-spend-card">
            <h3 className="llm-spend-card__title">Cloud LLM spend (last {days} days)</h3>
            {!spend && <p className="llm-spend-card__muted">Loading…</p>}
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
                                    <td>{row.priced ? formatUsd(row.cost_usd) : 'price unknown'}</td>
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
