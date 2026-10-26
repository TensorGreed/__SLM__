/**
 * SynthRejectSummary — what a playbook run turned away, grouped by reason,
 * with expandable examples. Rendered from the Job result (or sync result).
 */

import { useMemo } from 'react';

import type { RejectFields, RejectedExample } from '../../api/synthPlaybook';
import './SynthRejectSummary.css';

export const REJECT_REASON_LABEL: Record<string, string> = {
    parse_error: 'Not valid JSON',
    empty_code: 'Empty code',
    unchanged_code: 'Code left unchanged',
    label_mismatch: 'Finding label does not match the requested label',
    line_out_of_range: 'Finding lines outside the code',
    evidence_not_in_lines: 'Evidence not found in the cited lines',
    evidence_too_short: 'Evidence too short to be an excerpt',
    lines_not_changed: 'Cited lines are unchanged from the original',
    code_too_long: 'Code too long',
    has_findings: 'Hard negative still has findings',
    unknown_label: 'Label not in the taxonomy',
};

function excerpt(payload: RejectedExample['payload']): string {
    if (!payload) return '';
    const code = (payload as Record<string, unknown>).code;
    const text = typeof code === 'string' ? code : JSON.stringify(payload);
    return text.length > 600 ? `${text.slice(0, 600)}…` : text;
}

export default function SynthRejectSummary({ result }: { result: RejectFields | null | undefined }) {
    const summary = result?.reject_summary ?? {};
    const reasons = Object.entries(summary).sort((a, b) => b[1] - a[1]);
    const total = result?.rejected_count ?? reasons.reduce((n, [, c]) => n + c, 0);
    const byReason = useMemo(() => {
        const out: Record<string, RejectedExample[]> = {};
        for (const ex of result?.rejected_examples ?? []) (out[ex.reason] ??= []).push(ex);
        return out;
    }, [result]);
    if (!total) return null;
    return (
        <div className="synth-rejects" data-testid="synth-reject-summary">
            <p className="synth-rejects__headline">
                <strong>{total}</strong> candidate{total === 1 ? '' : 's'} rejected and not saved
            </p>
            <ul className="synth-rejects__list">
                {reasons.map(([reason, count]) => (
                    <li key={reason} data-testid={`synth-reject-${reason}`}>
                        <details>
                            <summary>
                                {REJECT_REASON_LABEL[reason] ?? reason} <code>{reason}</code>
                                <span className="synth-rejects__count">{count}</span>
                            </summary>
                            {(byReason[reason] ?? []).length === 0 ? (
                                <p className="u-hint">No example kept for this reason.</p>
                            ) : (
                                <ul className="synth-rejects__examples">
                                    {byReason[reason].map((ex, i) => (
                                        <li key={i}>
                                            {ex.detail && <p className="synth-rejects__detail">{ex.detail}</p>}
                                            {excerpt(ex.payload) && <pre className="synth-rejects__code">{excerpt(ex.payload)}</pre>}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </details>
                    </li>
                ))}
            </ul>
        </div>
    );
}
