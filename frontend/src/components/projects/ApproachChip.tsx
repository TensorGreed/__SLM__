/**
 * Decision-engine chip (Theme 7).
 *
 * Renders the recommended approach (prompt_only / rag_first / sft /
 * dpo / distillation) as a one-line callout under the brief textarea
 * in the create modal. Tone is green when the recommendation is
 * `sft` (the platform's default), warning-amber when the engine is
 * steering the user toward a different approach.
 *
 * When the recommendation isn't `sft`, an inline "or stay with SFT
 * anyway →" override link lets the user acknowledge the warning and
 * proceed. Dismissing keeps the chip visible but in a muted
 * acknowledged state so the user doesn't lose context.
 */

import type { ApproachRecommendation } from '../../api/blueprintAnalyze';

interface ApproachChipProps {
    /** Recommendation payload from the backend. `null` while
     * the debounced analyze call hasn't returned yet. */
    recommendation: ApproachRecommendation | null;
    /** True while the debounced analyze request is in flight. */
    loading: boolean;
    /** True if the user already clicked "stay with SFT anyway"
     * for the current brief. Surfaces an acknowledged state. */
    dismissed: boolean;
    /** Click handler for the override link. */
    onDismiss: () => void;
}

export default function ApproachChip({
    recommendation,
    loading,
    dismissed,
    onDismiss,
}: ApproachChipProps) {
    if (loading && !recommendation) {
        return (
            <div
                data-testid="approach-chip-loading"
                role="status"
                style={{
                    padding: 'var(--space-sm) var(--space-md)',
                    marginTop: 'var(--space-sm)',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-subtle)',
                    color: 'var(--text-secondary)',
                    fontSize: '0.85rem',
                }}
            >
                Analyzing your brief…
            </div>
        );
    }

    if (!recommendation) {
        return null;
    }

    const isSft = recommendation.approach === 'sft';

    if (dismissed && !isSft) {
        return (
            <div
                data-testid="approach-chip-dismissed"
                style={{
                    padding: 'var(--space-xs) var(--space-sm)',
                    marginTop: 'var(--space-sm)',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-subtle)',
                    color: 'var(--text-secondary)',
                    fontSize: '0.8rem',
                }}
            >
                ▣ Acknowledged — proceeding with SFT despite the{' '}
                <code>{recommendation.approach}</code> recommendation.
            </div>
        );
    }

    const bg = isSft ? 'var(--color-success-bg)' : 'var(--color-warning-bg)';
    const fg = isSft ? 'var(--color-success)' : 'var(--color-warning)';
    const border = isSft ? 'var(--color-success)' : 'var(--color-warning)';
    const icon = isSft ? '✓' : '💡';

    return (
        <div
            data-testid="approach-chip"
            data-approach={recommendation.approach}
            role="status"
            style={{
                marginTop: 'var(--space-sm)',
                padding: 'var(--space-sm) var(--space-md)',
                borderRadius: 'var(--radius-md)',
                border: `1px solid ${border}`,
                background: bg,
                color: fg,
                fontSize: '0.88rem',
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
            }}
        >
            <div
                style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 'var(--space-sm)',
                }}
            >
                <span aria-hidden="true">{icon}</span>
                <strong style={{ flex: 1 }} data-testid="approach-chip-headline">
                    {recommendation.headline}
                </strong>
                <span
                    title={recommendation.signals.join(', ') || 'no signals'}
                    aria-label="What signals were detected"
                    style={{
                        fontSize: '0.75rem',
                        cursor: 'help',
                        opacity: 0.7,
                    }}
                    data-testid="approach-chip-signals"
                >
                    {Math.round(recommendation.confidence * 100)}% · ?
                </span>
            </div>
            <div
                style={{ color: fg, opacity: 0.85, fontSize: '0.82rem' }}
                data-testid="approach-chip-rationale"
            >
                {recommendation.rationale}
            </div>
            {!isSft && (
                <div style={{ marginTop: 4 }}>
                    <button
                        type="button"
                        onClick={onDismiss}
                        data-testid="approach-chip-override"
                        style={{
                            background: 'none',
                            border: 'none',
                            padding: 0,
                            color: fg,
                            textDecoration: 'underline',
                            cursor: 'pointer',
                            fontSize: '0.82rem',
                        }}
                    >
                        or stay with SFT anyway →
                    </button>
                </div>
            )}
        </div>
    );
}
