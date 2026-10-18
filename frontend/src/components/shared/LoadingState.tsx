/**
 * LoadingState — the one way a panel says "fetching" (Epic U phase U7).
 *
 * Renders ``lines`` skeleton lines (the last one short) with a visually
 * hidden label for screen readers, as a ``role="status"`` region. Use it
 * in place of a bare "Loading…" string; pass ``card`` when the panel it
 * stands in for is a card so the placeholder keeps the page's rhythm.
 *
 *     {loading ? <LoadingState label="Loading runs" lines={3} /> : …}
 */

import Skeleton from './Skeleton';

interface LoadingStateProps {
    /** What is loading, for assistive tech and tests. */
    label: string;
    lines?: number;
    /** Lead with a title-sized bar. */
    title?: boolean;
    /** Wrap in a card frame. */
    card?: boolean;
    testId?: string;
}

export default function LoadingState({ label, lines = 3, title = false, card = false, testId }: LoadingStateProps) {
    const count = Math.max(1, lines);
    return (
        <div
            className={`loading-state${card ? ' loading-state--card' : ''}`}
            role="status"
            aria-live="polite"
            aria-busy="true"
            data-testid={testId ?? 'loading-state'}
        >
            <span className="sr-only">{label}</span>
            {title && <Skeleton variant="title" />}
            {Array.from({ length: count }, (_, i) => (
                <Skeleton key={i} variant={i === count - 1 && count > 1 ? 'line-short' : 'line'} />
            ))}
        </div>
    );
}
