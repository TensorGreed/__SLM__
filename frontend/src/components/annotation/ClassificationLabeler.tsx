/**
 * Keyboard-driven classification labeler.
 *
 * UX contract (Story 1.2):
 * - Renders the row's text + a button strip of allowed labels.
 * - Number keys 1–9 immediately submit the indexed label and advance.
 * - Clicking a label button does the same.
 * - 'esc' skips the row (parent unassigns + advances).
 * - When ``disabled`` is true (mid-flight save) all input is ignored
 *   so a fast typist doesn't double-submit.
 *
 * The component is presentational: parent owns the row + the API
 * call to submit / skip. We branch on parent-provided callbacks
 * rather than touching the API directly.
 */

import { useEffect } from 'react';

interface ClassificationLabelerProps {
    text: string;
    labels: string[];
    onSubmit: (label: string) => void;
    onSkip: () => void;
    disabled?: boolean;
}

function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
    return target.isContentEditable;
}

export default function ClassificationLabeler({
    text,
    labels,
    onSubmit,
    onSkip,
    disabled = false,
}: ClassificationLabelerProps) {
    useEffect(() => {
        if (disabled) return undefined;

        const handler = (event: KeyboardEvent) => {
            if (isEditableTarget(event.target)) return;
            if (event.metaKey || event.ctrlKey || event.altKey) return;

            const key = event.key;
            if (key === 'Escape') {
                event.preventDefault();
                onSkip();
                return;
            }
            if (key >= '1' && key <= '9') {
                const idx = Number.parseInt(key, 10) - 1;
                if (idx < labels.length) {
                    event.preventDefault();
                    onSubmit(labels[idx]);
                }
            }
        };

        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, [labels, onSubmit, onSkip, disabled]);

    return (
        <div className="classification-labeler" data-testid="classification-labeler">
            <div
                className="classification-labeler-text"
                data-testid="classification-labeler-text"
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: '0.95rem',
                    whiteSpace: 'pre-wrap',
                    overflow: 'auto',
                    maxHeight: 400,
                    margin: 0,
                }}
            >
                {text || '(empty row)'}
            </div>

            <div
                style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 'var(--space-sm)',
                    marginTop: 'var(--space-md)',
                }}
            >
                {labels.map((label, idx) => (
                    <button
                        key={label}
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => onSubmit(label)}
                        disabled={disabled}
                        data-testid={`classification-label-${label}`}
                        title={
                            idx < 9
                                ? `Submit ${label} (press ${idx + 1})`
                                : `Submit ${label}`
                        }
                    >
                        {idx < 9 && (
                            <span
                                style={{
                                    display: 'inline-block',
                                    marginRight: 6,
                                    minWidth: 16,
                                    textAlign: 'center',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: 4,
                                    padding: '0 4px',
                                    fontSize: '0.75rem',
                                    fontFamily: 'var(--font-mono, monospace)',
                                }}
                            >
                                {idx + 1}
                            </span>
                        )}
                        {label}
                    </button>
                ))}
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onSkip}
                    disabled={disabled}
                    data-testid="classification-skip"
                    title="Skip this row (esc)"
                    style={{ marginLeft: 'auto' }}
                >
                    <span
                        style={{
                            display: 'inline-block',
                            marginRight: 6,
                            border: '1px solid var(--border-color)',
                            borderRadius: 4,
                            padding: '0 4px',
                            fontSize: '0.75rem',
                            fontFamily: 'var(--font-mono, monospace)',
                        }}
                    >
                        esc
                    </span>
                    Skip
                </button>
            </div>
        </div>
    );
}
