/**
 * Preference-pair labeler (Story 1.3).
 *
 * Side-by-side ranking UI for DPO / ORPO data: the row carries a
 * prompt and two completions A / B; the reviewer picks one (or
 * declares a tie / both-bad) and optionally leaves a comment.
 *
 * Keyboard contract:
 *   ←   prefer A   → submit { chosen: 'A', tie: false, both_bad: false }
 *   →   prefer B   → submit { chosen: 'B', tie: false, both_bad: false }
 *   =   tie        → submit { chosen: null, tie: true,  both_bad: false }
 *   r   both bad   → submit { chosen: null, tie: false, both_bad: true  }
 *   esc skip       → onSkip()
 *
 * Shortcuts are suppressed while the comment textarea is focused so
 * the reviewer can use arrow keys to navigate inside the comment.
 */

import { useCallback, useEffect, useState } from 'react';

export type PreferenceChoice = 'A' | 'B' | null;

export interface PreferencePayload {
    chosen: PreferenceChoice;
    tie: boolean;
    both_bad: boolean;
    comment?: string;
}

interface PreferencePairLabelerProps {
    prompt: string;
    completionA: string;
    completionB: string;
    metadata?: Record<string, unknown>;
    onSubmit: (payload: PreferencePayload) => void;
    onSkip: () => void;
    disabled?: boolean;
}

function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
    return target.isContentEditable;
}

function attachComment(
    base: Omit<PreferencePayload, 'comment'>,
    comment: string,
): PreferencePayload {
    const trimmed = comment.trim();
    return trimmed ? { ...base, comment: trimmed } : { ...base };
}

const CARD_STYLE: React.CSSProperties = {
    flex: 1,
    minWidth: 0,
    padding: 'var(--space-md)',
    background: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: 'var(--radius-sm)',
    fontFamily: 'var(--font-mono, monospace)',
    fontSize: '0.9rem',
    whiteSpace: 'pre-wrap',
    overflow: 'auto',
    maxHeight: 360,
};

export default function PreferencePairLabeler({
    prompt,
    completionA,
    completionB,
    onSubmit,
    onSkip,
    disabled = false,
}: PreferencePairLabelerProps) {
    const [comment, setComment] = useState('');

    const choose = useCallback(
        (chosen: PreferenceChoice, tie: boolean, both_bad: boolean) => {
            if (disabled) return;
            onSubmit(
                attachComment({ chosen, tie, both_bad }, comment),
            );
        },
        [comment, disabled, onSubmit],
    );

    useEffect(() => {
        if (disabled) return undefined;

        const handler = (event: KeyboardEvent) => {
            if (isEditableTarget(event.target)) return;
            if (event.metaKey || event.ctrlKey || event.altKey) return;

            switch (event.key) {
                case 'ArrowLeft':
                    event.preventDefault();
                    choose('A', false, false);
                    return;
                case 'ArrowRight':
                    event.preventDefault();
                    choose('B', false, false);
                    return;
                case '=':
                    event.preventDefault();
                    choose(null, true, false);
                    return;
                case 'r':
                case 'R':
                    event.preventDefault();
                    choose(null, false, true);
                    return;
                case 'Escape':
                    event.preventDefault();
                    onSkip();
                    return;
                default:
                    return;
            }
        };

        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, [choose, disabled, onSkip]);

    return (
        <div className="preference-pair-labeler" data-testid="preference-pair-labeler">
            <div
                data-testid="pref-prompt"
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--bg-tertiary, #efefef)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.95rem',
                    whiteSpace: 'pre-wrap',
                    marginBottom: 'var(--space-md)',
                }}
            >
                <div
                    style={{
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: 0.5,
                        color: 'var(--text-secondary)',
                        marginBottom: 6,
                    }}
                >
                    Prompt
                </div>
                {prompt || '(empty prompt)'}
            </div>

            <div
                style={{
                    display: 'flex',
                    gap: 'var(--space-md)',
                    flexWrap: 'wrap',
                }}
            >
                <div data-testid="pref-completion-a" style={CARD_STYLE}>
                    <div
                        style={{
                            fontSize: '0.75rem',
                            textTransform: 'uppercase',
                            letterSpacing: 0.5,
                            color: 'var(--text-secondary)',
                            marginBottom: 6,
                        }}
                    >
                        Completion A ←
                    </div>
                    {completionA || '(empty)'}
                </div>
                <div data-testid="pref-completion-b" style={CARD_STYLE}>
                    <div
                        style={{
                            fontSize: '0.75rem',
                            textTransform: 'uppercase',
                            letterSpacing: 0.5,
                            color: 'var(--text-secondary)',
                            marginBottom: 6,
                        }}
                    >
                        Completion B →
                    </div>
                    {completionB || '(empty)'}
                </div>
            </div>

            <div style={{ marginTop: 'var(--space-md)' }}>
                <label
                    htmlFor="pref-comment-input"
                    style={{
                        display: 'block',
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: 0.5,
                        color: 'var(--text-secondary)',
                        marginBottom: 6,
                    }}
                >
                    Comment (optional — included in label_payload)
                </label>
                <textarea
                    id="pref-comment-input"
                    className="form-input"
                    rows={2}
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    disabled={disabled}
                    placeholder="Why A or B?"
                    data-testid="pref-comment"
                    style={{ width: '100%', resize: 'vertical' }}
                />
            </div>

            <div
                style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 'var(--space-sm)',
                    marginTop: 'var(--space-md)',
                    alignItems: 'center',
                }}
            >
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => choose('A', false, false)}
                    disabled={disabled}
                    data-testid="pref-a"
                    title="Prefer A (←)"
                >
                    ← Prefer A
                </button>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => choose('B', false, false)}
                    disabled={disabled}
                    data-testid="pref-b"
                    title="Prefer B (→)"
                >
                    Prefer B →
                </button>
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => choose(null, true, false)}
                    disabled={disabled}
                    data-testid="pref-tie"
                    title="Tie (=)"
                >
                    Tie =
                </button>
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => choose(null, false, true)}
                    disabled={disabled}
                    data-testid="pref-bothbad"
                    title="Both bad (r)"
                >
                    Both bad (r)
                </button>
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onSkip}
                    disabled={disabled}
                    data-testid="pref-skip"
                    title="Skip (esc)"
                    style={{ marginLeft: 'auto' }}
                >
                    Skip
                </button>
            </div>
        </div>
    );
}
