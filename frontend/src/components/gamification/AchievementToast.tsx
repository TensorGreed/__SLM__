/**
 * Terminal-glow achievement / level-up toast for the Lab Journal.
 *
 * Rendered by the shared <ToastContainer> when ``type === 'achievement'``
 * (the store's ``addAchievementToast`` helper sets this). Standard
 * toasts (success/error/info/warning) still go through the default
 * pill renderer — this variant is opt-in via the payload shape.
 *
 * The frame is plain CSS box-shadow + the .terminal-surface utility
 * defined in index.css. No external assets; the glyphs are unicode
 * box-drawing + arrows so the bundle stays small.
 */

import { useState } from 'react';
import type {
    AchievementToastPayload,
    ToastMessage,
} from '../../stores/toastStore';
import { useToastStore } from '../../stores/toastStore';

interface AchievementToastProps {
    toast: ToastMessage;
}

export default function AchievementToast({ toast }: AchievementToastProps) {
    const removeToast = useToastStore((s) => s.removeToast);
    const [removing, setRemoving] = useState(false);

    const handleClose = () => {
        setRemoving(true);
        setTimeout(() => removeToast(toast.id), 200);
    };

    const payload = toast.payload as AchievementToastPayload | undefined;
    if (!payload) {
        // Defensive: the store guards this, but render a stub if a
        // caller misuses the variant.
        return (
            <div
                className={`toast info ${removing ? 'removing' : ''}`}
                role="alert"
                data-testid={`toast-${toast.id}`}
            >
                <span className="toast-message">{toast.message}</span>
                <button
                    type="button"
                    className="toast-close"
                    onClick={handleClose}
                    aria-label="Close"
                >
                    ×
                </button>
            </div>
        );
    }

    const isLevelUp = payload.kind === 'level_up';
    const heading = isLevelUp ? 'LEVEL UP' : 'ACHIEVEMENT UNLOCKED';
    const glyph = isLevelUp ? '▲' : '▣';

    return (
        <div
            className={`toast terminal-surface ${removing ? 'removing' : ''}`}
            role="alert"
            data-testid={`toast-${toast.id}`}
            data-toast-type="achievement"
            style={{
                padding: 'var(--space-md)',
                minWidth: 260,
                maxWidth: 360,
                fontFamily: 'var(--font-mono)',
            }}
        >
            <div
                className="terminal-glow-bright"
                style={{
                    fontSize: '0.65rem',
                    letterSpacing: '0.18em',
                    textTransform: 'uppercase',
                    marginBottom: 4,
                }}
            >
                ┌─ {heading} ─
            </div>
            <div
                className="terminal-glow-bright"
                style={{
                    fontSize: '0.95rem',
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'baseline',
                    gap: 6,
                }}
            >
                <span aria-hidden="true">{glyph}</span>
                <span>
                    {isLevelUp
                        ? `Level ${payload.level_after ?? '?'} — ${payload.title}`
                        : payload.title}
                </span>
            </div>
            {payload.description && (
                <div
                    className="terminal-glow"
                    style={{
                        fontSize: '0.8rem',
                        opacity: 0.85,
                        marginTop: 2,
                        marginBottom: 4,
                    }}
                >
                    {payload.description}
                </div>
            )}
            {payload.xp_awarded ? (
                <div
                    className="terminal-glow"
                    style={{ fontSize: '0.8rem', opacity: 0.9 }}
                >
                    +{payload.xp_awarded} XP
                </div>
            ) : null}
            {payload.level_after && !isLevelUp ? (
                <div
                    className="terminal-glow"
                    style={{
                        fontSize: '0.75rem',
                        opacity: 0.7,
                        marginTop: 2,
                    }}
                >
                    LV {payload.level_after} reached
                </div>
            ) : null}
            <div
                className="terminal-glow"
                style={{
                    fontSize: '0.65rem',
                    letterSpacing: '0.18em',
                    marginTop: 4,
                    opacity: 0.55,
                }}
            >
                └─
            </div>
            <button
                type="button"
                className="toast-close"
                onClick={handleClose}
                aria-label="Close"
                data-testid={`toast-close-${toast.id}`}
            >
                ×
            </button>
        </div>
    );
}
