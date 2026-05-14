/**
 * Right-side overlay drawer that surfaces the full Lab Journal.
 *
 * Three sections:
 *   1. Status — ASCII-bordered card with level, XP, next-threshold,
 *      and a phosphor progress bar.
 *   2. Unlocked — most recent first. Each row carries the
 *      title, tier, XP, and ISO unlock timestamp (the API returns
 *      one per achievement).
 *   3. Locked — greyed list. Hidden discovery achievements show as
 *      ``▢ ??? — Discovery`` until unlocked, then they reveal in the
 *      Unlocked list.
 *
 * The drawer mounts on demand from ProgressChip and fetches the
 * achievement catalog from /achievements once per open. The
 * progression state in the header comes from the shared poll cache.
 */

import { useEffect, useState } from 'react';

import {
    fetchAchievements,
    type AchievementListItem,
} from '../../api/gamification';
import { useProgressionState } from './useProgressionPoll';

interface LabJournalDrawerProps {
    projectId: number;
    onClose: () => void;
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const detail = (err as { response?: { data?: { detail?: unknown } } })
            .response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) return detail;
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    return 'Failed to load Lab Journal.';
}

function bar(filled: number, total: number, width = 24): string {
    if (total <= 0) return '░'.repeat(width);
    const ratio = Math.max(0, Math.min(1, filled / total));
    const fill = Math.round(ratio * width);
    return '█'.repeat(fill) + '░'.repeat(width - fill);
}

export default function LabJournalDrawer({
    projectId,
    onClose,
}: LabJournalDrawerProps) {
    const progression = useProgressionState();
    const [items, setItems] = useState<AchievementListItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        fetchAchievements(projectId)
            .then((res) => {
                if (cancelled) return;
                setItems(res.achievements);
                setError('');
            })
            .catch((err) => {
                if (cancelled) return;
                setError(extractErrorMessage(err));
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    const level = progression?.level ?? 1;
    const levelTitle = progression?.level_title ?? 'Intern';
    const xp = progression?.xp_balance ?? 0;
    const into = progression?.xp_into_level ?? 0;
    const toNext = progression?.xp_to_next_level ?? 100;
    const progressBar = bar(into, toNext);

    const unlocked = items.filter((item) => item.unlocked);
    // Sort unlocked by unlock time (most recent first).
    unlocked.sort((a, b) => {
        if (!a.unlocked_at) return 1;
        if (!b.unlocked_at) return -1;
        return b.unlocked_at.localeCompare(a.unlocked_at);
    });
    const locked = items.filter((item) => !item.unlocked);

    return (
        <div
            role="dialog"
            aria-label="Lab Journal"
            data-testid="lab-journal-drawer"
            style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(10, 15, 12, 0.55)',
                zIndex: 1100,
                display: 'flex',
                justifyContent: 'flex-end',
            }}
            onClick={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <aside
                className="terminal-surface"
                style={{
                    width: 440,
                    maxWidth: '100vw',
                    height: '100%',
                    overflowY: 'auto',
                    padding: 'var(--space-lg)',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--crt-green)',
                }}
            >
                <header
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 'var(--space-md)',
                    }}
                >
                    <h2
                        className="terminal-glow-bright"
                        style={{
                            margin: 0,
                            fontSize: '1rem',
                            letterSpacing: '0.16em',
                            textTransform: 'uppercase',
                        }}
                    >
                        ▣ Lab Journal
                    </h2>
                    <button
                        type="button"
                        className="terminal-glow"
                        onClick={onClose}
                        aria-label="Close Lab Journal"
                        data-testid="close-lab-journal"
                        style={{
                            background: 'transparent',
                            border: '1px solid var(--crt-border)',
                            color: 'var(--crt-green)',
                            cursor: 'pointer',
                            width: 28,
                            height: 28,
                            borderRadius: 'var(--radius-sm)',
                        }}
                    >
                        ×
                    </button>
                </header>

                {/* Status card */}
                <section
                    style={{
                        marginBottom: 'var(--space-lg)',
                        padding: 'var(--space-md)',
                        border: '1px solid var(--crt-border)',
                        borderRadius: 'var(--radius-sm)',
                    }}
                >
                    <div
                        className="terminal-glow-bright"
                        style={{ fontSize: '0.95rem', fontWeight: 600 }}
                    >
                        L{level} — {levelTitle}
                    </div>
                    <div
                        className="terminal-glow"
                        style={{ fontSize: '0.8rem', opacity: 0.85, marginTop: 4 }}
                    >
                        {xp.toLocaleString()} XP total
                    </div>
                    <div
                        className="terminal-glow"
                        style={{
                            fontSize: '0.8rem',
                            marginTop: 8,
                            display: 'flex',
                            gap: 8,
                            alignItems: 'baseline',
                        }}
                        data-testid="lab-journal-xp-bar"
                    >
                        <code>[{progressBar}]</code>
                        <span style={{ opacity: 0.65 }}>
                            {into}/{toNext} to L{level + 1}
                        </span>
                    </div>
                </section>

                {error && (
                    <div
                        className="error-banner"
                        data-testid="lab-journal-error"
                        style={{ marginBottom: 'var(--space-md)' }}
                    >
                        {error}
                    </div>
                )}
                {loading && !error && (
                    <div
                        className="terminal-glow"
                        style={{ opacity: 0.6, fontSize: '0.85rem' }}
                    >
                        Loading achievements…
                    </div>
                )}

                {/* Unlocked */}
                <section style={{ marginBottom: 'var(--space-lg)' }}>
                    <h3
                        className="terminal-glow-bright"
                        style={{
                            margin: 0,
                            fontSize: '0.75rem',
                            letterSpacing: '0.18em',
                            textTransform: 'uppercase',
                            marginBottom: 'var(--space-sm)',
                        }}
                    >
                        Unlocked ({unlocked.length})
                    </h3>
                    {unlocked.length === 0 && !loading ? (
                        <div
                            className="terminal-glow"
                            style={{ opacity: 0.5, fontSize: '0.85rem' }}
                        >
                            Nothing yet. Import a dataset or kick off a training
                            run to see your first stamp.
                        </div>
                    ) : (
                        <ul
                            style={{
                                listStyle: 'none',
                                padding: 0,
                                margin: 0,
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 6,
                            }}
                        >
                            {unlocked.map((item) => (
                                <li
                                    key={item.id}
                                    data-testid={`unlocked-${item.id}`}
                                    style={{
                                        borderBottom: '1px dashed var(--crt-border)',
                                        paddingBottom: 4,
                                    }}
                                >
                                    <div
                                        className="terminal-glow-bright"
                                        style={{ fontSize: '0.85rem' }}
                                    >
                                        ▣ {item.title}{' '}
                                        <span
                                            style={{
                                                opacity: 0.55,
                                                fontSize: '0.7rem',
                                            }}
                                        >
                                            ({item.tier})
                                        </span>
                                    </div>
                                    <div
                                        className="terminal-glow"
                                        style={{ fontSize: '0.75rem', opacity: 0.75 }}
                                    >
                                        {item.description} · +{item.xp} XP
                                    </div>
                                </li>
                            ))}
                        </ul>
                    )}
                </section>

                {/* Locked */}
                <section>
                    <h3
                        className="terminal-glow"
                        style={{
                            margin: 0,
                            fontSize: '0.75rem',
                            letterSpacing: '0.18em',
                            textTransform: 'uppercase',
                            marginBottom: 'var(--space-sm)',
                            opacity: 0.7,
                        }}
                    >
                        Locked ({locked.length})
                    </h3>
                    <ul
                        style={{
                            listStyle: 'none',
                            padding: 0,
                            margin: 0,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 4,
                            opacity: 0.65,
                        }}
                    >
                        {locked.map((item) => {
                            const hide = item.hidden;
                            return (
                                <li
                                    key={item.id}
                                    data-testid={`locked-${item.id}`}
                                    style={{ fontSize: '0.82rem' }}
                                >
                                    <span
                                        className="terminal-glow"
                                        style={{ opacity: 0.55 }}
                                    >
                                        ▢{' '}
                                    </span>
                                    {hide ? (
                                        <span
                                            className="terminal-glow"
                                            style={{ opacity: 0.4 }}
                                        >
                                            ??? — Discovery
                                        </span>
                                    ) : (
                                        <span className="terminal-glow">
                                            {item.title}{' '}
                                            <span
                                                style={{ opacity: 0.55, fontSize: '0.7rem' }}
                                            >
                                                ({item.tier} · +{item.xp})
                                            </span>
                                        </span>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                </section>
            </aside>
        </div>
    );
}
