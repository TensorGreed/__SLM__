/**
 * Inline TopBar chip showing the project's current level + XP.
 *
 * Renders as `▣ L3 · ML Engineer · 1,240 XP` with the CRT phosphor
 * glow. Clicking opens the LabJournalDrawer. Stays compact so it
 * doesn't crowd the existing status badge.
 *
 * The poll lives in `useGamificationPoller`; this component just
 * reads `useProgressionState`. If the cache is empty (first paint
 * before the poll lands) we render a neutral L1 placeholder.
 */

import { useState } from 'react';

import LabJournalDrawer from './LabJournalDrawer';
import { useProgressionState } from './useProgressionPoll';

interface ProgressChipProps {
    projectId: number;
}

export default function ProgressChip({ projectId }: ProgressChipProps) {
    const state = useProgressionState();
    const [drawerOpen, setDrawerOpen] = useState(false);

    const level = state?.level ?? 1;
    const xp = state?.xp_balance ?? 0;
    const xpInLevel = state?.xp_into_level ?? 0;
    const xpToNext = state?.xp_to_next_level ?? 100;
    const levelTitle = state?.level_title ?? 'Intern';

    // Tooltip line for the chip itself.
    const tooltip = `Lab Journal — L${level} ${levelTitle} · ${xpInLevel}/${xpToNext} XP to next`;

    return (
        <>
            <button
                type="button"
                className="terminal-glow terminal-surface"
                onClick={() => setDrawerOpen(true)}
                title={tooltip}
                aria-label={`Open Lab Journal — Level ${level}, ${xp.toLocaleString()} XP`}
                data-testid="progress-chip"
                style={{
                    cursor: 'pointer',
                    padding: '4px 10px',
                    fontSize: '0.78rem',
                    letterSpacing: '0.04em',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                    height: 28,
                    lineHeight: 1,
                }}
            >
                <span aria-hidden="true">▣</span>
                <span>L{level}</span>
                <span style={{ opacity: 0.55 }}>·</span>
                <span style={{ opacity: 0.85 }}>{xp.toLocaleString()} XP</span>
            </button>
            {drawerOpen && (
                <LabJournalDrawer
                    projectId={projectId}
                    onClose={() => setDrawerOpen(false)}
                />
            )}
        </>
    );
}
