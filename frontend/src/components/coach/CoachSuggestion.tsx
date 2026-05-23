/**
 * One Coach Mode suggestion card with a click-to-execute action
 * button (USER-SUCCESS Epic 4 Phase 1).
 *
 * Phase 1 wires the ``run_playbook`` action to the existing
 * ``runPlaybook`` synth-playbook endpoint. ``navigate`` is rendered
 * as a hint (no router glue yet — the recipe-picker live in the
 * data tab already, the user is already there).
 */

import { useState } from 'react';

import type { CoachSuggestion } from '../../api/coach';
import { runPlaybook, type SynthMode } from '../../api/synthPlaybook';
import { toast } from '../../stores/toastStore';

interface CoachSuggestionCardProps {
    projectId: number;
    suggestion: CoachSuggestion;
    // Called after a successful action so the parent panel can
    // refetch its data (e.g. the IngestionPanel re-pulling its
    // ingested document count).
    onActionCompleted?: () => void;
}

const SEVERITY_COLORS: Record<
    CoachSuggestion['severity'],
    { bg: string; border: string; tag: string; fg: string }
> = {
    info: {
        bg: 'rgba(59, 130, 246, 0.08)',
        border: 'rgba(59, 130, 246, 0.35)',
        tag: 'rgba(59, 130, 246, 0.85)',
        fg: 'rgb(30, 64, 175)',
    },
    warning: {
        bg: 'rgba(245, 158, 11, 0.10)',
        border: 'rgba(245, 158, 11, 0.40)',
        tag: 'rgba(217, 119, 6, 0.95)',
        fg: 'rgb(146, 64, 14)',
    },
    critical: {
        bg: 'rgba(239, 68, 68, 0.08)',
        border: 'rgba(239, 68, 68, 0.40)',
        tag: 'rgba(220, 38, 38, 0.95)',
        fg: 'rgb(153, 27, 27)',
    },
};

export default function CoachSuggestionCard({
    projectId,
    suggestion,
    onActionCompleted,
}: CoachSuggestionCardProps) {
    const [isExecuting, setIsExecuting] = useState(false);
    const colors = SEVERITY_COLORS[suggestion.severity];

    const handleClick = async () => {
        if (suggestion.action.kind === 'run_playbook') {
            const mode = suggestion.action.params['mode'] as SynthMode | undefined;
            const targetCount = Number(suggestion.action.params['target_count']);
            const targetClass = suggestion.action.params['target_class'] as
                | string
                | null
                | undefined;
            if (!mode || !Number.isFinite(targetCount) || targetCount < 1) {
                toast.error('Coach suggestion is missing action parameters.');
                return;
            }
            setIsExecuting(true);
            try {
                const result = await runPlaybook(projectId, {
                    mode,
                    targetCount,
                    targetClass: targetClass ?? null,
                });
                toast.success(
                    `Generated ${result.rows.length} synthetic row${result.rows.length === 1 ? '' : 's'} via ${mode}.`,
                );
                onActionCompleted?.();
            } catch (err) {
                const detail =
                    (err as { response?: { data?: { detail?: string } } })?.response
                        ?.data?.detail;
                toast.error(
                    detail ?? 'Coach action failed. Check the synth panel for details.',
                );
            } finally {
                setIsExecuting(false);
            }
            return;
        }
        if (suggestion.action.kind === 'navigate') {
            // Phase 1: surface a hint. The recipe-picker for the data
            // stage already lives on the same tab, so the user can
            // act without a route change.
            toast.info(
                `Tip: ${suggestion.action.label.toLowerCase()} before retrying this Coach action.`,
            );
        }
    };

    return (
        <div
            data-testid={`coach-suggestion-${suggestion.id}`}
            style={{
                display: 'flex',
                gap: 'var(--space-md)',
                padding: 'var(--space-sm) var(--space-md)',
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: 'var(--radius-md)',
                alignItems: 'flex-start',
            }}
        >
            <div style={{ flex: 1, minWidth: 0 }}>
                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        marginBottom: 4,
                    }}
                >
                    <span
                        style={{
                            fontSize: '0.65rem',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            letterSpacing: '0.06em',
                            color: colors.tag,
                        }}
                    >
                        Coach · {suggestion.severity}
                    </span>
                </div>
                <div
                    style={{
                        fontWeight: 600,
                        color: colors.fg,
                        marginBottom: 2,
                        fontSize: 'var(--font-size-sm)',
                    }}
                >
                    {suggestion.title}
                </div>
                <div
                    style={{
                        color: 'var(--text-secondary)',
                        fontSize: 'var(--font-size-sm)',
                        lineHeight: 1.5,
                    }}
                >
                    {suggestion.body}
                </div>
            </div>
            <button
                type="button"
                className="btn btn-primary"
                style={{ fontSize: 'var(--font-size-xs)', whiteSpace: 'nowrap' }}
                onClick={handleClick}
                disabled={isExecuting}
                data-testid={`coach-suggestion-action-${suggestion.id}`}
            >
                {isExecuting ? '⏳ Working…' : suggestion.action.label}
            </button>
        </div>
    );
}
