/**
 * One Coach Mode suggestion card with a click-to-execute action
 * button (USER-SUCCESS Epic 4).
 *
 * Action kinds:
 * - ``run_playbook`` (Phase 1) — wraps ``runPlaybook`` synth endpoint.
 * - ``navigate`` (Phase 1) — surfaces a hint toast; route glue is
 *   deferred to Phase 4.
 * - ``augment_from_cluster`` (Phase 3) — wraps
 *   ``augmentFromCluster`` to generate synth rows targeting the
 *   eval's top failure cluster (Epic 2b primitive).
 */

import { Fragment, useMemo, useState, type ReactNode } from 'react';

import type { CoachSuggestion } from '../../api/coach';
import {
    augmentFromCluster,
    runPlaybook,
    type SynthMode,
} from '../../api/synthPlaybook';
import { Term } from '../shared/Term';
import { toast } from '../../stores/toastStore';

// Dictionary of plain-text phrases → glossary term IDs that
// ``wrapTermsInBody`` scans for. Multi-word phrases come before
// single-word ones so the regex prefers the longest match. Patterns
// are case-insensitive but only match on word boundaries so substrings
// like "f1234" don't collide with "F1".
//
// Keep this list aligned with the entries in
// ``components/shared/glossary.ts`` — adding a phrase here without a
// matching glossary entry would render an unbordered span (Term
// gracefully degrades to plain text in that case).
const COACH_TERM_PHRASES: Array<[RegExp, string]> = [
    [/\bpredicted[- ]pass[- ]probability\b/i, 'predicted_f1_confidence'],
    [/\bjaccard[- ]similarity\b/i, 'jaccard_similarity'],
    [/\bshannon[- ]entropy\b/i, 'shannon_entropy'],
    [/\bfailure[- ]cluster(s)?\b/i, 'failure_cluster'],
    [/\bclass[- ]imbalance\b/i, 'class_imbalance'],
    [/\bpass[- ]rate\b/i, 'pass_rate'],
    [/\bentropy\b/i, 'shannon_entropy'],
    [/\bf1\b/i, 'f1'],
];

/**
 * Scan a Coach suggestion body for known metric/jargon phrases and
 * wrap each occurrence in a ``<Term>`` popover. Returns an array of
 * mixed strings + Term nodes suitable for a ``{nodes.map(...)}``
 * render. Pure function — no React state, safe to memoize.
 *
 * The scan is single-pass left-to-right over the longest-match-first
 * pattern list, so overlapping phrases are resolved deterministically
 * (e.g. "predicted pass probability" wins over the bare "pass rate"
 * inside it).
 */
export function wrapTermsInBody(text: string): ReactNode[] {
    if (!text) return [text];
    const nodes: ReactNode[] = [];
    let cursor = 0;
    while (cursor < text.length) {
        let bestMatch: { start: number; end: number; termId: string } | null = null;
        for (const [pattern, termId] of COACH_TERM_PHRASES) {
            // Use a fresh regex per scan so the global lastIndex
            // state never leaks across iterations.
            const re = new RegExp(pattern.source, pattern.flags);
            const segment = text.slice(cursor);
            const m = re.exec(segment);
            if (m && m.index >= 0) {
                const start = cursor + m.index;
                const end = start + m[0].length;
                if (
                    !bestMatch
                    || start < bestMatch.start
                    || (start === bestMatch.start && end - start > bestMatch.end - bestMatch.start)
                ) {
                    bestMatch = { start, end, termId };
                }
            }
        }
        if (!bestMatch) {
            nodes.push(text.slice(cursor));
            break;
        }
        if (bestMatch.start > cursor) {
            nodes.push(text.slice(cursor, bestMatch.start));
        }
        const matchedText = text.slice(bestMatch.start, bestMatch.end);
        nodes.push(
            <Term
                key={`${bestMatch.termId}-${bestMatch.start}`}
                id={bestMatch.termId}
                label={matchedText}
            />,
        );
        cursor = bestMatch.end;
    }
    return nodes;
}

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
    // Cache the term-wrap parse so the regex scan doesn't re-run on
    // every state flip (e.g. while the action button is "Working…").
    const bodyNodes = useMemo(
        () => wrapTermsInBody(suggestion.body || ''),
        [suggestion.body],
    );

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
        if (suggestion.action.kind === 'augment_from_cluster') {
            const evalResultId = Number(
                suggestion.action.params['eval_result_id'],
            );
            const clusterId = String(
                suggestion.action.params['cluster_id'] ?? '',
            );
            const targetCount = Number(
                suggestion.action.params['target_count'] ?? 30,
            );
            if (
                !Number.isFinite(evalResultId)
                || !clusterId
                || !Number.isFinite(targetCount)
                || targetCount < 1
            ) {
                toast.error('Coach suggestion is missing action parameters.');
                return;
            }
            setIsExecuting(true);
            try {
                const result = await augmentFromCluster(projectId, {
                    evalResultId,
                    clusterId,
                    targetCount,
                });
                toast.success(
                    `Generated ${result.rows.length} synthetic row${result.rows.length === 1 ? '' : 's'} targeting cluster ${clusterId}.`,
                );
                onActionCompleted?.();
            } catch (err) {
                const detail =
                    (err as { response?: { data?: { detail?: string } } })?.response
                        ?.data?.detail;
                toast.error(
                    detail ?? 'Coach cluster-augment failed. Check the synth panel for details.',
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
                    {bodyNodes.map((node, idx) => (
                        <Fragment key={idx}>{node}</Fragment>
                    ))}
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
