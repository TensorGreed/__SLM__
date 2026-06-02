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
import { useNavigate } from 'react-router-dom';

import type { CoachSuggestion } from '../../api/coach';
import CoachSuggestionTrace from './CoachSuggestionTrace';
import './CoachSuggestionTrace.css';
import {
    augmentFromClusterAsync,
    runPlaybookAsync,
    type SynthMode,
} from '../../api/synthPlaybook';
import { Term } from '../shared/Term';
import { useJobsStore } from '../../stores/jobsStore';
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

// Phase 5c — known navigate targets that route to a concrete URL
// (vs. fall through to the toast-hint fallback). Builders receive
// the project id + the action's params bag so they can append
// focus / scroll hints (e.g. ``?focus_synth_source=...`` for the
// review-queue surface). Keep this map small and explicit so adding
// a new target is one obvious place to touch.
const NAVIGATE_TARGET_URLS: Record<
    string,
    (projectId: number, params: Record<string, unknown>) => string
> = {
    // Lands on the Synthetic pipeline tab where SynthReviewQueue is
    // mounted (the row-level accept/reject UI). When ``synth_source``
    // is set on the action (Coach stamps the top pending bucket), we
    // pass it as ?focus_synth_source=... so SynthReviewQueue can
    // render its one-click "Accept all N <source> rows" banner. The
    // #synth-review-queue hash is what SyntheticPanel scrolls to on
    // mount.
    'synthetic-review-queue': (projectId, params) => {
        const source = params['synth_source'];
        const query =
            typeof source === 'string' && source.length > 0
                ? `?focus_synth_source=${encodeURIComponent(source)}`
                : '';
        return `/project/${projectId}/pipeline/synthetic${query}#synth-review-queue`;
    },
    // Task-shape recipe picker — distinct from `/recipes` (which is
    // the pipeline-DAG recipes page; different concept). The
    // standalone `/recipe-picker` page sets `Project.selected_recipe`
    // via `applyRecipeToProject`, which is what every
    // recipe-required surface (synth playbooks, auto-RAG comparison,
    // archetype comparison, Coach Mode signals) actually reads.
    // Sending Coach Mode users to the pipeline-DAG page was the bug
    // — they'd pick a pipeline recipe and the "no recipe selected"
    // signals would persist.
    'recipe-picker': (projectId) => `/project/${projectId}/recipe-picker`,
    // Training Config is its own page too (separate from the
    // pipeline tab) — same fix as recipe-picker. The Phase 6d
    // curriculum nudge + Phase 9d auto-RAG nudge both emit this.
    'training-config': (projectId) => `/project/${projectId}/training-config`,
    // Base-model swap target emitted by the trainability-forecast
    // coach suggestion ("Consider Qwen/..."). The params carry
    // `recommended_base_model`; we forward it as a URL query so
    // TrainingPanel can read it on mount + auto-apply via
    // setBaseModel, plus a hash so the page scrolls to the picker.
    'training-base-model-picker': (projectId, params) => {
        const recommended = params['recommended_base_model'];
        const query =
            typeof recommended === 'string' && recommended.length > 0
                ? `?recommended_base_model=${encodeURIComponent(recommended)}`
                : '';
        return `/project/${projectId}/training-config${query}#base-model`;
    },
    // Phase 7d — Coach Mode's reroute nudge sends the user to the
    // Eval tab + scrolls to the RerouteRecommendationPanel via the
    // hash anchor. The panel reads `evalResults[0]?.id` from
    // EvalPanel so as long as an eval has run, the panel will be
    // mounted by the time the user lands.
    'reroute-recommendation-panel': (projectId) =>
        `/project/${projectId}/pipeline/eval#reroute-recommendation-panel`,
    // Sweep-inconclusive nudge — sends the user to the observability
    // page's FailureClusterList (the section was given the
    // `failure-clusters` anchor for this deep-link). When the sweep
    // verdict is inconclusive, this is where the user goes to see why
    // each cell missed the gate rather than promoting a sub-gate model.
    'failure-clusters-panel': (projectId) =>
        `/project/${projectId}/observability#failure-clusters`,
};

export default function CoachSuggestionCard({
    projectId,
    suggestion,
    onActionCompleted,
}: CoachSuggestionCardProps) {
    const [isExecuting, setIsExecuting] = useState(false);
    const navigate = useNavigate();
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
            // Phase 5c — coach_service stamps a schema-aware backend
            // pin on suggestions whose playbook defines response_schema
            // (today: class_balance_fill only). Forward it so the
            // orchestrator routes the run through vLLM / NeMo instead
            // of auto-picking Ollama and losing constrained decoding.
            const backend = suggestion.action.params['backend'] as
                | string
                | null
                | undefined;
            if (!mode || !Number.isFinite(targetCount) || targetCount < 1) {
                toast.error('Coach suggestion is missing action parameters.');
                return;
            }
            setIsExecuting(true);
            try {
                // Hardening — fire as a background Job (matches the
                // SyntheticPanel pattern). The bell takes over progress
                // + outcome; clicking the action no longer blocks for
                // the 30-180s LLM call.
                const job = await runPlaybookAsync(projectId, {
                    mode,
                    targetCount,
                    targetClass: targetClass ?? null,
                    backend: backend ?? null,
                });
                toast.info(
                    `Synth ${mode} queued — track in the bell (job #${job.id})`,
                    4000,
                );
                void useJobsStore.getState().refreshAfterLocalChange();
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
                // Hardening — fire as a background Job. Cluster-augment
                // is an LLM call that can take 30-180s; blocking the
                // request was the original "nothing happens" pain
                // point. The bell now surfaces progress + outcome.
                const job = await augmentFromClusterAsync(projectId, {
                    evalResultId,
                    clusterId,
                    targetCount,
                });
                toast.info(
                    `Cluster-augment queued — track in the bell (job #${job.id})`,
                    4000,
                );
                void useJobsStore.getState().refreshAfterLocalChange();
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
            // Phase 5c — known targets route to a concrete URL so the
            // action is genuinely one-click. Unknown targets fall back
            // to the Phase 1 toast-hint behavior (the recipe-picker on
            // the data stage, for instance, lives on the same tab so a
            // hint is appropriate). React-Router's `navigate` preserves
            // the hash, which the Data Studio page reads on mount to
            // scroll + expand the matching section.
            const target = suggestion.action.params['target'];
            // Hardening — base-model swap also dispatches a window
            // CustomEvent so TrainingPanel can react when the user
            // clicks from a Coach surface that's *already on the
            // training-config page*. Without this, react-router's
            // same-path navigate() doesn't re-mount the panel and
            // the URL-param read on mount never re-fires.
            if (
                target === 'training-base-model-picker'
                && typeof window !== 'undefined'
            ) {
                const recommended = suggestion.action.params['recommended_base_model'];
                if (typeof recommended === 'string' && recommended.length > 0) {
                    window.dispatchEvent(
                        new CustomEvent('brewslm:apply-recommended-base-model', {
                            detail: { recommendedBaseModel: recommended },
                        }),
                    );
                }
            }
            if (typeof target === 'string' && target in NAVIGATE_TARGET_URLS) {
                navigate(
                    NAVIGATE_TARGET_URLS[target](
                        projectId,
                        suggestion.action.params,
                    ),
                );
                onActionCompleted?.();
                return;
            }
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
                <CoachSuggestionTrace suggestion={suggestion} />
            </div>
            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-end',
                    gap: 4,
                    minWidth: 0,
                }}
            >
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
                {(() => {
                    // Phase 5c — surface auto-pinned backend so users
                    // see the constrained-decoding upgrade happening
                    // instead of it being silently applied. Reads
                    // params.backend (set by coach_service when a
                    // schema-aware backend is configured); the
                    // ``· schema-aware`` chip renders when
                    // context.schema_aware_backend matches (which is
                    // also stamped by coach_service so the UI doesn't
                    // have to maintain its own backend-name allowlist).
                    const params = suggestion.action.params ?? {};
                    const ctx = suggestion.context ?? {};
                    const pinnedBackend =
                        typeof params['backend'] === 'string'
                            ? (params['backend'] as string)
                            : null;
                    if (!pinnedBackend) return null;
                    const schemaAwarePin =
                        typeof ctx['schema_aware_backend'] === 'string'
                            ? (ctx['schema_aware_backend'] as string)
                            : null;
                    const isSchemaAware = schemaAwarePin === pinnedBackend;
                    return (
                        <div
                            style={{
                                fontSize: '0.65rem',
                                color: 'var(--text-tertiary)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 6,
                                maxWidth: 240,
                                textAlign: 'right',
                                lineHeight: 1.35,
                            }}
                            data-testid={`coach-suggestion-backend-${suggestion.id}`}
                        >
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                will run on <code style={{
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: '0.65rem',
                                    padding: '0 3px',
                                    background: 'var(--gray-50)',
                                    borderRadius: 'var(--radius-sm)',
                                    color: 'var(--text-secondary)',
                                }}>{pinnedBackend}</code>
                            </span>
                            {isSchemaAware && (
                                <span
                                    title="This backend forwards the playbook's JSON Schema as response_format=json_schema and enforces it during decoding."
                                    style={{
                                        padding: '1px 6px',
                                        border: '1px solid var(--color-success, #15803d)',
                                        borderRadius: 999,
                                        background: 'var(--color-success-bg, rgba(34, 197, 94, 0.10))',
                                        color: 'var(--color-success-fg, #166534)',
                                        fontSize: '0.6rem',
                                        fontWeight: 600,
                                        whiteSpace: 'nowrap',
                                        cursor: 'help',
                                    }}
                                    data-testid={`coach-suggestion-schema-badge-${suggestion.id}`}
                                >
                                    ✓ schema-aware
                                </span>
                            )}
                        </div>
                    );
                })()}
            </div>
        </div>
    );
}
