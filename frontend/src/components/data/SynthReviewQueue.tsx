/**
 * SynthReviewQueue — USER-SUCCESS Epic 2b.
 *
 * Lists pending synthetic rows grouped by `synth_source` so the user
 * can accept or reject batches. Per [rejected-rows-selectable]
 * memory: bulk select + bulk drop, never all-or-nothing.
 *
 * The dataset-prep gate (`_load_records_from_file(..., include_pending_synth=False)`)
 * keeps pending rows out of training; this surface is what un-blocks
 * them.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';
import { budgetExceededMessage, budgetWarningMessage } from '../../api/llmSpend';

import GenerationModelPicker from '../shared/GenerationModelPicker';
import type {
    AssistConsensus,
    ReviewAssist,
    ReviewAssistJudge,
    ReviewQueueEntry,
    ReviewAssistVerdict,
    ReviewQueueGroup,
    ReviewQueueGroupBy,
    ReviewQueueResponse,
    PlanRulesReport,
} from '../../api/synthPlaybook';
import {
    bulkUpdateSynthReviewQueue,
    checkPlanRules,
    listSynthReviewQueue,
    purgeRejectedSynthRows,
    recheckCodeFormat,
    type RecheckCodeFormatResult,
    startReviewAssist,
} from '../../api/synthPlaybook';
import './SynthReviewQueue.css';

interface Props {
    projectId: number;
    /** Phase 5c — when set, render a banner at the top of the panel
     *  with a one-click "Accept all N <source> rows" button. Coach
     *  Mode passes this via the URL (?focus_synth_source=...) so the
     *  click-to-execute loop closes in two clicks instead of N. */
    focusSource?: string | null;
}

type SelectedSet = Set<number>;

export default function SynthReviewQueue({ projectId, focusSource }: Props) {
    const [data, setData] = useState<ReviewQueueResponse | null>(null);
    const [selected, setSelected] = useState<SelectedSet>(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [flash, setFlash] = useState<string | null>(null);
    // Arc 5 — purge confirmation lives on the parent so it survives
    // the post-purge re-render (after a successful purge the rejected
    // section unmounts; the flash needs to outlive it).
    const [purgeFlash, setPurgeFlash] = useState<string | null>(null);
    const [purgeError, setPurgeError] = useState<string | null>(null);
    // ``focusActive`` lets the user dismiss the focus banner without
    // losing the navigation context (URL hash + query stay). Set when
    // the focusSource prop arrives; cleared on dismiss + on successful
    // bulk-accept (no point keeping a banner for an empty bucket).
    const [focusActive, setFocusActive] = useState<boolean>(!!focusSource);
    useEffect(() => {
        setFocusActive(!!focusSource);
    }, [focusSource]);

    // Epic E — group the queue by synth source (default), class label, or the
    // originating failure cluster.
    const [groupBy, setGroupBy] = useState<ReviewQueueGroupBy>('source');
    // Provenance filters: one Job / one backend (picked from the queue's facets).
    const [filterJob, setFilterJob] = useState<number | null>(null);
    const [filterBackend, setFilterBackend] = useState<string | null>(null);
    // Pre-screen (review assist) filters: verdict and reason code.
    const [filterVerdict, setFilterVerdict] = useState<ReviewAssistVerdict | null>(null);
    const [filterReason, setFilterReason] = useState<string | null>(null);
    // Several judges per row: filter by where they agree, or view one judge's verdicts.
    const [filterConsensus, setFilterConsensus] = useState<string | null>(null);
    const [filterJudge, setFilterJudge] = useState<string | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const payload = await listSynthReviewQueue(projectId, groupBy, {
                synthJobId: filterJob,
                synthBackend: filterBackend,
                verdict: filterVerdict,
                reasonCode: filterReason,
                assistConsensus: filterConsensus,
                judge: filterJudge,
            });
            setData(payload);
            setSelected(new Set());
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load review queue');
        } finally {
            setLoading(false);
        }
    }, [projectId, groupBy, filterJob, filterBackend, filterVerdict, filterReason, filterConsensus, filterJudge]);

    useEffect(() => {
        load();
    }, [load]);

    const handlePurgeCompleted = useCallback(
        async (result: import('../../api/synthPlaybook').PurgeRejectedResult) => {
            setPurgeFlash(
                `Purged ${result.purged} row${result.purged === 1 ? '' : 's'} from synthetic.jsonl. ${result.retained} rejected row${result.retained === 1 ? '' : 's'} remaining.`,
            );
            setPurgeError(null);
            await load();
        },
        [load],
    );
    const handlePurgeError = useCallback((msg: string) => {
        setPurgeError(msg);
        setPurgeFlash(null);
    }, []);

    const handleBulkUpdate = useCallback(
        async (action: 'accept' | 'reject') => {
            if (selected.size === 0) return;
            setBusy(true);
            setFlash(null);
            try {
                const result = await bulkUpdateSynthReviewQueue(projectId, {
                    rowIds: Array.from(selected),
                    action,
                });
                if (action === 'accept') {
                    setFlash(
                        `Accepted ${result.accepted} row${result.accepted === 1 ? '' : 's'}. ${result.total_remaining_pending} still pending.`,
                    );
                } else {
                    setFlash(
                        `Rejected ${result.rejected} row${result.rejected === 1 ? '' : 's'}. ${result.total_remaining_pending} still pending.`,
                    );
                }
                await load();
            } catch (err: any) {
                setError(err?.response?.data?.detail || err?.message || 'Bulk update failed');
            } finally {
                setBusy(false);
            }
        },
        [projectId, selected, load],
    );

    // Phase 5c — gather the pending row ids that belong to the
    // focused source (exact match on synth_source). The one-click
    // banner uses this to bulk-accept without forcing the user to
    // multi-select. We match on equality (not substring) so the
    // banner can't accidentally include rows from a sibling source
    // like ``...class=billing`` when the focus is on
    // ``...class=technical``.
    const focusedRowIds: number[] = (() => {
        if (!focusActive || !focusSource || !data) return [];
        const out: number[] = [];
        for (const group of data.groups) {
            if (group.synth_source === focusSource) {
                for (const row of group.rows) {
                    out.push(row.id);
                }
            }
        }
        return out;
    })();

    const handleAcceptFocused = useCallback(async () => {
        if (focusedRowIds.length === 0) return;
        setBusy(true);
        setFlash(null);
        try {
            const result = await bulkUpdateSynthReviewQueue(projectId, {
                rowIds: focusedRowIds,
                action: 'accept',
            });
            setFlash(
                `Accepted ${result.accepted} row${result.accepted === 1 ? '' : 's'} from ${focusSource}. ${result.total_remaining_pending} still pending.`,
            );
            // Clear the focus banner once the source is drained —
            // keeping it visible would imply more rows are coming.
            setFocusActive(false);
            await load();
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Bulk accept failed');
        } finally {
            setBusy(false);
        }
    }, [focusedRowIds, focusSource, projectId, load]);

    // Bulk accept / reject a whole group. The ids are NOT taken from what was rendered (a Job may have
    // re-judged rows, or another tab accepted some, since): the queue is re-fetched right before the
    // call and only the group's CURRENT pending rows whose verdict (and reason) still match go in.
    const handleGroupUpdate = async (group: ReviewQueueGroup, action: 'accept' | 'reject') => {
        const byConsensus = groupBy === 'assist_consensus';
        const verdict = byConsensus ? null : assistVerdictOf(group);
        if (action === 'accept' && !(byConsensus ? canAcceptConsensusGroup(group) : canAcceptGroup(group))) return;
        if (action === 'reject' && byConsensus && !canRejectConsensusGroup(group)) return;
        const code = byConsensus ? null : assistReasonOf(group);
        setBusy(true);
        setFlash(null);
        setError(null);
        try {
            const fresh = await listSynthReviewQueue(projectId, groupBy, {
                synthJobId: filterJob,
                synthBackend: filterBackend,
                verdict: filterVerdict,
                reasonCode: filterReason,
                assistConsensus: filterConsensus,
                judge: filterJudge,
            });
            const current = fresh.groups.find((g) => g.synth_source === group.synth_source);
            const matching = (current?.rows ?? []).filter((r) => (byConsensus
                ? consensusKeyOf(r) === group.synth_source
                : verdict === null
                    ? !r.review_assist
                    : r.review_assist?.verdict === verdict && (code === null || r.review_assist?.reason_code === code)));
            // Where several judges ran, one judge's pass never accepts a row another judge disagrees on.
            const ids = matching
                .filter((r) => !(action === 'accept' && r.assist_consensus === 'disagree'))
                .map((r) => r.id);
            if (ids.length === 0) {
                setFlash(`${group.synth_source} has no pending rows left to ${action}; the queue changed since it was loaded.`);
                setData(fresh);
                setSelected(new Set());
                return;
            }
            const result = await bulkUpdateSynthReviewQueue(projectId, {
                rowIds: ids,
                action,
                rejectReason: action === 'reject'
                    ? (byConsensus ? `assist:${group.synth_source.replace(':', '_')}` : code ? `assist:${code}` : null)
                    : null,
            });
            setFlash(
                action === 'accept'
                    ? `Accepted ${result.accepted} row${result.accepted === 1 ? '' : 's'} from ${group.synth_source}. ${result.total_remaining_pending} still pending.`
                    : `Rejected ${result.rejected} row${result.rejected === 1 ? '' : 's'} from ${group.synth_source}${code ? ` (assist:${code})` : ''}. ${result.total_remaining_pending} still pending.`,
            );
            await load();
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Bulk update failed');
        } finally {
            setBusy(false);
        }
    };

    const toggleRow = (id: number) => {
        setSelected((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    const toggleGroup = (group: ReviewQueueGroup) => {
        const ids = group.rows.map((r) => r.id);
        const allSelected = ids.every((id) => selected.has(id));
        setSelected((prev) => {
            const next = new Set(prev);
            if (allSelected) {
                ids.forEach((id) => next.delete(id));
            } else {
                ids.forEach((id) => next.add(id));
            }
            return next;
        });
    };

    if (loading) {
        return (
            <section className="synth-review-queue synth-review-queue--loading" data-testid="synth-review-queue-loading">
                <p>Loading review queue…</p>
            </section>
        );
    }

    if (error) {
        return (
            <section className="synth-review-queue synth-review-queue--error" data-testid="synth-review-queue-error">
                <p>{error}</p>
                <button type="button" onClick={load} className="btn btn-secondary">Retry</button>
            </section>
        );
    }

    const totalRejected = data?.total_rejected ?? 0;
    if (
        !data
        || (data.total_pending === 0 && data.total_accepted === 0 && totalRejected === 0)
    ) {
        return (
            <section className="synth-review-queue synth-review-queue--empty" data-testid="synth-review-queue-empty">
                <p>No synth rows pending review.</p>
                {purgeFlash && (
                    <p
                        className="synth-review-queue__rejected-flash"
                        data-testid="synth-review-queue-purge-flash"
                    >
                        {purgeFlash}
                    </p>
                )}
            </section>
        );
    }

    // If only accepted / rejected rows exist (queue is empty), render
    // a compact "what's queued for training + what's on hold" summary
    // so non-pending state stays visible.
    if (data.total_pending === 0) {
        return (
            <section className="synth-review-queue" data-testid="synth-review-queue">
                <header className="synth-review-queue__head">
                    <h3 className="synth-review-queue__title">Synth review queue</h3>
                    <p className="synth-review-queue__subtitle">
                        No rows pending review. <strong>{data.total_accepted}</strong> row{data.total_accepted === 1 ? '' : 's'} accepted and queued for the next training run.
                    </p>
                </header>
                <TotalCountStrip data={data} />
                <StaleConventionsStrip projectId={projectId} groups={[...data.groups, ...data.accepted_groups]} onMarked={load} labellerPins={data.labeller_pins || []} />
                <RetiredGroupsList projectId={projectId} groups={data.rejected_groups || []} onChanged={load} />
                {data.accepted_groups.length > 0 && (
                    <AcceptedRowsSection
                        groups={data.accepted_groups}
                        totalAccepted={data.total_accepted}
                        projectId={projectId}
                        onChanged={load}
                    />
                )}
                {purgeFlash && (
                    <p
                        className="synth-review-queue__rejected-flash"
                        data-testid="synth-review-queue-purge-flash"
                    >
                        {purgeFlash}
                    </p>
                )}
                {purgeError && (
                    <p
                        className="synth-review-queue__rejected-error"
                        data-testid="synth-review-queue-purge-error"
                    >
                        {purgeError}
                    </p>
                )}
                {(data.rejected_groups?.length ?? 0) > 0 && (
                    <RejectedRowsSection
                        projectId={projectId}
                        groups={data.rejected_groups || []}
                        totalRejected={totalRejected}
                        onPurgeCompleted={handlePurgeCompleted}
                        onPurgeError={handlePurgeError}
                    />
                )}
            </section>
        );
    }

    return (
        <section className="synth-review-queue" data-testid="synth-review-queue">
            <header className="synth-review-queue__head">
                <div className="synth-review-queue__head-row">
                    <h3 className="synth-review-queue__title">Synth review queue</h3>
                    <div
                        className="synth-review-queue__groupby"
                        role="group"
                        aria-label="Group review queue by"
                    >
                        <span>Group by</span>
                        {GROUP_BY_OPTIONS.map((opt) => (
                            <button
                                key={opt.value}
                                type="button"
                                className={`synth-review-queue__groupby-btn${groupBy === opt.value ? ' is-active' : ''}`}
                                aria-pressed={groupBy === opt.value ? 'true' : 'false'}
                                onClick={() => setGroupBy(opt.value)}
                                data-testid={`synth-review-queue-groupby-${opt.value}`}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                </div>
                <ProvenanceFilters
                    data={data}
                    job={filterJob}
                    backend={filterBackend}
                    verdict={filterVerdict}
                    reason={filterReason}
                    onJob={setFilterJob}
                    onBackend={setFilterBackend}
                    onVerdict={setFilterVerdict}
                    onReason={setFilterReason}
                    consensus={filterConsensus}
                    judge={filterJudge}
                    onConsensus={setFilterConsensus}
                    onJudge={setFilterJudge}
                />
                <ReviewAssistCard
                    projectId={projectId}
                    pending={data.total_pending}
                    synthJobId={filterJob}
                    onStarted={() => { setFlash('Pre-screen started — verdicts appear here as it runs (reload to refresh).'); }}
                />
                <FormatRecheckCard
                    projectId={projectId}
                    onApplied={(n) => { setFlash(`Marked ${n} mislabelled code example${n === 1 ? '' : 's'} as rejected (language_mismatch). Once the rejected view offers Restore group, they can be restored from there.`); void load(); }}
                />


                {hasPlanRows(data) && (
                    <PlanRulesCard projectId={projectId} onApplied={(n) => { setFlash(`Plan rules rejected ${n} row${n === 1 ? '' : 's'} (grouped by rule in Rejected; restorable).`); void load(); }} />
                )}
                <p className="synth-review-queue__subtitle">
                    {data.total_pending} row{data.total_pending === 1 ? '' : 's'} awaiting review,
                    grouped by {groupBy},{' '}
                    <strong>most uncertain first</strong> (lowest confidence —
                    where your review matters most). Accept to add to training; reject to discard.
                    {data.total_accepted > 0 && (
                        <> {' · '}<strong>{data.total_accepted}</strong> already accepted (see below).</>
                    )}
                </p>
            </header>
            <TotalCountStrip data={data} />
            <StaleConventionsStrip projectId={projectId} groups={[...data.groups, ...data.accepted_groups]} onMarked={load} labellerPins={data.labeller_pins || []} />
            <RetiredGroupsList projectId={projectId} groups={data.rejected_groups || []} onChanged={load} />

            {focusActive && focusSource && (
                <div
                    className={
                        'synth-review-queue__focus-banner'
                        + (focusedRowIds.length === 0 ? ' is-empty' : '')
                    }
                    data-testid="synth-review-queue-focus-banner"
                >
                    <div className="synth-review-queue__focus-text">
                        <span className="synth-review-queue__focus-label">
                            Focused on
                        </span>
                        <code
                            className="synth-review-queue__focus-source"
                            data-testid="synth-review-queue-focus-source"
                        >
                            {focusSource}
                        </code>
                        <span className="synth-review-queue__focus-count">
                            {focusedRowIds.length} pending row
                            {focusedRowIds.length === 1 ? '' : 's'}
                        </span>
                    </div>
                    <div className="synth-review-queue__focus-actions">
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={handleAcceptFocused}
                            disabled={focusedRowIds.length === 0 || busy}
                            data-testid="synth-review-queue-focus-accept-all"
                        >
                            {focusedRowIds.length === 0
                                ? 'Nothing to accept'
                                : `Accept all ${focusedRowIds.length} row${focusedRowIds.length === 1 ? '' : 's'}`}
                        </button>
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => setFocusActive(false)}
                            disabled={busy}
                            data-testid="synth-review-queue-focus-clear"
                        >
                            Clear focus
                        </button>
                    </div>
                </div>
            )}

            <div className="synth-review-queue__bulk-actions">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => handleBulkUpdate('accept')}
                    disabled={selected.size === 0 || busy}
                    data-testid="synth-review-queue-accept"
                >
                    Accept selected ({selected.size})
                </button>
                <button
                    type="button"
                    className="btn btn-secondary synth-review-queue__reject-btn"
                    onClick={() => handleBulkUpdate('reject')}
                    disabled={selected.size === 0 || busy}
                    data-testid="synth-review-queue-reject"
                >
                    Reject selected ({selected.size})
                </button>
                {flash && <span className="synth-review-queue__flash" data-testid="synth-review-queue-flash">{flash}</span>}
            </div>

            {data.groups.map((group) => {
                const allSelected = group.rows.every((r) => selected.has(r.id));
                return (
                    <div
                        key={group.synth_source}
                        className="synth-review-queue__group"
                        data-testid={`synth-review-queue-group-${group.synth_source}`}
                    >
                        <div className="synth-review-queue__group-head">
                            <label className="synth-review-queue__group-toggle">
                                <input
                                    type="checkbox"
                                    checked={allSelected}
                                    onChange={() => toggleGroup(group)}
                                    aria-label={`Select all rows from ${group.synth_source}`}
                                />
                                <code>{group.synth_source}</code>
                                {group.label && (
                                    <span className="synth-review-queue__group-label" data-testid="synth-review-queue-group-label">
                                        {group.label}
                                    </span>
                                )}
                                <span className="synth-review-queue__group-count">{group.count}</span>
                            </label>
                            {(groupBy === 'assist_reason' || groupBy === 'verdict' || groupBy === 'assist_consensus') && group.rows.length > 0 && (
                                <span className="synth-review-queue__group-actions">
                                    {(groupBy === 'assist_consensus' ? canAcceptConsensusGroup(group) : canAcceptGroup(group)) && (
                                        <button
                                            type="button"
                                            className="btn btn-primary"
                                            onClick={() => handleGroupUpdate(group, 'accept')}
                                            disabled={busy}
                                            data-testid={`synth-review-queue-group-accept-${group.synth_source}`}
                                        >
                                            Accept group ({group.count})
                                        </button>
                                    )}
                                    {(groupBy !== 'assist_consensus' || canRejectConsensusGroup(group)) && (
                                        <button
                                            type="button"
                                            className="btn btn-secondary synth-review-queue__reject-btn"
                                            onClick={() => handleGroupUpdate(group, 'reject')}
                                            disabled={busy}
                                            data-testid={`synth-review-queue-group-reject-${group.synth_source}`}
                                        >
                                            Reject group ({group.count})
                                        </button>
                                    )}
                                </span>
                            )}
                        </div>
                        <ul className="synth-review-queue__rows">
                            {group.rows.map((row) => (
                                <li
                                    key={row.id}
                                    className={
                                        'synth-review-queue__row'
                                        + (selected.has(row.id) ? ' is-selected' : '')
                                    }
                                    data-testid={`synth-review-queue-row-${row.id}`}
                                >
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={selected.has(row.id)}
                                            onChange={() => toggleRow(row.id)}
                                            aria-label={`Select row ${row.id}`}
                                        />
                                        <span className="synth-review-queue__confidence">
                                            {Math.round((row.synth_confidence ?? 0) * 100)}%
                                        </span>
                                        <code>{row.preview}</code>
                                    </label>
                                    {(row.review_assist_judges?.length ?? 0) >= 2
                                        ? <AssistJudges judges={row.review_assist_judges ?? []} consensus={row.assist_consensus} rowId={row.id} />
                                        : row.review_assist && <AssistBadge assist={row.review_assist} rowId={row.id} />}
                                    {(row.synth_job_id != null || row.synth_backend) && (
                                        <span className="synth-review-queue__provenance" data-testid={`synth-review-queue-provenance-${row.id}`}>
                                            {row.synth_job_id != null ? `job #${row.synth_job_id}` : 'no job'}
                                            {row.synth_backend ? ` · ${row.synth_backend}` : ''}
                                            {row.synth_batch ? ' · batch' : ''}
                                        </span>
                                    )}
                                </li>
                            ))}
                        </ul>
                    </div>
                );
            })}

            {data.accepted_groups.length > 0 && (
                <AcceptedRowsSection
                    groups={data.accepted_groups}
                    totalAccepted={data.total_accepted}
                    projectId={projectId}
                    onChanged={load}
                />
            )}
            {purgeFlash && (
                <p
                    className="synth-review-queue__rejected-flash"
                    data-testid="synth-review-queue-purge-flash"
                >
                    {purgeFlash}
                </p>
            )}
            {purgeError && (
                <p
                    className="synth-review-queue__rejected-error"
                    data-testid="synth-review-queue-purge-error"
                >
                    {purgeError}
                </p>
            )}
            {(data.rejected_groups?.length ?? 0) > 0 && (
                <RejectedRowsSection
                    projectId={projectId}
                    groups={data.rejected_groups || []}
                    totalRejected={data.total_rejected ?? 0}
                    onPurgeCompleted={handlePurgeCompleted}
                    onPurgeError={handlePurgeError}
                />
            )}
        </section>
    );
}


const GROUP_BY_OPTIONS: { value: ReviewQueueGroupBy; label: string }[] = [
    { value: 'source', label: 'Source' },
    { value: 'class', label: 'Class' },
    { value: 'cluster', label: 'Cluster' },
    { value: 'job', label: 'Job' },
    { value: 'backend', label: 'Backend' },
    { value: 'verdict', label: 'Verdict' },
    { value: 'assist_reason', label: 'Verdict + reason' },
    { value: 'assist_consensus', label: 'Judges agree?' },
];

/** A row's consensus group key as the queue builds it: `single` splits by the lone judge's verdict. */
function consensusKeyOf(row: ReviewQueueEntry): string {
    const c = row.assist_consensus ?? 'none';
    return c === 'single' ? `single:${row.review_assist_judges?.[0]?.verdict ?? 'unsure'}` : c;
}

/** Accept-group on a consensus grouping: only where every judge that ran says pass (two judges agreeing,
 *  or the one judge that ran). Never `disagree`, `none`, `unsure`, or a fail group. */
function canAcceptConsensusGroup(group: ReviewQueueGroup): boolean {
    return group.synth_source === 'agree_pass' || group.synth_source === 'single:pass';
}

function canRejectConsensusGroup(group: ReviewQueueGroup): boolean {
    return group.synth_source === 'agree_fail' || group.synth_source === 'single:fail';
}

/** Every judge's verdict + note on a row judged by 2+ models, with where they stand. */
function AssistJudges({ judges, consensus, rowId }: { judges: ReviewAssistJudge[]; consensus?: AssistConsensus; rowId: number }) {
    return (
        <span className="synth-review-queue__judges" data-testid={`synth-review-queue-judges-${rowId}`}>
            <span className={`synth-review-queue__consensus synth-review-queue__consensus--${consensus ?? 'none'}`}>
                {(consensus ?? 'none').replace('_', ' ')}
            </span>
            {judges.map((j) => (
                <span
                    key={j.judge}
                    className={`synth-review-queue__assist synth-review-queue__assist--${j.verdict}`}
                    data-testid={`synth-review-queue-judge-${rowId}-${j.judge}`}
                    title={j.job_id != null ? `job #${j.job_id}` : undefined}
                >
                    <code>{j.judge}</code>{' '}
                    <strong>{j.verdict}</strong>
                    {j.reason_code !== 'ok' ? ` · ${j.reason_code}` : ''}
                    {j.note ? ` — ${j.note}` : ''}
                </span>
            ))}
        </span>
    );
}

/** The reason code of an `assist_reason` group (key `<verdict>:<reason_code>`); null for other groupings. */
function assistReasonOf(group: ReviewQueueGroup): string | null {
    const m = /^(pass|fail|unsure):(.+)$/.exec(group.synth_source);
    return m ? m[2] : null;
}

/** The pre-screen verdict of a `verdict` / `assist_reason` group (key `pass`, `fail:<reason>` ...); null for the
 *  "(not pre-screened)" bucket, which has no verdict. */
function assistVerdictOf(group: ReviewQueueGroup): ReviewAssistVerdict | null {
    const m = /^(pass|fail|unsure)(:.+)?$/.exec(group.synth_source);
    return m ? (m[1] as ReviewAssistVerdict) : null;
}

/** Accept-group is offered only on `pass` groups: never on `fail`, `unsure` or the "(not pre-screened)"
 *  bucket (those can be rejected as a group, not accepted). */
function canAcceptGroup(group: ReviewQueueGroup): boolean {
    return assistVerdictOf(group) === 'pass';
}

function AssistBadge({ assist, rowId }: { assist: ReviewAssist; rowId: number }) {
    return (
        <span
            className={`synth-review-queue__assist synth-review-queue__assist--${assist.verdict}`}
            data-testid={`synth-review-queue-assist-${rowId}`}
            title={assist.backend ? `Pre-screened by ${assist.backend}${assist.job_id != null ? ` (job #${assist.job_id})` : ''}` : undefined}
        >
            <strong>{assist.verdict}</strong>
            {assist.reason_code !== 'ok' ? ` · ${assist.reason_code}` : ''}
            {assist.note ? ` — ${assist.note}` : ''}
        </span>
    );
}

/** Whether any pending row is a structured plan (the plan-rules check only applies to those). */
function hasPlanRows(data: ReviewQueueResponse): boolean {
    return data.groups.some((g) => g.rows.some((row) => {
        const p = row.payload as Record<string, unknown> | undefined;
        return p?.row_kind === 'structured_plan' || (p != null && 'plan_json' in p);
    }));
}

/** "Check plan rules": paste / upload a rules document, see how many pending plans break each rule
 *  (a dry run changes nothing), then confirm to reject them. Rejected rows are grouped by rule in the
 *  Rejected section and can be restored; nothing is deleted. */
function PlanRulesCard({ projectId, onApplied }: { projectId: number; onApplied: (rejected: number) => void }) {
    const [open, setOpen] = useState(false);
    const [text, setText] = useState('');
    const [report, setReport] = useState<PlanRulesReport | null>(null);
    const [busy, setBusy] = useState(false);

    const run = async (dryRun: boolean) => {
        setBusy(true);
        try {
            const result = await checkPlanRules(projectId, text, dryRun);
            if (dryRun) {
                setReport(result);
            } else {
                setReport(null);
                onApplied(result.rejected);
            }
        } catch (err: any) {
            const detail = err?.response?.data;
            toast.error(
                (typeof detail?.message === 'string' ? detail.message : typeof detail?.detail === 'string' ? detail.detail : err?.message)
                ?? 'Could not check the plan rules.',
                8000,
            );
        } finally {
            setBusy(false);
        }
    };

    const confirmApply = () => {
        if (!report) return;
        if (window.confirm(`Reject ${report.flagged} pending plan${report.flagged === 1 ? '' : 's'} that break a reject rule? They stay on disk under Rejected, grouped by rule, and can be restored.`)) {
            void run(false);
        }
    };

    const onFile = async (file: File | undefined) => {
        if (!file) return;
        setText(await file.text());
        setReport(null);
    };

    return (
        <div className="synth-review-queue__rules-card" data-testid="synth-review-queue-rules-card">
            <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="synth-review-queue-rules-toggle">
                {open ? 'Hide plan rules' : 'Check plan rules'}
            </button>
            {open && (
                <div className="synth-review-queue__rules-form">
                    <p className="synth-review-queue__subtitle">
                        Your own consistency rules, checked deterministically (no model) against every pending plan.
                        A dry run only counts; you confirm before anything is rejected.
                    </p>
                    <textarea
                        value={text}
                        onChange={(e) => { setText(e.target.value); setReport(null); }}
                        rows={8}
                        placeholder='{"rules": [{"id": "...", "type": "step_requires", "match": {"operation": "create_widget"}, "require": {"path": "args.arguments.body.size", "op": "max", "value": 10}}]}'
                        data-testid="synth-review-queue-rules-text"
                    />
                    <input type="file" accept=".json,application/json" onChange={(e) => void onFile(e.target.files?.[0])} data-testid="synth-review-queue-rules-file" />
                    <div className="synth-review-queue__rules-actions">
                        <button type="button" className="btn btn-primary" disabled={busy || !text.trim()} onClick={() => void run(true)} data-testid="synth-review-queue-rules-dry-run">
                            Dry run
                        </button>
                        {report && report.flagged > 0 && (
                            <button type="button" className="btn btn-danger" disabled={busy} onClick={confirmApply} data-testid="synth-review-queue-rules-apply">
                                Reject {report.flagged} plan{report.flagged === 1 ? '' : 's'}
                            </button>
                        )}
                    </div>
                    {report && (
                        <div data-testid="synth-review-queue-rules-report">
                            <p className="synth-review-queue__subtitle">
                                Checked {report.checked} plan{report.checked === 1 ? '' : 's'}: {report.flagged} would be rejected
                                {report.warned_only > 0 ? `, ${report.warned_only} only warned` : ''}.
                            </p>
                            {Object.keys(report.by_rule).length > 0 && (
                                <table className="synth-review-queue__rules-table">
                                    <thead><tr><th>Rule</th><th>Severity</th><th>Plans</th><th>Example rows</th></tr></thead>
                                    <tbody>
                                        {Object.entries(report.by_rule).map(([id, r]) => (
                                            <tr key={id} data-testid={`synth-review-queue-rules-row-${id}`}>
                                                <td title={r.description}>{id}</td>
                                                <td>{r.severity}</td>
                                                <td>{r.count}</td>
                                                <td>{r.sample_row_ids.map((n) => `#${n}`).join(', ')}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

interface ReviewAssistCardProps {
    projectId: number;
    pending: number;
    /** Narrow the pre-screen to the Job picked in the filter, if any. */
    synthJobId: number | null;
    onStarted: () => void;
}

/** "Pre-screen": an LLM reads the pending rows and gives each a verdict + reason + note, so the
 *  reviewer can bulk-accept the clean group and look first at the failures. Never accepts or
 *  rejects by itself. Runs as the `synth_review_assist` Job (the bell reports the outcome). */
function ReviewAssistCard({ projectId, pending, synthJobId, onStarted }: ReviewAssistCardProps) {
    const [open, setOpen] = useState(false);
    const [pin, setPin] = useState<string | null>(null);
    const [rubric, setRubric] = useState('');
    const [rowsPerCall, setRowsPerCall] = useState('10');
    const [useBatch, setUseBatch] = useState(false);
    const [onlyMissing, setOnlyMissing] = useState(false);
    // Which rows a second (third...) judge looks at: all, only where another judge said fail / unsure,
    // or only where the judges already disagree.
    const [target, setTarget] = useState<'all' | 'prior_doubt' | 'disagree'>('all');
    const [busy, setBusy] = useState(false);
    const batchable = Boolean(pin && pin.startsWith('cloud:anthropic:'));
    const perCall = Math.min(25, Math.max(1, Math.floor(Number(rowsPerCall) || 10)));

    const start = async () => {
        setBusy(true);
        try {
            const job = await startReviewAssist(projectId, {
                backend: pin,
                useBatch: useBatch && batchable,
                rubricText: rubric,
                rowsPerCall: perCall,
                synthJobId,
                onlyMissingJudge: onlyMissing,
                priorVerdicts: target === 'prior_doubt' ? ['fail', 'unsure'] : null,
                consensus: target === 'disagree' ? ['disagree'] : null,
            });
            const budgetMsg = budgetWarningMessage(job);
            if (budgetMsg) toast.warning(budgetMsg, 8000);
            toast.info(
                `Pre-screening ${synthJobId != null ? `job #${synthJobId}'s ` : 'the pending '}rows with ${pin ?? 'a local model'} (job #${job.id})`
                + (useBatch && batchable ? ' as one Anthropic batch — half price, results in minutes to hours' : '')
                + '. Open the bell for progress.',
                6000,
            );
            onStarted();
        } catch (err: any) {
            const detail = err?.response?.data;
            toast.error(
                budgetExceededMessage(err)
                ?? (typeof detail?.message === 'string' ? detail.message : typeof detail?.detail === 'string' ? detail.detail : err?.message)
                ?? 'Could not start the pre-screen.',
                8000,
            );
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="synth-review-queue__assist-card" data-testid="synth-review-queue-assist-card">
            <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setOpen((v) => !v)}
                data-testid="synth-review-queue-assist-toggle"
            >
                {open ? 'Hide pre-screen' : 'Pre-screen with a model'}
            </button>
            {open && (
                <div className="synth-review-queue__assist-form">
                    <p className="synth-review-queue__subtitle">
                        A model reads each pending row and says pass / fail / unsure with a reason. It never accepts or rejects
                        anything: group by verdict, then accept or reject a group yourself.
                    </p>
                    <GenerationModelPicker
                        projectId={projectId}
                        value={pin}
                        onChange={setPin}
                        disabled={busy}
                        testIdPrefix="synth-review-queue-assist"
                        label="Pre-screen model"
                    />
                    <label className="synth-review-queue__assist-field">
                        Extra rubric (optional){' '}
                        <textarea
                            value={rubric}
                            onChange={(e) => setRubric(e.target.value)}
                            rows={3}
                            maxLength={4000}
                            placeholder="Appended to the built-in rubric for each row type, e.g. the operations a plan may use."
                            data-testid="synth-review-queue-assist-rubric"
                        />
                    </label>
                    <label className="synth-review-queue__assist-field">
                        Rows per call{' '}
                        <input
                            type="number"
                            min={1}
                            max={25}
                            value={rowsPerCall}
                            onChange={(e) => setRowsPerCall(e.target.value)}
                            data-testid="synth-review-queue-assist-rows-per-call"
                        />
                    </label>
                    <label className="synth-review-queue__assist-field">
                        <input
                            type="checkbox"
                            checked={onlyMissing}
                            onChange={(e) => setOnlyMissing(e.target.checked)}
                            data-testid="synth-review-queue-assist-only-missing"
                        />
                        {' '}Only rows this model has not judged yet (resume)
                    </label>
                    <label className="synth-review-queue__assist-field">
                        Rows to judge{' '}
                        <select
                            value={target}
                            onChange={(e) => setTarget(e.target.value as 'all' | 'prior_doubt' | 'disagree')}
                            data-testid="synth-review-queue-assist-target"
                        >
                            <option value="all">All pending rows</option>
                            <option value="prior_doubt">Where another model said fail or unsure</option>
                            <option value="disagree">Where the judges disagree</option>
                        </select>
                    </label>
                    {batchable && (
                        <label className="synth-review-queue__assist-field">
                            <input
                                type="checkbox"
                                checked={useBatch}
                                onChange={(e) => setUseBatch(e.target.checked)}
                                data-testid="synth-review-queue-assist-batch"
                            />
                            {' '}Half price (Message Batch)
                        </label>
                    )}
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={start}
                        disabled={busy || pending === 0}
                        data-testid="synth-review-queue-assist-start"
                    >
                        Pre-screen {synthJobId != null ? `job #${synthJobId}'s rows` : `${pending} pending row${pending === 1 ? '' : 's'}`}
                    </button>
                </div>
            )}
        </div>
    );
}

/** "Check code formats": pending code examples labelled json / yaml / toml / xml whose code does not parse as
 *  that format are mislabels. A dry run groups them by label + kind of failure; the user picks groups and only
 *  those rows are marked rejected as `language_mismatch` (soft: they stay in the Rejected section, nothing is
 *  purged; restore them from the rejected view's Restore group). */
function FormatRecheckCard({ projectId, onApplied }: { projectId: number; onApplied: (rejected: number) => void }) {
    const [busy, setBusy] = useState(false);
    const [preview, setPreview] = useState<RecheckCodeFormatResult | null>(null);
    const [picked, setPicked] = useState<Set<string>>(new Set());
    const groupKey = (g: { language: string; detail: string }) => `${g.language}|${g.detail}`;

    const check = async () => {
        setBusy(true);
        try {
            const res = await recheckCodeFormat(projectId, true);
            setPreview(res);
            setPicked(new Set((res.groups ?? []).map(groupKey)));
        } catch (err: any) {
            toast.error(err?.response?.data?.detail ?? err?.message ?? 'Could not check the code formats.', 6000);
        } finally {
            setBusy(false);
        }
    };
    const chosen = (preview?.groups ?? []).filter((g) => picked.has(groupKey(g)));
    const chosenCount = chosen.reduce((n, g) => n + g.count, 0);
    const apply = async () => {
        setBusy(true);
        try {
            const res = await recheckCodeFormat(projectId, false, chosen.flatMap((g) => g.row_ids));
            setPreview(null);
            onApplied(res.rejected);
        } catch (err: any) {
            toast.error(err?.response?.data?.detail ?? err?.message ?? 'Could not reject the mislabelled rows.', 6000);
        } finally {
            setBusy(false);
        }
    };
    const toggle = (key: string) => setPicked((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key); else next.add(key);
        return next;
    });

    return (
        <div className="synth-review-queue__assist-card" data-testid="synth-review-queue-format-card">
            <button type="button" className="btn btn-secondary" onClick={check} disabled={busy} data-testid="synth-review-queue-format-check">
                Check code formats
            </button>
            {preview && (
                <div className="synth-review-queue__assist-form" data-testid="synth-review-queue-format-confirm">
                    {preview.mismatches === 0 ? (
                        <p className="synth-review-queue__subtitle">
                            All {preview.checked} pending code example{preview.checked === 1 ? '' : 's'} parse as the format they are labelled with.
                        </p>
                    ) : (
                        <>
                            <p className="synth-review-queue__subtitle">
                                {preview.mismatches} of {preview.checked} pending code example{preview.checked === 1 ? '' : 's'} do not parse as
                                their label. Pick the groups to mark rejected as language_mismatch; they stay in the Rejected section
                                (restorable with Restore group) and nothing is deleted.
                            </p>
                            {preview.groups.map((g) => (
                                <div key={groupKey(g)} data-testid={`synth-review-queue-format-group-${groupKey(g)}`}>
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={picked.has(groupKey(g))}
                                            onChange={() => toggle(groupKey(g))}
                                            disabled={busy}
                                            data-testid={`synth-review-queue-format-pick-${groupKey(g)}`}
                                        />
                                        {' '}<strong>{g.language}: {g.detail}</strong> ({g.count})
                                    </label>
                                    <ul>
                                        {g.examples.map((ex, i) => (
                                            <li key={ex.id ?? i}>{ex.task || `row ${ex.id}`}</li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                            <button
                                type="button"
                                className="btn btn-primary"
                                onClick={apply}
                                disabled={busy || chosenCount === 0}
                                data-testid="synth-review-queue-format-apply"
                            >
                                Reject {chosenCount} row{chosenCount === 1 ? '' : 's'}
                            </button>
                        </>
                    )}
                    <button type="button" className="btn btn-secondary" onClick={() => setPreview(null)} disabled={busy} data-testid="synth-review-queue-format-cancel">
                        {preview.mismatches === 0 ? 'Close' : 'Cancel'}
                    </button>
                </div>
            )}
        </div>
    );
}

interface ProvenanceFiltersProps {
    data: ReviewQueueResponse;
    job: number | null;
    backend: string | null;
    verdict: ReviewAssistVerdict | null;
    reason: string | null;
    onJob: (job: number | null) => void;
    onBackend: (backend: string | null) => void;
    onVerdict: (verdict: ReviewAssistVerdict | null) => void;
    onReason: (reason: string | null) => void;
    consensus: string | null;
    judge: string | null;
    onConsensus: (consensus: string | null) => void;
    onJudge: (judge: string | null) => void;
}

/** Narrow the queue to the rows one Job / one backend wrote, so lanes can be compared. */
function ProvenanceFilters({ data, job, backend, verdict, reason, onJob, onBackend, onVerdict, onReason, consensus, judge, onConsensus, onJudge }: ProvenanceFiltersProps) {
    const jobs = data.facets?.jobs ?? [];
    const backends = data.facets?.backends ?? [];
    const verdicts = data.facets?.verdicts ?? [];
    const reasons = data.facets?.reason_codes ?? [];
    const judges = data.facets?.judges ?? [];
    const consensuses = data.facets?.consensus ?? [];
    const filtered = job != null || Boolean(backend) || Boolean(verdict) || Boolean(reason) || Boolean(consensus) || Boolean(judge);
    if (jobs.length < 2 && backends.length < 2 && verdicts.length === 0 && !filtered) return null;
    return (
        <div className="synth-review-queue__filters" data-testid="synth-review-queue-filters">
            <label>
                Job{' '}
                <select
                    value={job ?? ''}
                    onChange={(e) => onJob(e.target.value ? Number(e.target.value) : null)}
                    data-testid="synth-review-queue-filter-job"
                >
                    <option value="">All jobs</option>
                    {jobs.map((j) => (
                        <option key={j.synth_job_id} value={j.synth_job_id}>
                            job #{j.synth_job_id} ({j.count})
                        </option>
                    ))}
                </select>
            </label>
            <label>
                Backend{' '}
                <select
                    value={backend ?? ''}
                    onChange={(e) => onBackend(e.target.value || null)}
                    data-testid="synth-review-queue-filter-backend"
                >
                    <option value="">All backends</option>
                    {backends.map((b) => (
                        <option key={b.synth_backend} value={b.synth_backend}>
                            {b.synth_backend} ({b.count})
                        </option>
                    ))}
                </select>
            </label>
            {(judges.length >= 2 || judge) && (
                <>
                    <label>
                        Judges{' '}
                        <select
                            value={consensus ?? ''}
                            onChange={(e) => onConsensus(e.target.value || null)}
                            data-testid="synth-review-queue-filter-consensus"
                        >
                            <option value="">Any</option>
                            {consensuses.map((c) => (
                                <option key={c.consensus} value={c.consensus}>{c.consensus.replace('_', ' ')} ({c.count})</option>
                            ))}
                        </select>
                    </label>
                    <label>
                        Judge{' '}
                        <select
                            value={judge ?? ''}
                            onChange={(e) => onJudge(e.target.value || null)}
                            data-testid="synth-review-queue-filter-judge"
                        >
                            <option value="">All judges</option>
                            {judges.map((j) => (
                                <option key={j.judge} value={j.judge}>{j.judge} ({j.count})</option>
                            ))}
                        </select>
                    </label>
                </>
            )}
            {(verdicts.length > 0 || verdict) && (
                <>
                    <label>
                        Pre-screen verdict{' '}
                        <select
                            value={verdict ?? ''}
                            onChange={(e) => onVerdict((e.target.value || null) as ReviewAssistVerdict | null)}
                            data-testid="synth-review-queue-filter-verdict"
                        >
                            <option value="">Any</option>
                            {verdicts.map((v) => (
                                <option key={v.verdict} value={v.verdict}>{v.verdict} ({v.count})</option>
                            ))}
                        </select>
                    </label>
                    <label>
                        Reason{' '}
                        <select
                            value={reason ?? ''}
                            onChange={(e) => onReason(e.target.value || null)}
                            data-testid="synth-review-queue-filter-reason"
                        >
                            <option value="">Any</option>
                            {reasons.map((r) => (
                                <option key={r.reason_code} value={r.reason_code}>{r.reason_code} ({r.count})</option>
                            ))}
                        </select>
                    </label>
                </>
            )}
        </div>
    );
}


interface TotalCountStripProps {
    data: import('../../api/synthPlaybook').ReviewQueueResponse;
}

/**
 * Always-visible strip that surfaces the three numbers the user
 * cares about for `synthetic.jsonl`: total rows, queued for training
 * (accepted), and awaiting review (pending). Answers "how many do
 * I have, and how many are headed into training?" at a glance.
 */
function TotalCountStrip({ data }: TotalCountStripProps) {
    return (
        <div className="synth-review-queue__totals" data-testid="synth-review-queue-totals">
            <div className="synth-review-queue__totals-cell" data-testid="synth-review-queue-total-rows">
                <span className="synth-review-queue__totals-value">{data.total_rows}</span>
                <span className="synth-review-queue__totals-label">total in synthetic.jsonl</span>
            </div>
            <div className="synth-review-queue__totals-cell" data-testid="synth-review-queue-total-accepted">
                <span className="synth-review-queue__totals-value synth-review-queue__totals-value--ok">
                    {data.total_accepted}
                </span>
                <span className="synth-review-queue__totals-label">queued for training</span>
            </div>
            <div className="synth-review-queue__totals-cell" data-testid="synth-review-queue-total-pending">
                <span className="synth-review-queue__totals-value synth-review-queue__totals-value--warn">
                    {data.total_pending}
                </span>
                <span className="synth-review-queue__totals-label">awaiting review</span>
            </div>
        </div>
    );
}


interface AcceptedRowsSectionProps {
    groups: import('../../api/synthPlaybook').ReviewQueueGroup[];
    totalAccepted: number;
    projectId?: number;
    onChanged?: () => void;
}

interface RetiredGroupsListProps {
    projectId: number;
    groups: ReviewQueueGroup[];
    onChanged?: () => void;
}

/** Source groups the user retired (rejected with reason "retired"), each
 *  with Restore — retiring is reversible until the rejected rows are purged. */
function RetiredGroupsList({ projectId, groups, onChanged }: RetiredGroupsListProps) {
    const retired = groups.filter((g) =>
        g.rows.some((row) => (row.payload as Record<string, unknown>)?.reject_reason === 'retired'),
    );
    if (retired.length === 0) return null;
    const restore = async (group: ReviewQueueGroup) => {
        try {
            const n = await setGroupRetired(projectId, group.synth_source, false);
            toast.success(`${group.synth_source}: ${n} rows restored to training.`, 4000);
            onChanged?.();
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not restore the group.', 5000);
        }
    };
    return (
        <div className="synth-review-queue__retired" data-testid="synth-review-queue-retired">
            <strong>Retired groups</strong> (out of training, kept on disk):
            <ul className="synth-review-queue__stale-list">
                {retired.map((group) => (
                    <li key={group.synth_source}>
                        <code>{group.synth_source}</code>{' '}
                        <span className="synth-review-queue__stale-detail">{group.count} rows</span>{' '}
                        <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => void restore(group)}
                            data-testid={`synth-review-queue-restore-${group.synth_source}`}
                        >
                            Restore
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
}

/**
 * Collapsible "Accepted — queued for training" section that surfaces
 * what's already passed review. Each source-group is itself
 * expandable: click to reveal up to 25 sample rows (the backend
 * caps each group at 25 to avoid blowing up the payload for legacy
 * buckets with thousands of rows). Truncated groups show a
 * "showing N of total" footer.
 */
function AcceptedRowsSection({ groups, totalAccepted, projectId, onChanged }: AcceptedRowsSectionProps) {
    return (
        <details
            className="synth-review-queue__accepted"
            data-testid="synth-review-queue-accepted"
        >
            <summary className="synth-review-queue__accepted-summary">
                <span className="synth-review-queue__accepted-headline">
                    <strong>{totalAccepted}</strong> accepted row{totalAccepted === 1 ? '' : 's'} queued for training
                </span>
                <span className="synth-review-queue__accepted-hint">
                    ({groups.length} source{groups.length === 1 ? '' : 's'} — click to expand)
                </span>
            </summary>
            <div className="synth-review-queue__accepted-groups">
                {groups.map((group) => (
                    <AcceptedGroupCard key={group.synth_source} group={group} projectId={projectId} onChanged={onChanged} />
                ))}
            </div>
            <p className="synth-review-queue__accepted-footnote">
                Accepted rows enter the training corpus on the next split (Data → Split) and training run.
            </p>
        </details>
    );
}


interface StaleConventionsStripProps {
    projectId: number;
    groups: ReviewQueueGroup[];
    /** Reload the queue after a group is marked current. */
    onMarked?: () => void;
    /** Models that labelled span rows in this project (relabel pins). */
    labellerPins?: string[];
}

function staleText(c: NonNullable<ReviewQueueGroup['conventions']>): string {
    const head = `${c.span_rows - c.current_rows} of ${c.span_rows} span rows`;
    if (c.stale_rows && c.untracked_rows) {
        return `${head}: ${c.stale_rows} labelled under an older version, ${c.untracked_rows} before versions were recorded`;
    }
    return `${head} labelled ${c.stale_rows ? 'under an older version' : 'before versions were recorded'}`;
}

/** Span-row groups labelled under older labelling conventions than the
 *  project's current ones — a definition change only reaches rows labelled
 *  after it. One click relabels a group in place with the model that
 *  labelled it (the `span_relabel` Job; re-split + retrain afterwards). */
function StaleConventionsStrip({ projectId, groups, onMarked, labellerPins = [] }: StaleConventionsStripProps) {
    const [started, setStarted] = useState<Record<string, number>>({});
    // '' = each group's own labeller; else one pin for every group.
    const [withPin, setWithPin] = useState('');
    // Tiered labelling: a second model checks only rows where the relabel
    // and the project's trained model disagree ('' = no review pass).
    const [reviewPin, setReviewPin] = useState('');
    // Review a row only when the relabel and the model agree less than this.
    const [reviewBelow, setReviewBelow] = useState('0.8');
    const pinFor = (group: ReviewQueueGroup) => withPin || group.conventions?.relabel_pin || null;
    const stale = groups.filter((g) => g.conventions?.is_stale);
    // Anthropic Message Batches (half price, slower) — only for an
    // Anthropic pin; other groups' relabels stay synchronous.
    const [useBatch, setUseBatch] = useState(false);
    const batchable = (pin: string | null) => Boolean(pin && pin.startsWith('cloud:anthropic:'));
    const anyBatchable = stale.some((g) => batchable(pinFor(g)));
    // The user attests a group already follows the current conventions
    // (labelled under the current wording before versions were recorded).
    const markCurrent = async (group: ReviewQueueGroup) => {
        if (!window.confirm(
            `Mark ${group.synth_source} as labelled under the current conventions? Only do this if its labels already follow them — nothing is relabelled.`,
        )) return;
        try {
            const res = await api.post(`/projects/${projectId}/synthetic/review-queue/mark-conventions-current`, {
                source: group.synth_source,
            });
            toast.success(`${group.synth_source}: ${res.data?.rows_marked ?? 0} rows marked current.`, 4000);
            onMarked?.();
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not mark the group current.', 5000);
        }
    };
    if (stale.length === 0) return null;
    const relabel = async (group: ReviewQueueGroup) => {
        const pin = pinFor(group);
        if (!pin) return;
        try {
            const res = await api.post(`/projects/${projectId}/synthetic/relabel-answer-key`, {
                target: 'training',
                source: group.synth_source,
                backend: pin,
                // Only the stale rows the convention change touches; the
                // rest are carried forward without a model call.
                only_stale: true,
                only_affected: true,
                ...(reviewPin ? { review_backend: reviewPin, review_min_agreement: Number(reviewBelow) } : {}),
                ...(useBatch && batchable(pin) ? { use_batch: true } : {}),
            });
            setStarted((prev) => ({ ...prev, [group.synth_source]: res.data?.id }));
            const budgetMsg = budgetWarningMessage(res.data);
            if (budgetMsg) toast.warning(budgetMsg, 8000);
            toast.info(
                `Relabelling ${group.synth_source} with ${pin} under the current conventions (job #${res.data?.id})`
                + (useBatch && batchable(pin) ? ' as one Anthropic batch — half price, results in minutes to hours' : '')
                + '. Re-split (keep the test examples) and retrain when it finishes.',
                6000,
            );
        } catch (err) {
            toast.error(budgetExceededMessage(err) ?? (err instanceof Error ? err.message : 'Could not start the relabel.'), 8000);
        }
    };
    return (
        <div className="synth-review-queue__stale" data-testid="synth-review-queue-stale">
            <p className="synth-review-queue__stale-head">
                <strong>{stale.length}</strong> group{stale.length === 1 ? ' was' : 's were'} labelled under older labelling
                conventions — a definition change only reaches the rows labelled after it.
            </p>
            {labellerPins.length > 0 && (
                <label className="synth-review-queue__stale-with">
                    Relabel with{' '}
                    <select
                        value={withPin}
                        onChange={(e) => setWithPin(e.target.value)}
                        data-testid="synth-review-queue-relabel-with"
                    >
                        <option value="">each group&apos;s own labeller</option>
                        {labellerPins.map((pin) => (
                            <option key={pin} value={pin}>{pin}</option>
                        ))}
                    </select>
                </label>
            )}
            {labellerPins.length > 0 && (
                <label className="synth-review-queue__stale-with">
                    Review disagreements with{' '}
                    <select
                        value={reviewPin}
                        onChange={(e) => setReviewPin(e.target.value)}
                        data-testid="synth-review-queue-review-with"
                        title="Only rows where the relabel and your trained model disagree go to this model — e.g. Haiku for the bulk, Sonnet for the disagreements."
                    >
                        <option value="">no review pass</option>
                        {labellerPins.map((pin) => (
                            <option key={pin} value={pin}>{pin}</option>
                        ))}
                    </select>
                    {reviewPin && (
                        <>
                            {' '}when a row&apos;s agreement with the model is below{' '}
                            <select
                                value={reviewBelow}
                                onChange={(e) => setReviewBelow(e.target.value)}
                                data-testid="synth-review-queue-review-below"
                                title="Row span F1 between the relabel and your trained model's prediction. 1.0 sends every row with any difference."
                            >
                                {['0.5', '0.7', '0.8', '0.9', '1.0'].map((v) => <option key={v} value={v}>{v}</option>)}
                            </select>
                        </>
                    )}
                </label>
            )}
            <label
                className="synth-review-queue__stale-with"
                title={anyBatchable
                    ? 'Send the relabel as one Anthropic Message Batch — half the list price; results arrive in minutes to hours. Only Anthropic-labelled groups batch.'
                    : 'Batch mode needs an Anthropic model (cloud:anthropic:…).'}
            >
                <input
                    type="checkbox"
                    checked={useBatch && anyBatchable}
                    disabled={!anyBatchable}
                    onChange={(e) => setUseBatch(e.target.checked)}
                    data-testid="synth-review-queue-relabel-batch"
                />{' '}
                Batch (half price, slower)
            </label>
            <ul className="synth-review-queue__stale-list">
                {stale.map((group) => {
                    const c = group.conventions!;
                    const jobId = started[group.synth_source];
                    return (
                        <li key={group.synth_source} data-testid={`synth-review-queue-stale-${group.synth_source}`}>
                            <code>{group.synth_source}</code>{' '}
                            <span className="synth-review-queue__stale-detail">
                                {staleText(c)}{c.labelled_by ? ` · labelled by ${c.labelled_by}` : ''}
                            </span>{' '}
                            {jobId ? (
                                <span className="synth-review-queue__stale-started">relabelling (job #{jobId})</span>
                            ) : pinFor(group) ? (
                                <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => void relabel(group)}
                                    data-testid={`synth-review-queue-relabel-${group.synth_source}`}
                                >
                                    Relabel {c.span_rows - c.current_rows} row{c.span_rows - c.current_rows === 1 ? '' : 's'} under current conventions with {pinFor(group)}
                                </button>
                            ) : (
                                <span className="synth-review-queue__stale-detail">
                                    relabel it from the span tools below (&quot;Relabel training rows&quot;).
                                </span>
                            )}
                            {!jobId && c.untracked_rows > 0 && (
                                <>
                                    {' '}
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-sm"
                                        onClick={() => void markCurrent(group)}
                                        data-testid={`synth-review-queue-mark-current-${group.synth_source}`}
                                    >
                                        Mark as current
                                    </button>
                                </>
                            )}
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}

interface AcceptedGroupCardProps {
    group: import('../../api/synthPlaybook').ReviewQueueGroup;
    projectId?: number;
    onChanged?: () => void;
}

/** Retire (accepted → rejected "retired") or restore a source group. */
async function setGroupRetired(projectId: number, source: string, retired: boolean): Promise<number> {
    const res = await api.post(`/projects/${projectId}/synthetic/review-queue/retire-group`, { source, retired });
    return Number(res.data?.rows_changed ?? 0);
}

function AcceptedGroupCard({ group, projectId, onChanged }: AcceptedGroupCardProps) {
    const retire = async () => {
        if (projectId == null) return;
        if (!window.confirm(
            `Retire ${group.synth_source}? Its ${group.count} accepted rows leave training (kept on disk as rejected, reason "retired"; you can restore them).`,
        )) return;
        try {
            const n = await setGroupRetired(projectId, group.synth_source, true);
            toast.success(`${group.synth_source}: ${n} rows retired. Re-split and retrain to see the effect.`, 5000);
            onChanged?.();
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not retire the group.', 5000);
        }
    };
    return (
        <details
            className="synth-review-queue__accepted-group"
            data-testid={`synth-review-queue-accepted-group-${group.synth_source}`}
        >
            <summary className="synth-review-queue__accepted-group-summary">
                <code>{group.synth_source || '(no source)'}</code>
                <span className="synth-review-queue__accepted-count">{group.count}</span>
                {group.conventions?.is_stale && (
                    <span
                        className="synth-review-queue__stale-badge"
                        data-testid={`synth-review-queue-stale-badge-${group.synth_source}`}
                    >
                        older conventions
                    </span>
                )}
            </summary>
            {projectId != null && (
                <button
                    type="button"
                    className="btn btn-ghost btn-sm synth-review-queue__retire"
                    onClick={() => void retire()}
                    data-testid={`synth-review-queue-retire-${group.synth_source}`}
                >
                    Retire group
                </button>
            )}
            {group.rows.length > 0 ? (
                <>
                    <ul className="synth-review-queue__accepted-row-list">
                        {group.rows.map((row) => (
                            <li
                                key={row.id}
                                className="synth-review-queue__accepted-row"
                                data-testid={`synth-review-queue-accepted-row-${row.id}`}
                            >
                                <span className="synth-review-queue__confidence">
                                    {Math.round((row.synth_confidence ?? 0) * 100)}%
                                </span>
                                <code>{row.preview}</code>
                            </li>
                        ))}
                    </ul>
                    {group.truncated && (
                        <p className="synth-review-queue__accepted-truncated">
                            Showing {group.rows.length} of {group.count} rows in this group. The rest are in your synthetic dataset and will enter training.
                        </p>
                    )}
                </>
            ) : (
                <p className="synth-review-queue__accepted-empty">(no preview rows)</p>
            )}
        </details>
    );
}


// ─────────────────────────────────────────────────────────────────────
// Arc 5 — soft-reject. Rejected rows stay on disk so the user can
// audit *why* a batch was rejected before purging. The backend groups
// by ``synth_source`` (mirrors pending/accepted); inside the section
// the rows are re-grouped by ``reject_reason`` so a noisy synth run
// becomes one card per cause-of-rejection with a per-reason "Purge"
// button. Per [rejected-rows-selectable] memory: never all-or-nothing.
// ─────────────────────────────────────────────────────────────────────

const NO_REASON_KEY = '(no reason)';

interface RejectedRow {
    id: number;
    synth_source: string;
    synth_confidence: number;
    preview: string;
    reason: string;
}

function _flattenRejectedRows(groups: ReviewQueueGroup[]): RejectedRow[] {
    const rows: RejectedRow[] = [];
    for (const group of groups) {
        for (const row of group.rows) {
            const reasonRaw = (row.payload as Record<string, unknown>)?.reject_reason;
            const reason =
                typeof reasonRaw === 'string' && reasonRaw.trim().length > 0
                    ? reasonRaw.trim()
                    : NO_REASON_KEY;
            rows.push({
                id: row.id,
                synth_source: group.synth_source,
                synth_confidence: row.synth_confidence ?? 0,
                preview: row.preview,
                reason,
            });
        }
    }
    return rows;
}

function _groupByReason(rows: RejectedRow[]): Map<string, RejectedRow[]> {
    const out = new Map<string, RejectedRow[]>();
    for (const row of rows) {
        const bucket = out.get(row.reason) ?? [];
        bucket.push(row);
        out.set(row.reason, bucket);
    }
    return out;
}


interface RejectedRowsSectionProps {
    projectId: number;
    groups: ReviewQueueGroup[];
    totalRejected: number;
    onPurgeCompleted: (
        result: import('../../api/synthPlaybook').PurgeRejectedResult,
    ) => Promise<void> | void;
    onPurgeError: (msg: string) => void;
}

function RejectedRowsSection({
    projectId,
    groups,
    totalRejected,
    onPurgeCompleted,
    onPurgeError,
}: RejectedRowsSectionProps) {
    const [purging, setPurging] = useState<string | null>(null);

    const rows = _flattenRejectedRows(groups);
    const byReason = _groupByReason(rows);
    const reasonKeys = Array.from(byReason.keys()).sort();

    const handlePurge = useCallback(
        async (scope: string, reasons: string[] | null) => {
            setPurging(scope);
            try {
                const result = await purgeRejectedSynthRows(projectId, { reasons });
                await onPurgeCompleted(result);
            } catch (err: any) {
                onPurgeError(
                    err?.response?.data?.detail || err?.message || 'Purge failed',
                );
            } finally {
                setPurging(null);
            }
        },
        [projectId, onPurgeCompleted, onPurgeError],
    );

    return (
        <details
            className="synth-review-queue__rejected"
            data-testid="synth-review-queue-rejected"
        >
            <summary className="synth-review-queue__rejected-summary">
                <span className="synth-review-queue__rejected-headline">
                    <strong>{totalRejected}</strong> rejected row{totalRejected === 1 ? '' : 's'} held on disk
                </span>
                <span className="synth-review-queue__rejected-hint">
                    ({reasonKeys.length} reason{reasonKeys.length === 1 ? '' : 's'} — click to expand)
                </span>
            </summary>
            <div className="synth-review-queue__rejected-toolbar">
                <button
                    type="button"
                    className="synth-review-queue__rejected-purge-all"
                    onClick={() => handlePurge('__all__', null)}
                    disabled={purging !== null || rows.length === 0}
                    data-testid="synth-review-queue-purge-all"
                >
                    {purging === '__all__' ? 'Purging…' : `Purge all ${rows.length}`}
                </button>
                <p className="synth-review-queue__rejected-help">
                    Soft-reject keeps rows on disk for audit. Purge removes them from <code>synthetic.jsonl</code> and decrements the dataset record count.
                </p>
            </div>
            <div className="synth-review-queue__rejected-groups">
                {reasonKeys.map((reason) => (
                    <RejectedReasonCard
                        key={reason}
                        reason={reason}
                        rows={byReason.get(reason) || []}
                        purging={purging === reason}
                        disablePurge={purging !== null}
                        onPurge={() =>
                            handlePurge(
                                reason,
                                reason === NO_REASON_KEY ? [''] : [reason],
                            )
                        }
                    />
                ))}
            </div>
        </details>
    );
}


interface RejectedReasonCardProps {
    reason: string;
    rows: RejectedRow[];
    purging: boolean;
    disablePurge: boolean;
    onPurge: () => void;
}

function RejectedReasonCard({
    reason,
    rows,
    purging,
    disablePurge,
    onPurge,
}: RejectedReasonCardProps) {
    const cardTestId = `synth-review-queue-rejected-reason-${reason}`;
    return (
        <details
            className="synth-review-queue__rejected-group"
            data-testid={cardTestId}
        >
            <summary className="synth-review-queue__rejected-group-summary">
                <span className="synth-review-queue__rejected-reason-badge">
                    {reason}
                </span>
                <span className="synth-review-queue__rejected-count">
                    {rows.length}
                </span>
                <button
                    type="button"
                    className="synth-review-queue__rejected-purge-reason"
                    onClick={(e) => {
                        // Don't toggle the <details> on button click.
                        e.preventDefault();
                        e.stopPropagation();
                        onPurge();
                    }}
                    disabled={disablePurge}
                    data-testid={`synth-review-queue-purge-reason-${reason}`}
                >
                    {purging ? 'Purging…' : 'Purge this reason'}
                </button>
            </summary>
            <ul className="synth-review-queue__rejected-row-list">
                {rows.map((row) => (
                    <li
                        key={row.id}
                        className="synth-review-queue__rejected-row"
                        data-testid={`synth-review-queue-rejected-row-${row.id}`}
                    >
                        <span className="synth-review-queue__confidence">
                            {Math.round((row.synth_confidence ?? 0) * 100)}%
                        </span>
                        <code className="synth-review-queue__rejected-source">
                            {row.synth_source || '(no source)'}
                        </code>
                        <code>{row.preview}</code>
                    </li>
                ))}
            </ul>
        </details>
    );
}
