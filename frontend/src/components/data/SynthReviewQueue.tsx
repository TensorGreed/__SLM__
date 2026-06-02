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

import type {
    ReviewQueueGroup,
    ReviewQueueResponse,
} from '../../api/synthPlaybook';
import {
    bulkUpdateSynthReviewQueue,
    listSynthReviewQueue,
    purgeRejectedSynthRows,
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

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const payload = await listSynthReviewQueue(projectId);
            setData(payload);
            setSelected(new Set());
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load review queue');
        } finally {
            setLoading(false);
        }
    }, [projectId]);

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
                {data.accepted_groups.length > 0 && (
                    <AcceptedRowsSection
                        groups={data.accepted_groups}
                        totalAccepted={data.total_accepted}
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
                <h3 className="synth-review-queue__title">Synth review queue</h3>
                <p className="synth-review-queue__subtitle">
                    {data.total_pending} row{data.total_pending === 1 ? '' : 's'} awaiting review,
                    grouped by source. Accept to add to training; reject to discard.
                    {data.total_accepted > 0 && (
                        <> {' · '}<strong>{data.total_accepted}</strong> already accepted (see below).</>
                    )}
                </p>
            </header>
            <TotalCountStrip data={data} />

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
                                <span className="synth-review-queue__group-count">{group.count}</span>
                            </label>
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
}

/**
 * Collapsible "Accepted — queued for training" section that surfaces
 * what's already passed review. Each source-group is itself
 * expandable: click to reveal up to 25 sample rows (the backend
 * caps each group at 25 to avoid blowing up the payload for legacy
 * buckets with thousands of rows). Truncated groups show a
 * "showing N of total" footer.
 */
function AcceptedRowsSection({ groups, totalAccepted }: AcceptedRowsSectionProps) {
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
                    <AcceptedGroupCard key={group.synth_source} group={group} />
                ))}
            </div>
            <p className="synth-review-queue__accepted-footnote">
                Accepted rows enter the training corpus on the next Dataset Prep + Training run.
            </p>
        </details>
    );
}


interface AcceptedGroupCardProps {
    group: import('../../api/synthPlaybook').ReviewQueueGroup;
}

function AcceptedGroupCard({ group }: AcceptedGroupCardProps) {
    return (
        <details
            className="synth-review-queue__accepted-group"
            data-testid={`synth-review-queue-accepted-group-${group.synth_source}`}
        >
            <summary className="synth-review-queue__accepted-group-summary">
                <code>{group.synth_source || '(no source)'}</code>
                <span className="synth-review-queue__accepted-count">{group.count}</span>
            </summary>
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
