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
} from '../../api/synthPlaybook';
import './SynthReviewQueue.css';

interface Props {
    projectId: number;
}

type SelectedSet = Set<number>;

export default function SynthReviewQueue({ projectId }: Props) {
    const [data, setData] = useState<ReviewQueueResponse | null>(null);
    const [selected, setSelected] = useState<SelectedSet>(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [flash, setFlash] = useState<string | null>(null);

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

    if (!data || (data.total_pending === 0 && data.total_accepted === 0)) {
        return (
            <section className="synth-review-queue synth-review-queue--empty" data-testid="synth-review-queue-empty">
                <p>No synth rows pending review.</p>
            </section>
        );
    }

    // If only accepted rows exist (queue is empty), render a compact
    // "what's queued for training" summary so approved rows are
    // visible somewhere in the UI.
    if (data.total_pending === 0) {
        return (
            <section className="synth-review-queue" data-testid="synth-review-queue">
                <header className="synth-review-queue__head">
                    <h3 className="synth-review-queue__title">Synth review queue</h3>
                    <p className="synth-review-queue__subtitle">
                        No rows pending review. <strong>{data.total_accepted}</strong> row{data.total_accepted === 1 ? '' : 's'} accepted and queued for the next training run.
                    </p>
                </header>
                <AcceptedRowsSection groups={data.accepted_groups} totalAccepted={data.total_accepted} />
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
        </section>
    );
}


interface AcceptedRowsSectionProps {
    groups: import('../../api/synthPlaybook').ReviewQueueGroup[];
    totalAccepted: number;
}

/**
 * Collapsible "Accepted — queued for training" section that surfaces
 * what's already passed review. Answers the user's "where do
 * approved synth rows show up?" question.
 */
function AcceptedRowsSection({ groups, totalAccepted }: AcceptedRowsSectionProps) {
    const [expanded, setExpanded] = useState(false);
    return (
        <details
            className="synth-review-queue__accepted"
            open={expanded}
            onToggle={(e) => setExpanded((e.target as HTMLDetailsElement).open)}
            data-testid="synth-review-queue-accepted"
        >
            <summary className="synth-review-queue__accepted-summary">
                <span className="synth-review-queue__accepted-headline">
                    <strong>{totalAccepted}</strong> accepted row{totalAccepted === 1 ? '' : 's'} queued for training
                </span>
                <span className="synth-review-queue__accepted-hint">
                    ({groups.length} source{groups.length === 1 ? '' : 's'})
                </span>
            </summary>
            <ul className="synth-review-queue__accepted-groups">
                {groups.map((group) => (
                    <li
                        key={group.synth_source}
                        className="synth-review-queue__accepted-group"
                        data-testid={`synth-review-queue-accepted-group-${group.synth_source}`}
                    >
                        <code>{group.synth_source || '(no source)'}</code>
                        <span className="synth-review-queue__accepted-count">{group.count}</span>
                    </li>
                ))}
            </ul>
            <p className="synth-review-queue__accepted-footnote">
                Accepted rows enter the training corpus on the next Dataset Prep + Training run.
            </p>
        </details>
    );
}
