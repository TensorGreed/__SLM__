/**
 * Quality-Lift phase 4 slice 3 — Label-noise review surface.
 *
 * Renders the suspected-mislabel queue from a LabelNoiseScan and lets
 * the user decide row-by-row: relabel as predicted, keep as-is, or
 * drop the row entirely (which returns it to phase 3's unlabeled
 * pool for a fresh annotation).
 *
 * Bulk affordance: rows are grouped by (given → predicted) transition.
 * The user selects all rows in a group and applies one decision —
 * useful when a class-name swap or annotation mistake created a whole
 * batch of identical mislabels.
 *
 * Deep-linked via the Coach nudges (slice 2). The CleaningPanel
 * mounts this component with an anchor (#label-noise-review) and
 * forwards the optional ``scan_id`` query param so the user can
 * review a historical scan rather than only the latest.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, RefreshCw, Trash2 } from 'lucide-react';

import api from '../../api/client';
import './LabelNoiseReviewPanel.css';

interface LabelNoiseSuspectEntry {
    label_row_id: number;
    label_job_id: number;
    given_label: string;
    predicted_label: string;
    predicted_prob: number;
    given_label_prob: number;
    mislabel_score: number;
    text_preview: string | null;
}

interface AppliedActionEntry {
    label_row_id: number;
    action: 'relabel' | 'keep' | 'drop';
    applied_label?: string;
    applied_at: string;
}

interface LabelNoiseScanPayload {
    scored_at: string;
    base_experiment_id: number | null;
    label_count_total: number;
    label_count_scored: number;
    suspected_count: number;
    confidence_threshold: number;
    given_label_floor: number;
    top_k: LabelNoiseSuspectEntry[];
    skipped_reason: string | null;
    applied_actions?: Record<string, AppliedActionEntry>;
}

interface LabelNoiseScanRecord {
    id: number;
    project_id: number;
    base_experiment_id: number | null;
    status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
    label_count_at_scan: number | null;
    suspected_count: number | null;
    confidence_threshold: number;
    given_label_floor: number;
    result_payload: LabelNoiseScanPayload | null;
    error: string | null;
    job_id: number | null;
    created_at: string | null;
    completed_at: string | null;
}

interface LabelNoiseLatestResponse {
    project_id: number;
    scan: LabelNoiseScanRecord | null;
    no_scan_reason: string | null;
}

interface ApplyResponse {
    scan_id: number;
    project_id: number;
    applied: number;
    relabeled: number;
    kept: number;
    dropped: number;
    skipped: Array<{ label_row_id: number; reason: string }>;
    applied_actions: Record<string, AppliedActionEntry>;
}

type RowDecision = 'relabel' | 'keep' | 'drop' | null;

interface LabelNoiseReviewPanelProps {
    projectId: number;
    /** Optional scan id from the URL query param. When absent, falls
     *  back to the latest endpoint. */
    scanId?: number | null;
}

function groupKey(entry: LabelNoiseSuspectEntry): string {
    return `${entry.given_label}::${entry.predicted_label}`;
}

function formatPct(value: number): string {
    return `${Math.round(value * 100)}%`;
}

export default function LabelNoiseReviewPanel({
    projectId,
    scanId,
}: LabelNoiseReviewPanelProps) {
    const [scan, setScan] = useState<LabelNoiseScanRecord | null>(null);
    const [noScanReason, setNoScanReason] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [decisions, setDecisions] = useState<Record<number, RowDecision>>({});
    const [submitting, setSubmitting] = useState(false);
    const [lastApply, setLastApply] = useState<ApplyResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const loadScan = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            if (scanId != null) {
                // Specific scan deep-link.
                const resp = await api.get<LabelNoiseScanRecord>(
                    `/projects/${projectId}/label-noise/scans/${scanId}`,
                );
                setScan(resp.data);
                setNoScanReason(null);
            } else {
                const resp = await api.get<LabelNoiseLatestResponse>(
                    `/projects/${projectId}/label-noise/latest`,
                );
                setScan(resp.data.scan);
                setNoScanReason(resp.data.no_scan_reason);
            }
        } catch (err: any) {
            setError(
                err?.response?.data?.detail
                || err?.message
                || 'Failed to load scan.',
            );
        } finally {
            setLoading(false);
        }
    }, [projectId, scanId]);

    useEffect(() => {
        void loadScan();
    }, [loadScan]);

    const topK = scan?.result_payload?.top_k ?? [];
    const appliedActions = scan?.result_payload?.applied_actions ?? {};

    // Group by (given → predicted) for the bulk affordance. Order
    // groups by size descending — bigger transitions are usually
    // the most common annotation mistakes and most impactful to fix
    // in one click.
    const groups = useMemo(() => {
        const acc = new Map<string, LabelNoiseSuspectEntry[]>();
        for (const entry of topK) {
            const key = groupKey(entry);
            const bucket = acc.get(key) ?? [];
            bucket.push(entry);
            acc.set(key, bucket);
        }
        return Array.from(acc.entries())
            .map(([key, rows]) => ({ key, rows }))
            .sort((a, b) => b.rows.length - a.rows.length);
    }, [topK]);

    const decisionsCount = Object.values(decisions).filter(Boolean).length;

    const setDecision = (rowId: number, decision: RowDecision) => {
        setDecisions((prev) => {
            const next = { ...prev };
            if (decision === null || prev[rowId] === decision) {
                // Toggle off when clicking the same button — useful
                // when the user mis-clicks and wants to back out.
                delete next[rowId];
            } else {
                next[rowId] = decision;
            }
            return next;
        });
    };

    const applyGroup = (
        rows: LabelNoiseSuspectEntry[],
        decision: 'relabel' | 'keep' | 'drop',
    ) => {
        setDecisions((prev) => {
            const next = { ...prev };
            for (const row of rows) {
                next[row.label_row_id] = decision;
            }
            return next;
        });
    };

    const handleApply = async () => {
        if (!scan) return;
        const actions = Object.entries(decisions)
            .filter(([, decision]) => decision !== null)
            .map(([rowId, decision]) => ({
                label_row_id: Number(rowId),
                action: decision as 'relabel' | 'keep' | 'drop',
            }));
        if (actions.length === 0) return;

        setSubmitting(true);
        setError(null);
        try {
            const resp = await api.post<ApplyResponse>(
                `/projects/${projectId}/label-noise/scans/${scan.id}/apply`,
                { actions },
            );
            setLastApply(resp.data);
            // Re-fetch so applied_actions in the rendered scan stays
            // in sync (the user might apply twice).
            await loadScan();
            // Clear the decision draft now that the server confirmed.
            setDecisions({});
        } catch (err: any) {
            setError(
                err?.response?.data?.detail
                || err?.message
                || 'Failed to apply actions.',
            );
        } finally {
            setSubmitting(false);
        }
    };

    if (loading) {
        return (
            <section className="label-noise-review label-noise-review--loading">
                <h3>Label-noise review</h3>
                <p>Loading scan…</p>
            </section>
        );
    }

    if (error && !scan) {
        return (
            <section className="label-noise-review label-noise-review--error">
                <h3>Label-noise review</h3>
                <p>{error}</p>
                <button type="button" className="btn btn-secondary" onClick={() => void loadScan()}>
                    <RefreshCw size={14} aria-hidden="true" /> Retry
                </button>
            </section>
        );
    }

    if (scan === null) {
        return (
            <section className="label-noise-review label-noise-review--empty">
                <h3>Label-noise review</h3>
                <p>
                    {noScanReason === 'no_succeeded_scan_yet'
                        ? 'No label-noise scan has run yet. The cleaning Coach will nudge when one is worth running.'
                        : 'No scan available.'}
                </p>
            </section>
        );
    }

    if (topK.length === 0) {
        const skip = scan.result_payload?.skipped_reason;
        return (
            <section className="label-noise-review label-noise-review--clean">
                <h3>Label-noise review</h3>
                <p>
                    {skip
                        ? `Scan skipped: ${skip}.`
                        : 'No suspected mislabels in this scan — your labels look clean.'}
                </p>
            </section>
        );
    }

    return (
        <section className="label-noise-review" data-testid="label-noise-review">
            <div className="label-noise-review__header">
                <div>
                    <h3>Label-noise review</h3>
                    <p className="label-noise-review__meta">
                        scan #{scan.id}
                        {scan.base_experiment_id != null && ` · exp #${scan.base_experiment_id}`}
                        {' · '}
                        confidence ≥ {formatPct(scan.confidence_threshold)}
                        {' · '}
                        given label ≤ {formatPct(scan.given_label_floor)}
                    </p>
                </div>
                <button
                    type="button"
                    className="btn btn-ghost label-noise-review__refresh"
                    onClick={() => void loadScan()}
                    aria-label="Refresh scan"
                >
                    <RefreshCw size={14} aria-hidden="true" />
                </button>
            </div>

            {lastApply && (
                <div className="label-noise-review__apply-result">
                    Applied: <strong>{lastApply.relabeled}</strong> relabeled
                    {' · '}<strong>{lastApply.kept}</strong> kept
                    {' · '}<strong>{lastApply.dropped}</strong> dropped
                    {lastApply.skipped.length > 0 && (
                        <> · <strong>{lastApply.skipped.length}</strong> skipped</>
                    )}
                </div>
            )}
            {error && <div className="label-noise-review__error">{error}</div>}

            <div className="label-noise-review__groups">
                {groups.map(({ key, rows }) => {
                    const [given, predicted] = key.split('::');
                    return (
                        <article key={key} className="label-noise-review__group">
                            <header className="label-noise-review__group-head">
                                <div>
                                    <span className="label-noise-review__given-badge">{given}</span>
                                    {' → '}
                                    <span className="label-noise-review__pred-badge">{predicted}</span>
                                    <span className="label-noise-review__group-count">
                                        {' '}({rows.length} row{rows.length !== 1 ? 's' : ''})
                                    </span>
                                </div>
                                <div className="label-noise-review__group-actions">
                                    <button
                                        type="button"
                                        className="btn btn-ghost"
                                        onClick={() => applyGroup(rows, 'relabel')}
                                        aria-label={`Relabel all ${rows.length} rows as ${predicted}`}
                                    >
                                        <CheckCircle2 size={13} aria-hidden="true" />
                                        {' '}Relabel all
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn-ghost"
                                        onClick={() => applyGroup(rows, 'drop')}
                                        aria-label={`Drop all ${rows.length} rows`}
                                    >
                                        <Trash2 size={13} aria-hidden="true" />
                                        {' '}Drop all
                                    </button>
                                </div>
                            </header>
                            <ul className="label-noise-review__row-list">
                                {rows.map((row) => {
                                    const decision = decisions[row.label_row_id] ?? null;
                                    const prior = appliedActions[String(row.label_row_id)];
                                    return (
                                        <li
                                            key={row.label_row_id}
                                            className={`label-noise-review__row ${decision ? `label-noise-review__row--${decision}` : ''}`}
                                        >
                                            <div className="label-noise-review__row-id">
                                                <code>#{row.label_row_id}</code>
                                                {prior && (
                                                    <span
                                                        className={`label-noise-review__prior label-noise-review__prior--${prior.action}`}
                                                        title={`Previously: ${prior.action}${prior.applied_label ? ` → ${prior.applied_label}` : ''}`}
                                                    >
                                                        {' '}prev: {prior.action}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="label-noise-review__row-text">
                                                {row.text_preview || <em>(no text)</em>}
                                            </div>
                                            <div className="label-noise-review__row-conf">
                                                <strong>{formatPct(row.predicted_prob)}</strong>
                                                {' '}
                                                <span className="label-noise-review__row-conf-meta">
                                                    (Δ {formatPct(row.mislabel_score)})
                                                </span>
                                            </div>
                                            <div className="label-noise-review__row-actions">
                                                <button
                                                    type="button"
                                                    className={`btn btn-sm ${decision === 'relabel' ? 'btn-primary' : 'btn-ghost'}`}
                                                    onClick={() => setDecision(row.label_row_id, 'relabel')}
                                                    aria-pressed={decision === 'relabel'}
                                                >
                                                    Relabel
                                                </button>
                                                <button
                                                    type="button"
                                                    className={`btn btn-sm ${decision === 'keep' ? 'btn-primary' : 'btn-ghost'}`}
                                                    onClick={() => setDecision(row.label_row_id, 'keep')}
                                                    aria-pressed={decision === 'keep'}
                                                >
                                                    Keep
                                                </button>
                                                <button
                                                    type="button"
                                                    className={`btn btn-sm ${decision === 'drop' ? 'btn-primary' : 'btn-ghost'}`}
                                                    onClick={() => setDecision(row.label_row_id, 'drop')}
                                                    aria-pressed={decision === 'drop'}
                                                >
                                                    Drop
                                                </button>
                                            </div>
                                        </li>
                                    );
                                })}
                            </ul>
                        </article>
                    );
                })}
            </div>

            <div className="label-noise-review__apply-bar">
                <span>
                    {decisionsCount === 0
                        ? 'Select decisions per row or per group, then apply.'
                        : `${decisionsCount} decision${decisionsCount !== 1 ? 's' : ''} pending`}
                </span>
                <button
                    type="button"
                    className="btn btn-primary"
                    disabled={decisionsCount === 0 || submitting}
                    onClick={() => void handleApply()}
                >
                    {submitting ? 'Applying…' : 'Apply decisions'}
                </button>
            </div>

            <footer className="label-noise-review__footer">
                <AlertTriangle size={13} aria-hidden="true" />
                {' '}Dropped rows return to the unlabeled pool — phase 3's active-learning
                queue surfaces them for fresh labeling on the next training run.
            </footer>
        </section>
    );
}
