/**
 * Panel tracking review work queues across synthetic, Gold Set, and annotation workflows.
 *
 * Quality-Lift phase 3 slice 3 — Active-learning card surfaces the most
 * recent training run's top-K most-uncertain unlabeled rows. The card
 * reads ``GET /api/projects/{id}/active-learning/latest`` and renders a
 * top-5 row table with text previews and a click-through to the
 * existing labeler with assign_strategy pre-set to ``active``.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    Check,
    CheckCircle2,
    ClipboardCheck,
    ExternalLink,
    GitBranch,
    ListChecks,
    RefreshCw,
    ShieldCheck,
    Sparkles,
    UserCheck,
    X,
} from 'lucide-react';

import api from '../../api/client';
import {
    getDataStudioReviewQueue,
    getPreparedVersionPreview,
} from '../../api/dataStudio';
import type {
    DataStudioReviewQueue,
    DataStudioReviewQueueGroup,
    DataStudioReviewQueueTriageItem,
    PreparedVersionPreview,
} from '../../api/dataStudio';
import { bulkUpdateSynthReviewBySource } from '../../api/synthPlaybook';
import './DataStudioReviewQueuePanel.css';

// Quality-Lift phase 3 slice 3 — Active-learning snapshot shape.
// Mirrors the backend response from /api/projects/{id}/active-learning/latest;
// kept local to this panel because the snapshot is a per-component
// concern (the Coach nudge already reads it server-side and emits a
// CoachSuggestion). If a second consumer lands later, lift to api/.
interface ActiveLearningTopKEntry {
    label_row_id: number;
    label_job_id: number;
    uncertainty_score: number;
    text_preview: string | null;
    labeled: boolean;
}

// Quality-Lift phase 4 slice 2 — Label-noise scan snapshot shape.
// Mirrors /api/projects/{id}/label-noise/latest; same locality reason
// as the active-learning interfaces above.
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

interface ActiveLearningSnapshot {
    scored_at: string;
    model_experiment_id: number;
    task_type: string | null;
    uncertainty_metric: string;
    pool_size_total: number;
    pool_size_scored: number;
    top_k: ActiveLearningTopKEntry[];
    skipped_reason: string | null;
}

interface ActiveLearningLatestResponse {
    project_id: number;
    snapshot: ActiveLearningSnapshot | null;
    experiment_id: number | null;
    experiment_name: string | null;
    top_k_size: number;
    labeled_count: number;
    unlabeled_count: number;
    staleness_ratio: number;
    is_stale: boolean;
    no_snapshot_reason: string | null;
    staleness_threshold: number;
    dominant_label_job_id: number | null;
}

interface DataStudioReviewQueuePanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const REVIEW_VERDICT_COPY: Record<DataStudioReviewQueue['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No queue',
        detail: 'Create synthetic, Gold Set, or annotation review work to start a queue.',
    },
    attention: {
        label: 'Needs review',
        detail: 'Review work is open across synthetic, Gold Set, or annotation workflows.',
    },
    ready: {
        label: 'Clear',
        detail: 'Review gates are clear and accepted or promoted examples are ready downstream.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForToken(value: string | undefined | null): string {
    if (!value) return 'Unknown';
    return value.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function priorityClass(priority: string): string {
    if (priority === 'high' || priority === 'medium' || priority === 'low') {
        return priority;
    }
    return 'low';
}

function TriageCard({
    item,
    onOpenTarget,
}: {
    item: DataStudioReviewQueueTriageItem;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <article className={`data-studio-review__triage-card data-studio-review__triage-card--${priorityClass(item.priority)}`}>
            <div className="data-studio-review__triage-head">
                <div>
                    <strong>{item.title}</strong>
                    <small>{labelForToken(item.priority)} priority</small>
                </div>
                <span>{formatNumber(item.count)}</span>
            </div>
            <p>{item.message}</p>
            {item.evidence.length > 0 ? (
                <ul>
                    {item.evidence.slice(0, 3).map((evidence) => (
                        <li key={evidence}>{evidence}</li>
                    ))}
                </ul>
            ) : null}
            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(item.target_tab)}>
                <ExternalLink size={15} aria-hidden="true" />
                {item.action_label}
            </button>
        </article>
    );
}

function formatNoSnapshotReason(reason: string | null | undefined): string {
    // Slice 1 stamps these reasons onto an empty snapshot so this card
    // can stay informative rather than disappearing. Mapping to
    // user-facing copy here so the backend stays terse + the panel
    // stays self-contained.
    switch (reason) {
        case 'no_completed_experiment_with_snapshot':
            return 'No completed training run has scored the unlabeled pool yet.';
        case 'unsupported_task_type':
            return 'Scoring is classification-only today; this project trains a different task type.';
        case 'empty_pool':
            return 'No unlabeled rows in the project pool to score.';
        case 'no_label_space_configured':
            return 'No classification label_job is configured for this project.';
        case 'checkpoint_path_missing':
            return 'The last training run did not save a checkpoint we could score against.';
        case 'scoring_failed':
            return 'Scoring failed during the last training run; check the experiment runtime for details.';
        case 'snapshot_empty':
            return 'The last run produced an empty snapshot.';
        default:
            return reason ? `Scoring skipped: ${labelForToken(reason)}.` : '';
    }
}

function ActiveLearningCard({
    snapshot,
    onOpenLabelQueue,
}: {
    snapshot: ActiveLearningLatestResponse;
    onOpenLabelQueue: (jobId: number | null) => void;
}) {
    // No snapshot AT ALL (fresh project, no completed run with
    // scoring) — render a quiet placeholder so the user understands
    // the surface is here but not yet populated.
    if (snapshot.snapshot === null || snapshot.top_k_size === 0) {
        const message = formatNoSnapshotReason(snapshot.no_snapshot_reason);
        if (!message) {
            return null;
        }
        return (
            <article className="data-studio-review__al-card data-studio-review__al-card--empty">
                <div className="data-studio-review__al-head">
                    <div>
                        <Sparkles size={16} aria-hidden="true" />
                        <strong>Active-learning queue</strong>
                    </div>
                </div>
                <p className="data-studio-review__al-empty">{message}</p>
            </article>
        );
    }

    const top5 = snapshot.snapshot.top_k.slice(0, 5);
    const samplePct = snapshot.snapshot.pool_size_total > 0
        ? Math.round(
            (snapshot.snapshot.pool_size_scored / snapshot.snapshot.pool_size_total) * 100,
        )
        : 0;
    const stalePct = Math.round(snapshot.staleness_ratio * 100);

    return (
        <article className={`data-studio-review__al-card ${snapshot.is_stale ? 'data-studio-review__al-card--stale' : ''}`}>
            <div className="data-studio-review__al-head">
                <div>
                    <Sparkles size={16} aria-hidden="true" />
                    <strong>Active-learning queue</strong>
                    <small>
                        scored by exp #{snapshot.experiment_id}
                        {snapshot.experiment_name ? ` — ${snapshot.experiment_name}` : ''}
                    </small>
                </div>
                <span className="data-studio-review__al-count">
                    {snapshot.unlabeled_count} <small>/ {snapshot.top_k_size} unlabeled</small>
                </span>
            </div>
            <p className="data-studio-review__al-meta">
                {snapshot.snapshot.uncertainty_metric} · sampled {snapshot.snapshot.pool_size_scored} of {snapshot.snapshot.pool_size_total} unlabeled rows ({samplePct}%)
                {snapshot.top_k_size > 0 ? ` · ${stalePct}% labeled` : ''}
            </p>
            {snapshot.is_stale && (
                <p className="data-studio-review__al-stale">
                    You've worked through most of this snapshot — consider re-training to score a fresh batch.
                </p>
            )}
            <table className="data-studio-review__al-table">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Score</th>
                        <th>Preview</th>
                    </tr>
                </thead>
                <tbody>
                    {top5.map((entry) => (
                        <tr
                            key={entry.label_row_id}
                            className={entry.labeled ? 'data-studio-review__al-row--labeled' : ''}
                        >
                            <td>
                                <code>#{entry.label_row_id}</code>
                                {entry.labeled && (
                                    <span className="data-studio-review__al-labeled-tag" title="already labeled">
                                        {' '}✓
                                    </span>
                                )}
                            </td>
                            <td>{entry.uncertainty_score.toFixed(3)}</td>
                            <td>
                                {entry.text_preview ? (
                                    <span className="data-studio-review__al-preview">{entry.text_preview}</span>
                                ) : (
                                    <em className="data-studio-review__al-preview--missing">(no text)</em>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onOpenLabelQueue(snapshot.dominant_label_job_id)}
            >
                <ExternalLink size={15} aria-hidden="true" />
                Open label queue
            </button>
        </article>
    );
}

// Quality-Lift phase 4 slice 2 — Label-noise card.
function formatLabelNoiseSkipReason(reason: string | null | undefined): string {
    // Slice 1 stamps these onto an empty result_payload so the card
    // stays informative rather than disappearing. Mapping to user-facing
    // copy here keeps the backend terse + the panel self-contained.
    switch (reason) {
        case 'no_classifier_checkpoint':
            return 'No completed classification training run to score against.';
        case 'empty_labeled_pool':
            return 'No labeled rows yet — scanning will activate once you label some.';
        case 'no_label_space_configured':
            return 'The classification label_job has no allowed_labels set.';
        case 'scoring_failed':
            return 'Scoring failed during the last scan; check the scan record for details.';
        default:
            return reason ? `Scan skipped: ${labelForToken(reason)}.` : '';
    }
}

function LabelNoiseCard({
    response,
    onOpenReview,
}: {
    response: LabelNoiseLatestResponse;
    onOpenReview: (scanId: number | null) => void;
}) {
    const scan = response.scan;

    // No scan run yet — quiet placeholder. The Coach nudge in the
    // cleaning stage will surface the "scan ready" suggestion when
    // labeled_count crosses 50; this card just confirms the surface
    // exists.
    if (scan === null) {
        return (
            <article className="data-studio-review__noise-card data-studio-review__noise-card--empty">
                <div className="data-studio-review__noise-head">
                    <div>
                        <AlertTriangle size={16} aria-hidden="true" />
                        <strong>Suspected mislabels</strong>
                    </div>
                </div>
                <p className="data-studio-review__noise-empty">
                    No label-noise scan has run yet. Train a classifier and
                    label some rows; the Cleaning Coach will nudge when it's
                    worth scanning.
                </p>
            </article>
        );
    }

    const payload = scan.result_payload;
    const suspectedCount = scan.suspected_count ?? 0;

    // Scan succeeded but produced no suspects — either the user's
    // labels are clean (the win condition) OR the scan was skipped
    // for a structural reason (no checkpoint, empty pool). The
    // payload's skipped_reason discriminates; we surface either case.
    if (suspectedCount === 0) {
        const skipReason = payload?.skipped_reason;
        const message = skipReason
            ? formatLabelNoiseSkipReason(skipReason)
            : 'No suspected mislabels in the latest scan — your labels look clean.';
        return (
            <article className="data-studio-review__noise-card data-studio-review__noise-card--clean">
                <div className="data-studio-review__noise-head">
                    <div>
                        <CheckCircle2 size={16} aria-hidden="true" />
                        <strong>Suspected mislabels</strong>
                        <small>scan #{scan.id}</small>
                    </div>
                </div>
                <p className="data-studio-review__noise-empty">{message}</p>
            </article>
        );
    }

    // Happy path — render top-5 suspects with given→predicted badges.
    const top5 = (payload?.top_k ?? []).slice(0, 5);
    const labelCount = scan.label_count_at_scan ?? 0;
    const confPct = Math.round((scan.confidence_threshold ?? 0.85) * 100);
    const floorPct = Math.round((scan.given_label_floor ?? 0.15) * 100);

    return (
        <article className="data-studio-review__noise-card">
            <div className="data-studio-review__noise-head">
                <div>
                    <AlertTriangle size={16} aria-hidden="true" />
                    <strong>Suspected mislabels</strong>
                    <small>
                        scan #{scan.id}
                        {scan.base_experiment_id ? ` · exp #${scan.base_experiment_id}` : ''}
                    </small>
                </div>
                <span className="data-studio-review__noise-count">
                    {suspectedCount} <small>/ {labelCount} labels</small>
                </span>
            </div>
            <p className="data-studio-review__noise-meta">
                confidence ≥ {confPct}% · given label ≤ {floorPct}%
            </p>
            <table className="data-studio-review__noise-table">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Given → Predicted</th>
                        <th>Confidence</th>
                        <th>Preview</th>
                    </tr>
                </thead>
                <tbody>
                    {top5.map((entry) => (
                        <tr key={entry.label_row_id}>
                            <td><code>#{entry.label_row_id}</code></td>
                            <td>
                                <span className="data-studio-review__noise-given-badge">
                                    {entry.given_label}
                                </span>
                                {' → '}
                                <span className="data-studio-review__noise-pred-badge">
                                    {entry.predicted_label}
                                </span>
                            </td>
                            <td>
                                <strong>{(entry.predicted_prob * 100).toFixed(0)}%</strong>
                                {' '}
                                <span className="data-studio-review__noise-meta-inline">
                                    (Δ {(entry.mislabel_score * 100).toFixed(0)}%)
                                </span>
                            </td>
                            <td>
                                {entry.text_preview ? (
                                    <span className="data-studio-review__noise-preview">
                                        {entry.text_preview}
                                    </span>
                                ) : (
                                    <em className="data-studio-review__noise-preview--missing">
                                        (no text)
                                    </em>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onOpenReview(scan.id)}
            >
                <ExternalLink size={15} aria-hidden="true" />
                Review suspected mislabels
            </button>
        </article>
    );
}

export default function DataStudioReviewQueuePanel({
    projectId,
    onOpenTarget,
}: DataStudioReviewQueuePanelProps) {
    const [queue, setQueue] = useState<DataStudioReviewQueue | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    // Slice 3 — Active-learning card state. Separate fetch + state from
    // the main review queue so an AL endpoint failure (network blip,
    // first-time project) doesn't pull down the rest of the panel.
    const [alSnapshot, setAlSnapshot] = useState<ActiveLearningLatestResponse | null>(null);
    // Quality-Lift phase 4 slice 2 — Label-noise card state. Same
    // separation rationale as active-learning above.
    const [labelNoise, setLabelNoise] = useState<LabelNoiseLatestResponse | null>(null);
    // Epic E — in-flight bulk-by-source action (keyed by group key) + the last
    // result flash, so the read-only panel becomes actionable without leaving it.
    const [bulkBusyKey, setBulkBusyKey] = useState<string | null>(null);
    const [bulkFlash, setBulkFlash] = useState<string | null>(null);
    // Epic E — "what version will include this?" preview. Best-effort.
    const [versionPreview, setVersionPreview] = useState<PreparedVersionPreview | null>(null);

    const loadVersionPreview = async () => {
        try {
            setVersionPreview(await getPreparedVersionPreview(projectId));
        } catch {
            setVersionPreview(null);
        }
    };

    const loadQueue = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioReviewQueue(projectId);
            setQueue(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio review queue.');
        } finally {
            setLoading(false);
        }
    };

    const loadActiveLearning = async () => {
        try {
            const resp = await api.get<ActiveLearningLatestResponse>(
                `/projects/${projectId}/active-learning/latest`,
            );
            setAlSnapshot(resp.data);
        } catch {
            // Best-effort — leave card hidden if the read fails. The
            // Coach nudge already independently reads this snapshot;
            // a panel-level read failure shouldn't take down the
            // whole review queue surface.
            setAlSnapshot(null);
        }
    };

    const loadLabelNoise = async () => {
        try {
            const resp = await api.get<LabelNoiseLatestResponse>(
                `/projects/${projectId}/label-noise/latest`,
            );
            setLabelNoise(resp.data);
        } catch {
            // Same best-effort rationale as the AL fetch.
            setLabelNoise(null);
        }
    };

    useEffect(() => {
        void loadQueue();
        void loadActiveLearning();
        void loadLabelNoise();
        void loadVersionPreview();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const handleBulkBySource = async (
        group: DataStudioReviewQueueGroup,
        action: 'accept' | 'reject',
    ) => {
        if (!group.synth_source || bulkBusyKey) {
            return;
        }
        setBulkBusyKey(group.key);
        setBulkFlash(null);
        try {
            const result = await bulkUpdateSynthReviewBySource(projectId, {
                source: group.synth_source,
                action,
            });
            const n = action === 'accept' ? result.accepted : result.rejected;
            setBulkFlash(
                `${action === 'accept' ? 'Accepted' : 'Rejected'} ${n} row${n === 1 ? '' : 's'} `
                + `from “${group.label}”. ${result.total_remaining_pending} still pending.`,
            );
            await loadQueue();
            await loadVersionPreview();
        } catch (err: any) {
            setBulkFlash(
                err?.response?.data?.detail || err?.message || `Failed to ${action} the group.`,
            );
        } finally {
            setBulkBusyKey(null);
        }
    };

    const handleOpenLabelQueue = (jobId: number | null) => {
        // Mirror CoachSuggestion's ``active-labeling-queue`` target —
        // pre-set localStorage so ProjectAnnotatePage honors the active
        // strategy on mount, then route to the dominant job.
        try {
            window.localStorage.setItem('slm_annotate_strategy', 'active');
        } catch {
            // Private mode / quota — user can still toggle the radio.
        }
        const url = jobId != null
            ? `/project/${projectId}/annotate/${jobId}`
            : `/project/${projectId}/annotate`;
        window.location.assign(url);
    };

    const handleOpenLabelNoiseReview = (scanId: number | null) => {
        // Mirror CoachSuggestion's ``label-noise-review`` target — slice
        // 3 mounts the review surface at /pipeline/cleaning#label-noise-review.
        // Forward scan_id as a query param so the surface can deep-link
        // to a specific scan (the user may have multiple in history).
        const qs = new URLSearchParams();
        if (scanId != null) {
            qs.set('scan_id', String(scanId));
        }
        const query = qs.toString() ? `?${qs.toString()}` : '';
        window.location.assign(
            `/project/${projectId}/pipeline/cleaning${query}#label-noise-review`,
        );
    };

    const topIssues = useMemo(
        () => queue?.issues.slice(0, 4) ?? [],
        [queue],
    );
    const topTriage = useMemo(
        () => queue?.triage.slice(0, 4) ?? [],
        [queue],
    );
    const sourceGroups = useMemo(
        () => queue?.groupings.by_source.slice(0, 6) ?? [],
        [queue],
    );
    const statusGroups = useMemo(
        () => (queue?.groupings.by_status ?? []).filter((group) => group.count > 0).slice(0, 7),
        [queue],
    );

    if (loading && !queue) {
        return (
            <section className="data-studio-review data-studio-review--loading">
                <span>Loading review queue...</span>
            </section>
        );
    }

    if (error && !queue) {
        return (
            <section className="data-studio-review data-studio-review--error">
                <div>
                    <h3>Review Queue</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadQueue()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!queue) {
        return null;
    }

    const verdict = REVIEW_VERDICT_COPY[queue.verdict];

    return (
        <section
            className={`data-studio-review data-studio-review--${queue.verdict}`}
            data-testid="data-studio-review-queue"
        >
            <div className="data-studio-review__header">
                <div>
                    <p className="data-studio-review__eyebrow">Review</p>
                    <h3>Review Queue</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-review__actions">
                    <span className={`data-studio-review__verdict data-studio-review__verdict--${queue.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-review__refresh"
                        onClick={() => void loadQueue()}
                        aria-label="Refresh Data Studio review queue"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-review__metrics" aria-label="Review queue metrics">
                <div className="data-studio-review__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Open review</span>
                    <strong>{formatNumber(queue.totals.open_review_items)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <ClipboardCheck size={18} aria-hidden="true" />
                    <span>Accepted synthetic</span>
                    <strong>{formatNumber(queue.totals.synthetic_accepted)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Trusted gold</span>
                    <strong>{formatNumber(queue.totals.gold_trusted_examples)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <UserCheck size={18} aria-hidden="true" />
                    <span>Promoted labels</span>
                    <strong>{formatNumber(queue.totals.annotation_promoted)}</strong>
                </div>
            </div>

            <div className="data-studio-review__signals">
                <span>{queue.domain.label}</span>
                <span>{formatPercent(queue.domain.confidence)} domain confidence</span>
                <span>{formatNumber(queue.totals.synthetic_pending)} pending synthetic</span>
                <span>{formatNumber(queue.totals.annotation_labeled_unpromoted)} labels to promote</span>
            </div>

            {versionPreview && (
                <div className="data-studio-review__version-preview" data-testid="version-preview">
                    <GitBranch size={16} aria-hidden="true" />
                    <span>
                        Rows you accept feed the next prepared dataset{' '}
                        <strong>v{versionPreview.next_version}</strong> — currently{' '}
                        <strong>{formatNumber(versionPreview.staged.synthetic_accepted)}</strong>{' '}
                        accepted synthetic
                        {versionPreview.staged.gold > 0 && (
                            <> {' + '}{formatNumber(versionPreview.staged.gold)} gold</>
                        )}
                        {versionPreview.staged.cleaned > 0 && (
                            <> {' + '}{formatNumber(versionPreview.staged.cleaned)} cleaned</>
                        )}
                        {versionPreview.staged.synthetic_pending > 0 && (
                            <>
                                {' '}({formatNumber(versionPreview.staged.synthetic_pending)} pending
                                rows excluded until reviewed)
                            </>
                        )}.
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => onOpenTarget('dataprep')}
                    >
                        Prepare →
                    </button>
                </div>
            )}

            <div className="data-studio-review__entrypoints">
                {queue.entry_points.slice(0, 4).map((entry) => (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        key={entry.target_tab}
                        onClick={() => onOpenTarget(entry.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {entry.label}
                    </button>
                ))}
            </div>

            <div className="data-studio-review__body">
                <div className="data-studio-review__triage">
                    <h4>What to review first</h4>
                    {topTriage.length > 0 ? (
                        <div className="data-studio-review__triage-list">
                            {topTriage.map((item) => (
                                <TriageCard item={item} key={item.id} onOpenTarget={onOpenTarget} />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-review__empty">
                            No review actions are waiting right now.
                        </p>
                    )}
                </div>

                <div className="data-studio-review__groups">
                    <h4>Power grouping</h4>
                    {bulkFlash && (
                        <p className="data-studio-review__bulk-flash" role="status" data-testid="bulk-flash">
                            {bulkFlash}
                        </p>
                    )}
                    {sourceGroups.length > 0 ? (
                        <div className="data-studio-review__source-list">
                            {sourceGroups.map((group) => {
                                const actionable = Boolean(
                                    group.synth_source
                                    && group.kind === 'synthetic'
                                    && group.status === 'pending',
                                );
                                const busy = bulkBusyKey === group.key;
                                return (
                                    <div
                                        className="data-studio-review__source"
                                        key={group.key}
                                    >
                                        <button
                                            type="button"
                                            className="data-studio-review__source-open"
                                            onClick={() => onOpenTarget(group.target_tab)}
                                        >
                                            <span>
                                                <strong>{group.label}</strong>
                                                <small>
                                                    {labelForToken(group.kind)}
                                                    {' · '}
                                                    {labelForToken(group.status)}
                                                </small>
                                            </span>
                                            <b>{formatNumber(group.count)}</b>
                                        </button>
                                        {actionable && (
                                            <div className="data-studio-review__source-actions">
                                                <button
                                                    type="button"
                                                    className="btn btn-secondary data-studio-review__bulk-accept"
                                                    disabled={busy || bulkBusyKey !== null}
                                                    onClick={() => handleBulkBySource(group, 'accept')}
                                                >
                                                    <Check size={13} aria-hidden="true" />
                                                    {busy ? 'Working…' : `Accept all (${group.count})`}
                                                </button>
                                                <button
                                                    type="button"
                                                    className="btn btn-ghost data-studio-review__bulk-reject"
                                                    disabled={busy || bulkBusyKey !== null}
                                                    onClick={() => handleBulkBySource(group, 'reject')}
                                                >
                                                    <X size={13} aria-hidden="true" />
                                                    Reject all
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <p className="data-studio-review__empty">
                            Source groupings appear after reviewable rows exist.
                        </p>
                    )}

                    <div className="data-studio-review__status-list">
                        {statusGroups.map((group) => (
                            <button
                                type="button"
                                className="data-studio-review__status"
                                key={group.status}
                                onClick={() => onOpenTarget(group.target_tab)}
                            >
                                <span>{group.label}</span>
                                <strong>{formatNumber(group.count)}</strong>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {alSnapshot ? (
                <ActiveLearningCard
                    snapshot={alSnapshot}
                    onOpenLabelQueue={handleOpenLabelQueue}
                />
            ) : null}

            {labelNoise ? (
                <LabelNoiseCard
                    response={labelNoise}
                    onOpenReview={handleOpenLabelNoiseReview}
                />
            ) : null}

            {topIssues.length > 0 ? (
                <ul className="data-studio-review__issues">
                    {topIssues.map((issue) => (
                        <li key={issue.id} className={`data-studio-review__issue data-studio-review__issue--${issue.severity}`}>
                            <span>{issueIcon(issue.severity)}</span>
                            <div>
                                <strong>{issue.title}</strong>
                                <small>{issue.message}</small>
                            </div>
                            <button type="button" className="btn btn-ghost" onClick={() => onOpenTarget(issue.target_tab)}>
                                {issue.action_label}
                            </button>
                        </li>
                    ))}
                </ul>
            ) : null}

            <details className="data-studio-review__details">
                <summary>Power details</summary>
                <pre>{compactJson(queue)}</pre>
            </details>
        </section>
    );
}
