/**
 * NotificationBell — Hardening Phase H1.
 *
 * Top-bar bell icon + dropdown showing in-flight + recently-
 * completed background jobs (synth playbook runs, RAG clones,
 * future training/eval kicks). Azure-portal pattern: user starts a
 * long-running operation, leaves the page, and gets notified via
 * the bell when it completes.
 *
 * Polling is driven by ``useJobsStore`` — only ticks when there's
 * something to watch or the dropdown is open. Closing/clicking
 * outside the dropdown stops the poll if no job is in-flight.
 */

import { useEffect, useRef, useState } from 'react';

import { cancelJob, dismissJob, type Job } from '../../api/jobs';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import TrainingKillSwitch from './TrainingKillSwitch';
import TrainingLossSparkline from './TrainingLossSparkline';
import './NotificationBell.css';


/**
 * Navigate via window.location.assign — matches the pattern in
 * RerouteRecommendationPanel and keeps the bell mountable from
 * any context (no Router requirement).
 */
function navigateTo(url: string): void {
    window.location.assign(url);
}


function jobDeepLink(job: Job): string | null {
    // Map each job kind to where clicking "Open" should land.
    if (job.kind === 'reroute_to_rag' && job.status === 'succeeded') {
        const newId = job.result?.new_project_id;
        if (typeof newId === 'number') return `/project/${newId}`;
    }
    if (job.kind === 'synth_playbook' && job.project_id) {
        return `/project/${job.project_id}/pipeline/synthetic#synth-review-queue`;
    }
    if (job.kind === 'synth_augment_from_cluster' && job.project_id) {
        return `/project/${job.project_id}/pipeline/synthetic#synth-review-queue`;
    }
    // Legacy synth (QA / spans / conversations) all land on the
    // Synthetic tab where the inline review UI lives — the user
    // reviews the generated rows + saves from there.
    if (job.kind.startsWith('synth_legacy') && job.project_id) {
        return `/project/${job.project_id}/pipeline/synthetic`;
    }
    if (job.kind === 'training_start' && job.project_id) {
        return `/project/${job.project_id}/training-config`;
    }
    if (job.project_id) {
        return `/project/${job.project_id}`;
    }
    return null;
}


function jobStatusLabel(job: Job): string {
    switch (job.status) {
        case 'queued': return 'Queued';
        case 'running': return job.progress_message || 'Running…';
        case 'succeeded': return 'Done';
        case 'failed': return 'Failed';
        case 'cancelled': return 'Cancelled';
        default: return job.status;
    }
}


/**
 * Per-kind one-line outcome summary for a terminal job. Pulls
 * fields out of ``job.result`` and formats them so the bell row
 * tells the user WHAT happened, not just THAT it finished.
 * Returns null when the job has no useful summary (caller hides
 * the line entirely).
 */
function jobOutcomeSummary(job: Job): string | null {
    if (job.status !== 'succeeded') return null;
    const r = (job.result || {}) as Record<string, unknown>;

    if (job.kind === 'synth_playbook' || job.kind === 'synth_augment_from_cluster') {
        const rows = typeof r.rows_generated === 'number' ? r.rows_generated : null;
        const backend = typeof r.backend_used === 'string' ? r.backend_used : null;
        const elapsed = typeof r.elapsed_sec === 'number' ? r.elapsed_sec : null;
        const clusterId =
            typeof r.cluster_id === 'string' && r.cluster_id.length > 0
                ? r.cluster_id
                : null;
        const parts: string[] = [];
        if (rows !== null) parts.push(`${rows} row${rows === 1 ? '' : 's'} generated`);
        if (clusterId) parts.push(`cluster ${clusterId.slice(0, 16)}`);
        if (backend) parts.push(`via ${backend}`);
        if (elapsed !== null) parts.push(`${elapsed.toFixed(1)}s`);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind.startsWith('synth_legacy')) {
        const rowsSaved = typeof r.rows_saved === 'number' ? r.rows_saved : null;
        const rowsGenerated = typeof r.rows_generated === 'number' ? r.rows_generated : null;
        const batches = typeof r.batches_done === 'number' ? r.batches_done : null;
        const total = typeof r.batches_total === 'number' ? r.batches_total : null;
        const parts: string[] = [];
        // Prefer rows_saved (the persisted count) — that's the
        // number the user actually cares about. Fall back to
        // rows_generated for older jobs from before auto-save shipped.
        if (rowsSaved !== null) {
            parts.push(`${rowsSaved} row${rowsSaved === 1 ? '' : 's'} saved to dataset`);
        } else if (rowsGenerated !== null) {
            parts.push(`${rowsGenerated} row${rowsGenerated === 1 ? '' : 's'} generated`);
        }
        if (batches !== null && total !== null) parts.push(`${batches}/${total} batches`);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'reroute_to_rag') {
        const newId = typeof r.new_project_id === 'number' ? r.new_project_id : null;
        const newName = typeof r.new_project_name === 'string' ? r.new_project_name : null;
        if (newId !== null) {
            return newName
                ? `Created "${newName}" (project #${newId})`
                : `Created project #${newId}`;
        }
        return null;
    }

    if (job.kind === 'training_start') {
        const loss = typeof r.final_train_loss === 'number' ? r.final_train_loss : null;
        const steps = typeof r.total_steps === 'number' ? r.total_steps : null;
        const terminal = typeof r.terminal_status === 'string' ? r.terminal_status : null;
        const parts: string[] = [];
        if (terminal && terminal !== 'completed') {
            parts.push(terminal);
        }
        if (loss !== null) parts.push(`final loss ${loss.toFixed(4)}`);
        if (steps !== null) parts.push(`${steps} steps`);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'auto_rag_comparison') {
        const offF1 = typeof r.off_mean_f1 === 'number' ? r.off_mean_f1 : null;
        const onF1 = typeof r.on_mean_f1 === 'number' ? r.on_mean_f1 : null;
        const lift = typeof r.relative_lift_pct === 'number' ? r.relative_lift_pct : null;
        const nVal = typeof r.n_val_rows === 'number' ? r.n_val_rows : null;
        const parts: string[] = [];
        if (offF1 !== null && onF1 !== null) {
            parts.push(`off F1 ${offF1.toFixed(2)} → on F1 ${onF1.toFixed(2)}`);
        } else if (onF1 !== null) {
            parts.push(`on F1 ${onF1.toFixed(2)}`);
        }
        if (lift !== null) {
            const sign = lift >= 0 ? '+' : '';
            parts.push(`${sign}${lift.toFixed(2)}% lift`);
        }
        if (nVal !== null) {
            parts.push(`${nVal} val row${nVal === 1 ? '' : 's'}`);
        }
        return parts.length ? parts.join(' · ') : null;
    }

    return null;
}


export default function NotificationBell() {
    const { jobs, isPolling, bellOpen, setBellOpen, refreshAfterLocalChange } =
        useJobsStore();
    const [, setTick] = useState(0);  // forces re-render on dismissals
    const wrapperRef = useRef<HTMLDivElement>(null);

    // Mount → start polling so the bell starts fresh when the app
    // boots and any jobs already in-flight show up.
    useEffect(() => {
        void useJobsStore.getState().fetchOnce();
        return () => {
            // Component unmount: leave polling alone — store decides.
        };
    }, []);

    // Close dropdown on outside-click.
    useEffect(() => {
        if (!bellOpen) return;
        const onDocClick = (ev: MouseEvent) => {
            const target = ev.target as Node | null;
            if (wrapperRef.current && target && !wrapperRef.current.contains(target)) {
                setBellOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, [bellOpen, setBellOpen]);

    const inFlight = jobs.filter(
        (j) => j.status === 'queued' || j.status === 'running',
    );
    const recentlyDone = jobs.filter(
        (j) => j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled',
    );

    const badgeCount = inFlight.length;
    // A failed job that hasn't been dismissed shows as a red badge
    // even when no jobs are currently running — same urgency
    // signal as an Azure-portal alert.
    const hasUnacknowledgedFailure = recentlyDone.some((j) => j.status === 'failed');

    const handleDismiss = async (job: Job) => {
        try {
            await dismissJob(job.id);
            await refreshAfterLocalChange();
            setTick((t) => t + 1);
        } catch (err) {
            const msg =
                (err as { response?: { data?: { detail?: string } } })?.response
                    ?.data?.detail || 'Dismiss failed';
            toast.error(msg);
        }
    };

    const handleCancel = async (job: Job) => {
        try {
            await cancelJob(job.id);
            toast.info(`Cancelled "${job.title}"`, 3000);
            await refreshAfterLocalChange();
            setTick((t) => t + 1);
        } catch (err) {
            const msg =
                (err as { response?: { data?: { detail?: string } } })?.response
                    ?.data?.detail || 'Cancel failed';
            toast.error(msg);
        }
    };

    const handleOpen = (job: Job) => {
        const url = jobDeepLink(job);
        if (url) {
            setBellOpen(false);
            navigateTo(url);
        }
    };

    return (
        <div
            ref={wrapperRef}
            className="notif-bell"
            data-testid="notification-bell"
        >
            <button
                type="button"
                className={
                    'notif-bell__btn'
                    + (badgeCount > 0 ? ' has-active' : '')
                    + (hasUnacknowledgedFailure ? ' has-failure' : '')
                }
                onClick={() => setBellOpen(!bellOpen)}
                aria-label={
                    badgeCount > 0
                        ? `${badgeCount} background jobs running`
                        : 'Notifications'
                }
                data-testid="notification-bell-button"
            >
                {/* Inline SVG bell so we don't need an icon library. */}
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
                {badgeCount > 0 && (
                    <span
                        className="notif-bell__badge"
                        data-testid="notification-bell-badge"
                    >
                        {badgeCount}
                    </span>
                )}
                {badgeCount === 0 && hasUnacknowledgedFailure && (
                    <span
                        className="notif-bell__badge notif-bell__badge--failure"
                        data-testid="notification-bell-badge-failure"
                    >
                        !
                    </span>
                )}
            </button>

            {bellOpen && (
                <div
                    className="notif-bell__panel"
                    data-testid="notification-bell-panel"
                    role="dialog"
                    aria-label="Background jobs"
                >
                    <div className="notif-bell__head">
                        <h3>Background jobs</h3>
                        <span className="notif-bell__head-meta">
                            {isPolling ? 'live' : 'idle'}
                        </span>
                    </div>

                    {jobs.length === 0 && (
                        <p
                            className="notif-bell__empty"
                            data-testid="notification-bell-empty"
                        >
                            No background jobs right now. Long-running
                            operations like synth runs and RAG clones will
                            show up here.
                        </p>
                    )}

                    {inFlight.length > 0 && (
                        <section className="notif-bell__group">
                            <div className="notif-bell__group-head">
                                In progress ({inFlight.length})
                            </div>
                            {inFlight.map((job) => (
                                <JobRow
                                    key={job.id}
                                    job={job}
                                    onOpen={handleOpen}
                                    onDismiss={handleDismiss}
                                    onCancel={handleCancel}
                                />
                            ))}
                        </section>
                    )}

                    {recentlyDone.length > 0 && (
                        <section className="notif-bell__group">
                            <div className="notif-bell__group-head">
                                Recently completed
                            </div>
                            {recentlyDone.map((job) => (
                                <JobRow
                                    key={job.id}
                                    job={job}
                                    onOpen={handleOpen}
                                    onDismiss={handleDismiss}
                                />
                            ))}
                        </section>
                    )}
                </div>
            )}
        </div>
    );
}


interface JobRowProps {
    job: Job;
    onOpen: (job: Job) => void;
    onDismiss: (job: Job) => void;
    /** Optional — only passed for in-flight jobs by the parent. */
    onCancel?: (job: Job) => void;
}


function JobRow({ job, onOpen, onDismiss, onCancel }: JobRowProps) {
    const isInFlight = job.status === 'queued' || job.status === 'running';
    const link = jobDeepLink(job);
    return (
        <div
            className={`notif-bell__row notif-bell__row--${job.status}`}
            data-testid={`notification-bell-row-${job.id}`}
        >
            <div className="notif-bell__row-main">
                <div className="notif-bell__row-title">{job.title}</div>
                <div className="notif-bell__row-status">
                    {jobStatusLabel(job)}
                </div>
                {(() => {
                    const summary = jobOutcomeSummary(job);
                    return summary ? (
                        <div
                            className="notif-bell__row-summary"
                            data-testid={`notification-bell-row-${job.id}-summary`}
                        >
                            {summary}
                        </div>
                    ) : null;
                })()}
                {job.status === 'failed' && job.error && (
                    <div
                        className="notif-bell__row-error"
                        data-testid={`notification-bell-row-${job.id}-error`}
                    >
                        {job.error}
                    </div>
                )}
                {isInFlight && typeof job.progress === 'number' && (() => {
                    // role="progressbar" with dynamic aria-valuenow trips
                    // a strict-mode ARIA validator that wants literal
                    // string values; aria-label conveys the same info
                    // for screen readers and keeps the lint clean.
                    // We snap the fill width to nearest-10% (defined as
                    // CSS classes) so we don't need inline styles —
                    // strict CSS lint also wants those off.
                    const pct = Math.max(0, Math.min(100, Math.round(job.progress * 100)));
                    const bucket = Math.round(pct / 10) * 10;
                    return (
                        <div
                            className="notif-bell__progress"
                            aria-label={`${pct}% complete`}
                        >
                            <div
                                className={`notif-bell__progress-fill notif-bell__progress-fill--w-${bucket}`}
                            />
                        </div>
                    );
                })()}
                {isInFlight
                    && job.kind === 'training_start'
                    && Array.isArray(job.metrics_recent) && (
                    <div
                        className="notif-bell__loss-sparkline"
                        data-testid={`notification-bell-row-${job.id}-sparkline`}
                    >
                        <TrainingLossSparkline points={job.metrics_recent} />
                    </div>
                )}
                {isInFlight && job.kind === 'training_start' && (
                    <TrainingKillSwitch job={job} />
                )}
            </div>
            <div className="notif-bell__row-actions">
                {link && (
                    <button
                        type="button"
                        className="notif-bell__row-btn"
                        onClick={() => onOpen(job)}
                        data-testid={`notification-bell-row-${job.id}-open`}
                    >
                        Open
                    </button>
                )}
                {!isInFlight && (
                    <button
                        type="button"
                        className="notif-bell__row-btn notif-bell__row-btn--ghost"
                        onClick={() => onDismiss(job)}
                        data-testid={`notification-bell-row-${job.id}-dismiss`}
                    >
                        Dismiss
                    </button>
                )}
                {isInFlight && onCancel && (
                    <button
                        type="button"
                        className="notif-bell__row-btn notif-bell__row-btn--danger"
                        onClick={() => onCancel(job)}
                        data-testid={`notification-bell-row-${job.id}-cancel`}
                        title="Stop showing this job. The underlying work may still finish server-side until the runner honors the cancel flag."
                    >
                        Cancel
                    </button>
                )}
            </div>
        </div>
    );
}
