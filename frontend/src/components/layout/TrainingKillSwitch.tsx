/**
 * Kill-switch action that surfaces in the NotificationBell when a
 * training run has been diverging (sparkline trend = ``up``) for N
 * consecutive polls. Built on top of the bell's existing trend
 * counter (``jobsStore.upTrendCountById``); the threshold lives at
 * ``KILL_SWITCH_TREND_UP_THRESHOLD = 3`` (12s of sustained
 * divergence at the 4s poll cadence).
 *
 * Click-flow:
 *   1. Button reveal: a single ⚠ kill-switch button appears at the
 *      end of the row when the trend counter crosses the threshold.
 *   2. Confirm dialog: spells out the situation + lets the user
 *      pick `cancel` (just stop the run) or `cancel + clone`
 *      (stop + spin up a new PENDING experiment with the same
 *      config so they can immediately re-launch).
 *   3. On confirm: calls ``cancelJob`` (always) +
 *      ``cloneExperiment`` (optional). Toast on success.
 *
 * The component is intentionally self-contained — the bell row
 * just renders ``<TrainingKillSwitch job={job} />`` and the
 * component decides whether to show anything based on the
 * trend counter.
 */

import { useCallback, useState } from 'react';

import api from '../../api/client';
import { cancelJob, type Job } from '../../api/jobs';
import {
    KILL_SWITCH_TREND_UP_THRESHOLD,
    useJobsStore,
} from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';


interface TrainingKillSwitchProps {
    job: Job;
}


export default function TrainingKillSwitch({ job }: TrainingKillSwitchProps) {
    const upCount = useJobsStore(
        (s) => s.upTrendCountById[job.id] || 0,
    );
    const refreshJobs = useJobsStore((s) => s.refreshAfterLocalChange);
    const [confirmOpen, setConfirmOpen] = useState(false);
    const [working, setWorking] = useState(false);

    // Gate: kill switch only renders for in-flight training jobs
    // that have crossed the divergence threshold AND have a
    // resolvable experiment id (clone target).
    const projectId = job.project_id;
    const experimentId =
        typeof job.params?.experiment_id === 'number'
            ? job.params.experiment_id
            : null;
    const eligible =
        job.kind === 'training_start'
        && (job.status === 'queued' || job.status === 'running')
        && upCount >= KILL_SWITCH_TREND_UP_THRESHOLD
        && projectId !== null
        && experimentId !== null;

    const handleAction = useCallback(
        async (clone: boolean) => {
            if (working) return;
            setWorking(true);
            try {
                // Always cancel first. Clone is opt-in via the
                // dialog action.
                await cancelJob(job.id);
                let clonedName: string | null = null;
                if (clone && projectId !== null && experimentId !== null) {
                    try {
                        const resp = await api.post<{ id: number; name: string }>(
                            `/projects/${projectId}/training/experiments/`
                            + `${experimentId}/clone`,
                        );
                        clonedName = resp.data?.name || `#${resp.data?.id}`;
                    } catch (err) {
                        const msg =
                            (err as { message?: string })?.message
                            || 'clone failed';
                        toast.error(`Cancelled — but clone failed: ${msg}`);
                        setWorking(false);
                        setConfirmOpen(false);
                        await refreshJobs();
                        return;
                    }
                }
                if (clonedName) {
                    toast.success(
                        `Cancelled diverging run · cloned to "${clonedName}" (PENDING).`,
                        6000,
                    );
                } else {
                    toast.info('Cancelled diverging run.', 4000);
                }
                await refreshJobs();
            } catch (err) {
                const msg =
                    (err as { message?: string })?.message || 'cancel failed';
                toast.error(`Kill switch: ${msg}`);
            } finally {
                setWorking(false);
                setConfirmOpen(false);
            }
        },
        [job.id, projectId, experimentId, refreshJobs, working],
    );

    if (!eligible) return null;

    if (!confirmOpen) {
        return (
            <button
                type="button"
                className="notif-bell__kill-switch"
                onClick={() => setConfirmOpen(true)}
                disabled={working}
                data-testid={`notification-bell-row-${job.id}-kill-switch`}
                title={
                    `Training loss has been trending up for `
                    + `${upCount} consecutive polls (~${upCount * 4}s). `
                    + `This run is probably diverging — click to cancel `
                    + `and optionally clone the config for a fresh retry.`
                }
            >
                ⚠ kill
            </button>
        );
    }

    return (
        <div
            className="notif-bell__kill-switch-confirm"
            role="dialog"
            aria-label="Confirm kill switch"
            data-testid={`notification-bell-row-${job.id}-kill-switch-confirm`}
        >
            <div className="notif-bell__kill-switch-msg">
                Loss diverging ({upCount} polls). Cancel this run?
            </div>
            <div className="notif-bell__kill-switch-actions">
                <button
                    type="button"
                    className="notif-bell__row-btn notif-bell__row-btn--danger"
                    onClick={() => void handleAction(true)}
                    disabled={working}
                    data-testid={
                        `notification-bell-row-${job.id}-kill-switch-cancel-clone`
                    }
                    title={
                        'Cancel + create a new PENDING experiment with the '
                        + 'same config so you can re-launch immediately.'
                    }
                >
                    {working ? 'Working…' : 'Cancel & clone'}
                </button>
                <button
                    type="button"
                    className="notif-bell__row-btn"
                    onClick={() => void handleAction(false)}
                    disabled={working}
                    data-testid={
                        `notification-bell-row-${job.id}-kill-switch-cancel-only`
                    }
                    title="Cancel without creating a retry experiment."
                >
                    {working ? 'Working…' : 'Cancel only'}
                </button>
                <button
                    type="button"
                    className="notif-bell__row-btn notif-bell__row-btn--ghost"
                    onClick={() => setConfirmOpen(false)}
                    disabled={working}
                    data-testid={
                        `notification-bell-row-${job.id}-kill-switch-dismiss`
                    }
                >
                    Keep going
                </button>
            </div>
        </div>
    );
}
