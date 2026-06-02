/**
 * Inline live signals for an in-flight experiment row on the
 * TrainingPanel dashboard — sparkline + kill switch, mirroring
 * what the NotificationBell already renders. Closes the
 * dashboard/bell asymmetry so users who live on the training
 * page don't have to watch the bell for training-health signal.
 *
 * Subscribes to nothing directly — caller resolves the matching
 * Job (kind=``training_start``, params.experiment_id matches the
 * row's experiment id) from the jobs store and passes it in.
 * Decoupling the lookup from the render keeps this component
 * testable in isolation without mounting the full TrainingPanel.
 *
 * Renders nothing when:
 *   - no job is passed (experiment isn't tracked by a job, or
 *     the jobs poll hasn't surfaced one yet),
 *   - the job's status is terminal (succeeded / failed /
 *     cancelled — the row's existing status badge carries the
 *     final state; live signals would be misleading).
 */

import type { Job } from '../../api/jobs';
import TrainingKillSwitch from '../layout/TrainingKillSwitch';
import TrainingLossSparkline from '../layout/TrainingLossSparkline';


interface ExperimentLiveSignalsProps {
    /** The matching training_start job for this experiment row.
     *  Caller (TrainingPanel) builds a job-by-experiment-id map
     *  once and passes the lookup result here. */
    job: Job | undefined;
}


export default function ExperimentLiveSignals({
    job,
}: ExperimentLiveSignalsProps) {
    if (!job) return null;
    if (job.kind !== 'training_start') return null;
    if (job.status !== 'queued' && job.status !== 'running') return null;

    const hasMetrics = Array.isArray(job.metrics_recent);

    return (
        <span
            className="training-experiment-live-signals"
            data-testid={`experiment-live-signals-${job.params?.experiment_id ?? job.id}`}
        >
            {hasMetrics && (
                <TrainingLossSparkline
                    points={job.metrics_recent || []}
                    // Slightly narrower than the bell variant — the
                    // dashboard row has more horizontal space
                    // contention from action buttons, and the
                    // sparkline's trend tint is the dominant
                    // signal anyway, not pixel-level resolution.
                    width={60}
                    height={14}
                />
            )}
            <TrainingKillSwitch job={job} />
        </span>
    );
}
