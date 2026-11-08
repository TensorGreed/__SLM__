/**
 * RunFailureBanner — why a failed real run stopped, and a one-click way to continue it.
 *
 * Reads ``config._runtime.failure`` (saved by the backend when an external run does
 * not complete). When a resumable checkpoint exists, "Continue from step N" calls
 * ``POST .../runs/{id}/resume-from/{N}``: a new run that continues at that step with
 * its weights, optimizer and scheduler state. The failed run stays as the record.
 */

import { useState } from 'react';
import api from '../../api/client';
import {
    errorDetail,
    forkNotice,
    runFailure,
    type ResumeFromResponse,
} from './checkpointsPanelUtils';

interface RunFailureBannerProps {
    projectId: number;
    experiment: { id: number; status: string; config?: Record<string, unknown> };
    /** Called with the new run id after a successful continue. */
    onContinued?: (newExperimentId: number) => void;
}

const TITLE: Record<string, string> = {
    timeout: 'Training hit its time limit',
    signal: 'Training was stopped',
    exit: 'Training exited with an error',
    error: 'Training could not be monitored',
};

export default function RunFailureBanner({ projectId, experiment, onContinued }: RunFailureBannerProps) {
    const [busy, setBusy] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const failure = runFailure(experiment.config);
    if ((experiment.status || '').toLowerCase() !== 'failed' || !failure) return null;
    const step = failure.resumable_checkpoint_step ?? null;

    const handleContinue = async () => {
        if (step === null) return;
        setBusy(true);
        setError(null);
        setNotice(null);
        try {
            const response = await api.post<ResumeFromResponse>(
                `/projects/${projectId}/training/runs/${experiment.id}/resume-from/${step}`,
                {},
            );
            const newId = Number(response.data?.experiment_id);
            if (Number.isFinite(newId) && newId > 0) {
                setNotice(forkNotice(newId, step, response.data));
                onContinued?.(newId);
            }
        } catch (err) {
            setError(errorDetail(err, `Continuing from step ${step} failed.`));
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="run-failure-banner" role="alert" data-testid="run-failure-banner">
            <div className="run-failure-banner__body">
                <strong>{TITLE[failure.code ?? ''] ?? 'Training failed'}</strong>
                <span className="run-failure-banner__message">{failure.message}</span>
                {notice && <span className="run-failure-banner__notice">{notice}</span>}
                {error && <span className="run-failure-banner__error">{error}</span>}
            </div>
            {step !== null && !notice && (
                <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => void handleContinue()}
                    disabled={busy}
                >
                    {busy ? 'Creating run…' : `Continue from step ${step}`}
                </button>
            )}
        </div>
    );
}
