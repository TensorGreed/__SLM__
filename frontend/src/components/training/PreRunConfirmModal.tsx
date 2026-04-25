/**
 * PreRunConfirmModal — P20 pre-run confirm with cost provenance badge.
 *
 * Shown right before the user fires "Start training". POSTs to P18's
 * cost estimator on mount with the planned config + base model + target
 * profile, then renders the gpu_hours / USD / CO2 numbers and a
 * ``measured | estimated`` provenance badge driven by the same
 * heuristic the Why-this-plan panel uses.
 *
 * Two affordances:
 *   - Confirm: invokes the parent's ``onConfirm`` callback (which is
 *     responsible for actually launching the run).
 *   - Cancel: closes the modal via ``onCancel``.
 *
 * The modal is its own component (not coupled to TrainingPanel internals)
 * so the same surface can be reused from any "I'm about to launch a run"
 * affordance — Wave-D rerun-from-manifest, P20 autopilot one-click, etc.
 */

import { useEffect, useState } from 'react';
import api from '../../api/client';

interface CostEstimate {
    gpu_hours: number;
    usd: number;
    co2_kg: number;
    provenance: 'measured' | 'estimated';
    confidence: number;
    confidence_band: 'high' | 'medium' | 'low';
    calibration: { cohort: string; sample_count: number };
    pricing: { hourly_rate_usd: number; source: string };
}

interface PreRunConfirmModalProps {
    projectId: number;
    config: Record<string, unknown>;
    baseModel?: string | null;
    targetProfileId?: string | null;
    onConfirm: () => void;
    onCancel: () => void;
    /** Optional title override — defaults to "Confirm training launch". */
    title?: string;
    /** Override the affirmative button label (e.g., "Launch", "Start"). */
    confirmLabel?: string;
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

function formatHours(gpuHours: number): string {
    if (!Number.isFinite(gpuHours) || gpuHours <= 0) return '—';
    if (gpuHours < 1) return `${Math.round(gpuHours * 60)} min`;
    return `${gpuHours.toFixed(2)} h`;
}

export default function PreRunConfirmModal({
    projectId,
    config,
    baseModel,
    targetProfileId,
    onConfirm,
    onCancel,
    title = 'Confirm training launch',
    confirmLabel = 'Launch',
}: PreRunConfirmModalProps) {
    const [estimate, setEstimate] = useState<CostEstimate | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        async function fetchEstimate() {
            try {
                const response = await api.post<CostEstimate>(
                    `/projects/${projectId}/training/plan/estimate-cost`,
                    {
                        config,
                        base_model: baseModel || undefined,
                        target_profile_id: targetProfileId || undefined,
                    },
                );
                if (!cancelled) {
                    setEstimate(response.data);
                }
            } catch (err) {
                if (!cancelled) {
                    setErrorMessage(errorDetail(err, 'Failed to load cost estimate.'));
                }
            } finally {
                if (!cancelled) {
                    setIsLoading(false);
                }
            }
        }
        void fetchEstimate();
        return () => {
            cancelled = true;
        };
    }, [projectId, config, baseModel, targetProfileId]);

    return (
        <div
            className="pre-run-modal__backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="pre-run-modal-title"
        >
            <div className="pre-run-modal">
                <div className="pre-run-modal__head">
                    <h3 id="pre-run-modal-title">{title}</h3>
                </div>

                <div className="pre-run-modal__body">
                    {isLoading && <div>Estimating cost…</div>}
                    {errorMessage && !isLoading && (
                        <div className="pre-run-modal__error">{errorMessage}</div>
                    )}
                    {estimate && (
                        <>
                            <div className="pre-run-modal__provenance-row">
                                <span className="pre-run-modal__label">Provenance</span>
                                <span
                                    className={`pre-run-modal__provenance pre-run-modal__provenance--${estimate.provenance}`}
                                    title={
                                        estimate.provenance === 'measured'
                                            ? `Calibrated from ${estimate.calibration.sample_count} historical run(s)`
                                            : 'Heuristic estimate — no comparable historical runs yet.'
                                    }
                                >
                                    {estimate.provenance}
                                </span>
                                <span className="pre-run-modal__confidence">
                                    {(estimate.confidence * 100).toFixed(0)}% (
                                    {estimate.confidence_band})
                                </span>
                            </div>
                            <ul className="pre-run-modal__metrics">
                                <li>
                                    <span>GPU hours</span>
                                    <strong>{formatHours(estimate.gpu_hours)}</strong>
                                </li>
                                <li>
                                    <span>USD</span>
                                    <strong>${estimate.usd.toFixed(2)}</strong>
                                </li>
                                <li>
                                    <span>CO₂</span>
                                    <strong>{estimate.co2_kg.toFixed(3)} kg</strong>
                                </li>
                                <li>
                                    <span>Pricing</span>
                                    <strong>
                                        ${estimate.pricing.hourly_rate_usd.toFixed(2)}/h ·{' '}
                                        {estimate.pricing.source}
                                    </strong>
                                </li>
                                <li>
                                    <span>Cohort</span>
                                    <strong>
                                        {estimate.calibration.cohort} (n=
                                        {estimate.calibration.sample_count})
                                    </strong>
                                </li>
                            </ul>
                        </>
                    )}
                </div>

                <div className="pre-run-modal__footer">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onCancel}
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={onConfirm}
                        disabled={isLoading}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
