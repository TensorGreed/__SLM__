/**
 * WhyThisPlanPanel — P20 Training Planner reproducibility & cost view.
 *
 * Replaces the old plain "planning card" with three layered facts:
 *   1. Strategy + memory budget (sourced from the Experiment.config snapshot
 *      already in TrainingPanel; nothing extra to fetch).
 *   2. Projected gpu_hours / USD / CO2 from P18 cost-estimator with a
 *      "measured vs estimated" badge driven by ``provenance``.
 *   3. "Show the manifest" expand that GETs the P14 immutable manifest
 *      (only after the run starts, since the manifest is captured at
 *      launch).
 *
 * Designed to be drop-in inside TrainingPanel's active-experiment view —
 * takes the projectId + experiment summary as props and fetches its own
 * data via the shared api client.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import api from '../../api/client';

interface ExperimentSummary {
    id: number;
    name: string;
    base_model: string;
    training_mode: string;
    status: string;
    config?: Record<string, unknown> | null;
}

interface CostEstimate {
    gpu_hours: number;
    usd: number;
    co2_kg: number;
    provenance: 'measured' | 'estimated';
    confidence: number;
    confidence_band: 'high' | 'medium' | 'low';
    calibration: {
        cohort: string;
        sample_count: number;
        median_cohort_seconds: number | null;
        variability_cv: number | null;
        heuristic_seconds: number;
        fallback_used: boolean;
        planned_total_steps: number;
        planned_gpu_count: number;
        planned_size_bucket: string;
        training_mode: string;
        target_profile_id: string;
    };
    pricing: { hourly_rate_usd: number; source: string };
    co2: { power_kw: number; grid_intensity_g_per_kwh: number; intensity_source: string };
}

interface ManifestDump {
    [key: string]: unknown;
}

interface WhyThisPlanPanelProps {
    projectId: number;
    experiment: ExperimentSummary;
    /**
     * Optional override of target profile when computing the cost estimate.
     * Falls back to ``experiment.config.target_profile_id`` then ``vllm_server``.
     */
    targetProfileId?: string;
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

function formatNumber(value: number, digits = 4): string {
    if (!Number.isFinite(value)) return '—';
    return value.toFixed(digits);
}

function formatHours(gpuHours: number): string {
    if (!Number.isFinite(gpuHours) || gpuHours <= 0) return '—';
    if (gpuHours < 1) {
        return `${Math.round(gpuHours * 60)} min`;
    }
    return `${gpuHours.toFixed(2)} h`;
}

export default function WhyThisPlanPanel({
    projectId,
    experiment,
    targetProfileId,
}: WhyThisPlanPanelProps) {
    const [estimate, setEstimate] = useState<CostEstimate | null>(null);
    const [estimateLoading, setEstimateLoading] = useState(false);
    const [estimateError, setEstimateError] = useState<string | null>(null);

    const [manifest, setManifest] = useState<ManifestDump | null>(null);
    const [manifestLoading, setManifestLoading] = useState(false);
    const [manifestError, setManifestError] = useState<string | null>(null);
    const [manifestExpanded, setManifestExpanded] = useState(false);

    const config = useMemo(
        () => (experiment.config && typeof experiment.config === 'object' ? experiment.config : {}),
        [experiment.config],
    );

    const resolvedTarget = useMemo(() => {
        if (targetProfileId && targetProfileId.trim()) return targetProfileId.trim();
        const fromConfig = config['target_profile_id'];
        if (typeof fromConfig === 'string' && fromConfig.trim()) return fromConfig.trim();
        return 'vllm_server';
    }, [config, targetProfileId]);

    const strategy = useMemo(() => {
        const recipe = config['recipe'] || config['recipe_id'];
        const runtime = config['training_runtime_id'];
        const trainingMode = experiment.training_mode || (config['training_mode'] as string);
        return {
            mode: typeof trainingMode === 'string' ? trainingMode : 'sft',
            runtime: typeof runtime === 'string' ? runtime : 'auto',
            recipe: typeof recipe === 'string' ? recipe : '—',
        };
    }, [config, experiment.training_mode]);

    const memoryBudget = useMemo(() => {
        const batchSize = Number(config['batch_size'] ?? 0);
        const grad = Number(config['gradient_accumulation_steps'] ?? 0);
        const seq = Number(config['max_seq_length'] ?? 0);
        const useLora = Boolean(config['use_lora']);
        const gradCheckpoint = Boolean(config['gradient_checkpointing']);
        return {
            batchSize: Number.isFinite(batchSize) && batchSize > 0 ? batchSize : null,
            gradAccum: Number.isFinite(grad) && grad > 0 ? grad : null,
            maxSeq: Number.isFinite(seq) && seq > 0 ? seq : null,
            useLora,
            gradCheckpoint,
        };
    }, [config]);

    const fetchEstimate = useCallback(async () => {
        setEstimateLoading(true);
        setEstimateError(null);
        try {
            const response = await api.post<CostEstimate>(
                `/projects/${projectId}/training/plan/estimate-cost`,
                {
                    config: { ...config, training_mode: strategy.mode },
                    base_model: experiment.base_model,
                    target_profile_id: resolvedTarget,
                },
            );
            setEstimate(response.data);
        } catch (err) {
            setEstimateError(errorDetail(err, 'Failed to load cost estimate.'));
        } finally {
            setEstimateLoading(false);
        }
    }, [projectId, config, experiment.base_model, resolvedTarget, strategy.mode]);

    useEffect(() => {
        void fetchEstimate();
    }, [fetchEstimate]);

    const fetchManifest = useCallback(async () => {
        setManifestLoading(true);
        setManifestError(null);
        try {
            const response = await api.get<ManifestDump>(
                `/projects/${projectId}/training/runs/${experiment.id}/manifest`,
            );
            setManifest(response.data);
        } catch (err) {
            setManifestError(errorDetail(err, 'Manifest not yet captured for this run.'));
            setManifest(null);
        } finally {
            setManifestLoading(false);
        }
    }, [projectId, experiment.id]);

    const handleToggleManifest = useCallback(() => {
        const next = !manifestExpanded;
        setManifestExpanded(next);
        if (next && !manifest && !manifestLoading) {
            void fetchManifest();
        }
    }, [manifestExpanded, manifest, manifestLoading, fetchManifest]);

    return (
        <div className="card why-this-plan">
            <div className="why-this-plan__head">
                <h4 className="why-this-plan__title">Why this plan</h4>
                <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => void fetchEstimate()}
                    disabled={estimateLoading}
                >
                    {estimateLoading ? 'Refreshing…' : 'Refresh estimate'}
                </button>
            </div>

            <div className="why-this-plan__grid">
                <section className="why-this-plan__section" aria-label="Strategy">
                    <div className="why-this-plan__section-title">Strategy</div>
                    <ul className="why-this-plan__kv">
                        <li>
                            <span>Mode</span>
                            <strong>{strategy.mode}</strong>
                        </li>
                        <li>
                            <span>Runtime</span>
                            <strong>{strategy.runtime}</strong>
                        </li>
                        <li>
                            <span>Recipe</span>
                            <strong>{strategy.recipe}</strong>
                        </li>
                    </ul>
                </section>

                <section className="why-this-plan__section" aria-label="Memory budget">
                    <div className="why-this-plan__section-title">Memory budget</div>
                    <ul className="why-this-plan__kv">
                        <li>
                            <span>Batch × accum</span>
                            <strong>
                                {memoryBudget.batchSize ?? '—'}
                                {memoryBudget.gradAccum ? ` × ${memoryBudget.gradAccum}` : ''}
                            </strong>
                        </li>
                        <li>
                            <span>Max seq</span>
                            <strong>{memoryBudget.maxSeq ?? '—'}</strong>
                        </li>
                        <li>
                            <span>LoRA</span>
                            <strong>{memoryBudget.useLora ? 'on' : 'off'}</strong>
                        </li>
                        <li>
                            <span>Grad checkpoint</span>
                            <strong>{memoryBudget.gradCheckpoint ? 'on' : 'off'}</strong>
                        </li>
                    </ul>
                </section>

                <section className="why-this-plan__section" aria-label="Projected cost">
                    <div className="why-this-plan__section-title">
                        Projected cost
                        {estimate && (
                            <span
                                className={`why-this-plan__provenance why-this-plan__provenance--${estimate.provenance}`}
                                title={
                                    estimate.provenance === 'measured'
                                        ? `Calibrated from ${estimate.calibration.sample_count} historical run(s)`
                                        : 'Heuristic estimate — no comparable historical runs yet.'
                                }
                            >
                                {estimate.provenance}
                            </span>
                        )}
                    </div>
                    {estimateError && !estimateLoading && (
                        <div className="why-this-plan__error">{estimateError}</div>
                    )}
                    {estimate && (
                        <ul className="why-this-plan__kv why-this-plan__kv--cost">
                            <li>
                                <span>GPU hours</span>
                                <strong>{formatHours(estimate.gpu_hours)}</strong>
                            </li>
                            <li>
                                <span>USD</span>
                                <strong>${formatNumber(estimate.usd, 2)}</strong>
                            </li>
                            <li>
                                <span>CO₂</span>
                                <strong>{formatNumber(estimate.co2_kg, 3)} kg</strong>
                            </li>
                            <li>
                                <span>Confidence</span>
                                <strong>
                                    {(estimate.confidence * 100).toFixed(0)}% ({estimate.confidence_band})
                                </strong>
                            </li>
                            <li className="why-this-plan__kv-meta">
                                <span>Cohort</span>
                                <strong>
                                    {estimate.calibration.cohort} (n={estimate.calibration.sample_count})
                                </strong>
                            </li>
                            <li className="why-this-plan__kv-meta">
                                <span>Pricing</span>
                                <strong>
                                    ${formatNumber(estimate.pricing.hourly_rate_usd, 2)}/h ·{' '}
                                    {estimate.pricing.source}
                                </strong>
                            </li>
                        </ul>
                    )}
                </section>
            </div>

            <div className="why-this-plan__manifest">
                <button
                    className="btn btn-link btn-sm"
                    onClick={handleToggleManifest}
                    aria-expanded={manifestExpanded}
                >
                    {manifestExpanded ? '▼ Hide the manifest' : '▶ Show the manifest'}
                </button>
                {manifestExpanded && (
                    <div className="why-this-plan__manifest-body">
                        {manifestLoading && <div>Loading manifest…</div>}
                        {manifestError && !manifestLoading && (
                            <div className="why-this-plan__error">{manifestError}</div>
                        )}
                        {manifest && (
                            <pre className="why-this-plan__manifest-json">
                                {JSON.stringify(manifest, null, 2)}
                            </pre>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
