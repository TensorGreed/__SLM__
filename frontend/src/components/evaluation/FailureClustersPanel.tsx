/**
 * Failure pattern grouper with cluster explanation and augmentation recommendations.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import api from '../../api/client';
import { augmentFromCluster } from '../../api/synthPlaybook';
import './FailureClustersPanel.css';

interface AugmentState {
    status: 'idle' | 'running' | 'ok' | 'error';
    rows?: number;
    backend?: string;
    elapsed?: number;
    message?: string;
}

interface ClusterExplanation {
    state: 'loading' | 'ok' | 'judge_unavailable' | 'error' | 'cluster_not_found';
    text?: string;
    note?: string;
    model?: string;
    cached?: boolean;
}

interface ClusterExplanationResponse {
    cluster_id: string;
    explanation: string;
    status: 'ok' | 'judge_unavailable' | 'error' | 'cluster_not_found';
    cached: boolean;
    generated_at: string | null;
    model: string | null;
    exemplar_count: number;
    note?: string | null;
}

interface EvalResultSummary {
    id: number;
    dataset_name: string;
    eval_type: string;
    pass_rate: number | null;
}

interface ClusterExemplar {
    prompt: string;
    reference: string;
    prediction: string;
    source?: string;
    judge_score?: number | null;
    judge_rationale?: string;
    metric_name?: string;
    metric_value?: number | null;
    test_type?: string;
}

interface FailureCluster {
    cluster_id: string;
    reason_code: string;
    output_pattern: string;
    failure_count: number;
    share_of_total: number;
    classifier_confidence: number;
    classifier_reason: string;
    exemplars: ClusterExemplar[];
}

interface RemediationPlanLink {
    plan_id: string;
    artifact_id: number;
    created_at: string | null;
    root_causes?: string[];
    summary?: {
        total_failures_analyzed?: number;
        cluster_count?: number;
        dominant_root_cause?: string;
    };
}

interface FailureClustersResponse {
    eval_result_id: number;
    experiment_id: number | null;
    dataset_name: string;
    eval_type: string;
    total_failures_analyzed: number;
    reason_code_totals: Record<string, number>;
    dominant_reason_code: string | null;
    clusters: FailureCluster[];
    remediation_plans: RemediationPlanLink[];
}

interface FailureClustersPanelProps {
    projectId: number;
    evalResults: EvalResultSummary[];
    onGenerateRemediation?: (evalResultId: number) => void;
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

export default function FailureClustersPanel({
    projectId,
    evalResults,
    onGenerateRemediation,
}: FailureClustersPanelProps) {
    const [selectedResultId, setSelectedResultId] = useState<number | ''>('');
    const [clusters, setClusters] = useState<FailureClustersResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [expandedClusters, setExpandedClusters] = useState<Set<string>>(new Set());
    // Theme 8 Epic 3: per-cluster LLM-judge explanation, keyed by
    // cluster_id. Server-side cache lives on the EvalResult; this
    // session-local map tracks loading + in-flight state so the UI
    // can show a spinner without re-fetching on every expand.
    const [explanations, setExplanations] = useState<
        Record<string, ClusterExplanation>
    >({});

    // USER-SUCCESS Epic 2b: cluster-targeted augmentation status per cluster.
    const [augmentStates, setAugmentStates] = useState<Record<string, AugmentState>>({});

    useEffect(() => {
        if (selectedResultId === '' && evalResults.length > 0) {
            setSelectedResultId(evalResults[0].id);
        }
        if (selectedResultId !== '' && !evalResults.some((r) => r.id === selectedResultId)) {
            // Selected result disappeared after the experiment changed — reset.
            setSelectedResultId(evalResults.length > 0 ? evalResults[0].id : '');
            setClusters(null);
        }
    }, [evalResults, selectedResultId]);

    const fetchClusters = useCallback(async (resultId: number) => {
        setIsLoading(true);
        setErrorMessage(null);
        try {
            const res = await api.get<FailureClustersResponse>(
                `/projects/${projectId}/evaluation/${resultId}/failure-clusters`,
            );
            setClusters(res.data);
            setExpandedClusters(new Set());
            setExplanations({});
        } catch (err) {
            setErrorMessage(errorDetail(err, 'Failed to load failure clusters.'));
            setClusters(null);
        } finally {
            setIsLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        if (typeof selectedResultId === 'number') {
            void fetchClusters(selectedResultId);
        }
    }, [selectedResultId, fetchClusters]);

    const fetchExplanation = useCallback(
        async (clusterId: string, opts?: { force?: boolean }) => {
            if (typeof selectedResultId !== 'number') return;
            setExplanations((prev) => ({
                ...prev,
                [clusterId]: { ...(prev[clusterId] ?? {}), state: 'loading' },
            }));
            try {
                const res = await api.post<ClusterExplanationResponse>(
                    `/projects/${projectId}/evaluation/${selectedResultId}/clusters/${encodeURIComponent(clusterId)}/explain`,
                    null,
                    { params: opts?.force ? { force: true } : undefined },
                );
                const payload = res.data;
                const status = (payload.status || 'error') as ClusterExplanation['state'];
                setExplanations((prev) => ({
                    ...prev,
                    [clusterId]: {
                        state: status,
                        text: payload.explanation || '',
                        note: payload.note || undefined,
                        model: payload.model || undefined,
                        cached: payload.cached,
                    },
                }));
            } catch (err) {
                setExplanations((prev) => ({
                    ...prev,
                    [clusterId]: {
                        state: 'error',
                        note: errorDetail(err, 'Failed to fetch explanation.'),
                    },
                }));
            }
        },
        [projectId, selectedResultId],
    );

    const toggleCluster = useCallback(
        (clusterId: string) => {
            const isCurrentlyExpanded = expandedClusters.has(clusterId);
            // Lazy-fetch the explanation only the first time this
            // cluster is expanded *this session*. Server caches the
            // result on the EvalResult so the POST is cheap on
            // subsequent sessions, but we still skip it once local
            // state has a verdict to avoid the spinner flash.
            const shouldFetch =
                !isCurrentlyExpanded && !explanations[clusterId]?.state;
            setExpandedClusters((prev) => {
                const next = new Set(prev);
                if (next.has(clusterId)) {
                    next.delete(clusterId);
                } else {
                    next.add(clusterId);
                }
                return next;
            });
            if (shouldFetch) {
                setExplanations((prev) => ({
                    ...prev,
                    [clusterId]: { state: 'loading' },
                }));
                void fetchExplanation(clusterId);
            }
        },
        [expandedClusters, explanations, fetchExplanation],
    );

    const summary = useMemo(() => {
        if (!clusters) {
            return null;
        }
        return {
            total: clusters.total_failures_analyzed,
            clusterCount: clusters.clusters.length,
            dominant: clusters.dominant_reason_code,
            reasonTotals: Object.entries(clusters.reason_code_totals).sort((a, b) => b[1] - a[1]),
        };
    }, [clusters]);

    if (evalResults.length === 0) {
        return (
            <div className="card failure-clusters-card">
                <h3>Failure clusters</h3>
                <p className="failure-clusters-empty">
                    Run at least one evaluation to cluster its failures here.
                </p>
            </div>
        );
    }

    return (
        <div className="card failure-clusters-card">
            <div className="failure-clusters-head">
                <div>
                    <h3>Failure clusters</h3>
                    <p className="failure-clusters-subtitle">
                        Row-level failures grouped by <strong>reason code</strong> and
                        <strong> output-pattern signature</strong>. Each cluster links back to remediation.
                    </p>
                </div>
                <div className="failure-clusters-controls">
                    <select
                        aria-label="Eval result to cluster"
                        className="input"
                        value={selectedResultId}
                        onChange={(e) => {
                            const next = e.target.value === '' ? '' : Number(e.target.value);
                            setSelectedResultId(next);
                        }}
                    >
                        {evalResults.map((result) => (
                            <option key={result.id} value={result.id}>
                                #{result.id} · {result.dataset_name} · {result.eval_type}
                            </option>
                        ))}
                    </select>
                    <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => {
                            if (typeof selectedResultId === 'number') {
                                void fetchClusters(selectedResultId);
                            }
                        }}
                        disabled={isLoading || selectedResultId === ''}
                    >
                        {isLoading ? 'Loading…' : 'Refresh'}
                    </button>
                </div>
            </div>

            {errorMessage && (
                <div className="failure-clusters-error">{errorMessage}</div>
            )}

            {summary && (
                <div className="failure-clusters-summary">
                    <span><strong>{summary.total}</strong> failures · <strong>{summary.clusterCount}</strong> clusters</span>
                    {summary.dominant && (
                        <span>Dominant: <code>{summary.dominant}</code></span>
                    )}
                    {summary.reasonTotals.length > 0 && (
                        <span className="failure-clusters-reason-totals">
                            {summary.reasonTotals.map(([code, count]) => (
                                <span key={code} className={`failure-clusters-chip failure-clusters-chip-${code}`}>
                                    {code} ({count})
                                </span>
                            ))}
                        </span>
                    )}
                </div>
            )}

            {clusters && clusters.clusters.length === 0 && !isLoading && (
                <p className="failure-clusters-empty">
                    No failures to cluster for this eval result — every scored row is above the failure threshold.
                </p>
            )}

            {clusters && clusters.clusters.length > 0 && (
                <div className="failure-clusters-list">
                    {clusters.clusters.map((cluster) => {
                        const expanded = expandedClusters.has(cluster.cluster_id);
                        return (
                            <div
                                key={cluster.cluster_id}
                                className={`failure-cluster failure-cluster-${cluster.reason_code}`}
                            >
                                <button
                                    type="button"
                                    className="failure-cluster-head"
                                    onClick={() => toggleCluster(cluster.cluster_id)}
                                    aria-expanded={expanded ? 'true' : 'false'}
                                >
                                    <span className="failure-cluster-title">
                                        <span className={`failure-clusters-chip failure-clusters-chip-${cluster.reason_code}`}>
                                            {cluster.reason_code}
                                        </span>
                                        <span className="failure-cluster-pattern">{cluster.output_pattern}</span>
                                    </span>
                                    <span className="failure-cluster-stats">
                                        <span><strong>{cluster.failure_count}</strong> rows</span>
                                        <span>{Math.round(cluster.share_of_total * 100)}%</span>
                                        <span className="failure-cluster-caret">{expanded ? '▾' : '▸'}</span>
                                    </span>
                                </button>
                                {expanded && (
                                    <div className="failure-cluster-body">
                                        <ClusterExplanationChip
                                            explanation={explanations[cluster.cluster_id]}
                                            onRetry={() =>
                                                void fetchExplanation(cluster.cluster_id, {
                                                    force: true,
                                                })
                                            }
                                        />
                                        <ClusterAugmentControl
                                            projectId={projectId}
                                            evalResultId={selectedResultId as number}
                                            clusterId={cluster.cluster_id}
                                            state={augmentStates[cluster.cluster_id]}
                                            onStateChange={(next) =>
                                                setAugmentStates((prev) => ({
                                                    ...prev,
                                                    [cluster.cluster_id]: next,
                                                }))
                                            }
                                        />
                                        {cluster.classifier_reason && (
                                            <p className="failure-cluster-reason">{cluster.classifier_reason}</p>
                                        )}
                                        <ul className="failure-cluster-exemplars">
                                            {cluster.exemplars.map((ex, idx) => (
                                                <li key={idx} className="failure-cluster-exemplar">
                                                    {ex.prompt && (
                                                        <div>
                                                            <span className="failure-cluster-ex-label">prompt</span>
                                                            <span className="failure-cluster-ex-text">{ex.prompt}</span>
                                                        </div>
                                                    )}
                                                    {ex.reference && (
                                                        <div>
                                                            <span className="failure-cluster-ex-label">reference</span>
                                                            <span className="failure-cluster-ex-text">{ex.reference}</span>
                                                        </div>
                                                    )}
                                                    {ex.prediction && (
                                                        <div>
                                                            <span className="failure-cluster-ex-label">prediction</span>
                                                            <span className="failure-cluster-ex-text">{ex.prediction}</span>
                                                        </div>
                                                    )}
                                                    {typeof ex.judge_score === 'number' && (
                                                        <div className="failure-cluster-ex-note">
                                                            judge: {ex.judge_score}/5
                                                            {ex.judge_rationale ? ` — ${ex.judge_rationale}` : ''}
                                                        </div>
                                                    )}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}

            {clusters && (clusters.remediation_plans.length > 0 || onGenerateRemediation) && (
                <div className="failure-clusters-remediation">
                    <h4>Remediation</h4>
                    {clusters.remediation_plans.length > 0 ? (
                        <ul className="failure-clusters-plans">
                            {clusters.remediation_plans.map((plan) => (
                                <li key={plan.artifact_id}>
                                    <strong>{plan.plan_id}</strong>
                                    {plan.summary?.dominant_root_cause && (
                                        <span> · dominant: {plan.summary.dominant_root_cause}</span>
                                    )}
                                    {plan.summary?.cluster_count !== undefined && (
                                        <span> · {plan.summary.cluster_count} clusters</span>
                                    )}
                                    {plan.created_at && (
                                        <span className="failure-clusters-plan-time">
                                            {' — '}
                                            {new Date(plan.created_at).toLocaleString()}
                                        </span>
                                    )}
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="failure-clusters-empty">
                            No remediation plan yet for this eval result.
                        </p>
                    )}
                    {onGenerateRemediation && typeof selectedResultId === 'number' && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => onGenerateRemediation(selectedResultId)}
                        >
                            {clusters.remediation_plans.length > 0
                                ? 'Generate a fresh plan'
                                : 'Generate remediation plan'}
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}


interface ClusterExplanationChipProps {
    explanation: ClusterExplanation | undefined;
    onRetry: () => void;
}

function ClusterExplanationChip({
    explanation,
    onRetry,
}: ClusterExplanationChipProps) {
    if (!explanation || explanation.state === 'loading') {
        return (
            <p
                className="failure-cluster-explanation failure-cluster-explanation-loading"
                data-testid="cluster-explanation-loading"
            >
                💡 Analyzing the cluster…
            </p>
        );
    }
    if (explanation.state === 'ok' && explanation.text) {
        return (
            <p
                className="failure-cluster-explanation failure-cluster-explanation-ok"
                data-testid="cluster-explanation-ok"
            >
                <span aria-hidden="true">💡 </span>
                <strong>{explanation.text}</strong>
                {explanation.cached && (
                    <span
                        className="failure-cluster-explanation-note"
                        data-testid="cluster-explanation-cached"
                    >
                        {' '}· cached
                    </span>
                )}
                {explanation.model && (
                    <span className="failure-cluster-explanation-note">
                        {' '}· {explanation.model}
                    </span>
                )}
            </p>
        );
    }
    if (explanation.state === 'judge_unavailable') {
        return (
            <p
                className="failure-cluster-explanation failure-cluster-explanation-soft"
                data-testid="cluster-explanation-judge-unavailable"
            >
                💡 No judge model configured — explanations skipped.
                {explanation.note && (
                    <span className="failure-cluster-explanation-note">
                        {' '}{explanation.note}
                    </span>
                )}
            </p>
        );
    }
    if (explanation.state === 'cluster_not_found') {
        return (
            <p
                className="failure-cluster-explanation failure-cluster-explanation-soft"
                data-testid="cluster-explanation-missing"
            >
                💡 Cluster no longer present in this eval result.
            </p>
        );
    }
    // 'error'
    return (
        <p
            className="failure-cluster-explanation failure-cluster-explanation-error"
            data-testid="cluster-explanation-error"
        >
            💡 Couldn't generate explanation
            {explanation.note ? `: ${explanation.note}` : '.'}
            {' '}
            <button
                type="button"
                className="failure-cluster-explanation-retry"
                onClick={onRetry}
                data-testid="cluster-explanation-retry"
            >
                Retry
            </button>
        </p>
    );
}


// ─────────────────────────────────────────────────────────────────────
// ClusterAugmentControl (USER-SUCCESS Epic 2b)
// Renders a per-cluster "Augment from this cluster" CTA. Generated
// rows land in the project's synthetic dataset with
// review_status=pending — the SynthReviewQueue on the Synthetic tab
// is where the user accepts or rejects them.
// ─────────────────────────────────────────────────────────────────────

interface ClusterAugmentControlProps {
    projectId: number;
    evalResultId: number;
    clusterId: string;
    state: AugmentState | undefined;
    onStateChange: (next: AugmentState) => void;
}

function ClusterAugmentControl({
    projectId,
    evalResultId,
    clusterId,
    state,
    onStateChange,
}: ClusterAugmentControlProps) {
    const [targetCount, setTargetCount] = useState(20);
    const status = state?.status ?? 'idle';

    const handleRun = useCallback(async () => {
        onStateChange({ status: 'running' });
        try {
            const result = await augmentFromCluster(projectId, {
                evalResultId,
                clusterId,
                targetCount,
            });
            onStateChange({
                status: 'ok',
                rows: result.rows.length,
                backend: result.backend_used,
                elapsed: result.elapsed_sec,
            });
        } catch (err: any) {
            const message =
                err?.response?.data?.detail || err?.message || 'Augment failed';
            onStateChange({ status: 'error', message });
        }
    }, [projectId, evalResultId, clusterId, targetCount, onStateChange]);

    return (
        <div
            className="failure-cluster-augment"
            data-testid={`failure-cluster-augment-${clusterId}`}
        >
            <div className="failure-cluster-augment-row">
                <label className="failure-cluster-augment-count-label">
                    Generate
                    <input
                        type="number"
                        min={1}
                        max={500}
                        value={targetCount}
                        onChange={(e) =>
                            setTargetCount(
                                Math.max(1, Math.min(500, Number(e.target.value) || 1)),
                            )
                        }
                        data-testid={`failure-cluster-augment-count-${clusterId}`}
                    />
                    rows from this cluster
                </label>
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleRun}
                    disabled={status === 'running'}
                    data-testid={`failure-cluster-augment-run-${clusterId}`}
                >
                    {status === 'running' ? 'Generating…' : 'Augment from this cluster →'}
                </button>
            </div>
            {status === 'ok' && state && (
                <p
                    className="failure-cluster-augment-result"
                    data-testid={`failure-cluster-augment-ok-${clusterId}`}
                >
                    ✓ Generated {state.rows} rows via <code>{state.backend}</code> in {state.elapsed?.toFixed(2)}s.
                    Pending review on the Synthetic tab.
                </p>
            )}
            {status === 'error' && state?.message && (
                <p
                    className="failure-cluster-augment-error"
                    data-testid={`failure-cluster-augment-error-${clusterId}`}
                >
                    {state.message}
                </p>
            )}
        </div>
    );
}
