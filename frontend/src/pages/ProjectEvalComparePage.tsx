/**
 * Side-by-side eval comparison page (E3).
 *
 * Mounted at ``/project/:projectId/eval/compare?a=<exp>&b=<exp>``.
 * Renders:
 *   - Two experiment summary chips (A vs B) with pass_rate badges.
 *   - Per-metric delta table — regressions to the top.
 *   - Per-cluster failure diff: only-in-A (fixed), only-in-B
 *     (new regressions), shared (delta).
 *   - Config diff — primary fields always visible, other changed
 *     fields appended.
 *   - "Fix the gap" CTA when ``regressed`` — one-click rerun-from-
 *     manifest of A's config; degrades cleanly when A has no manifest.
 *
 * Advisory only — the rollback button creates a NEW experiment;
 * never overwrites anything.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useOutletContext } from 'react-router-dom';

import type {
    CompareResponse,
    ConfigDiffRow,
} from '../api/experimentCompare';
import {
    fetchExperimentCompare,
    rerunExperimentFromManifest,
} from '../api/experimentCompare';
import { toast } from '../stores/toastStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectEvalComparePage.css';


function formatNumeric(value: unknown): string {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'number') {
        if (Number.isInteger(value)) return String(value);
        return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    }
    if (typeof value === 'string') return value;
    return JSON.stringify(value);
}


function formatPassRate(value: number | null): string {
    if (value === null) return '—';
    return `${(value * 100).toFixed(1)}%`;
}


function formatDelta(value: number | null, asPct: boolean): string {
    if (value === null) return '—';
    if (asPct) {
        const sign = value > 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(1)} pp`;
    }
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')}`;
}


export default function ProjectEvalComparePage() {
    const navigate = useNavigate();
    const location = useLocation();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const [data, setData] = useState<CompareResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [rerunning, setRerunning] = useState(false);

    const { aId, bId } = useMemo(() => {
        const params = new URLSearchParams(location.search);
        const a = Number(params.get('a'));
        const b = Number(params.get('b'));
        return {
            aId: Number.isFinite(a) && a > 0 ? a : null,
            bId: Number.isFinite(b) && b > 0 ? b : null,
        };
    }, [location.search]);

    const load = useCallback(async () => {
        if (!aId || !bId) {
            setLoading(false);
            setError('Pass ?a=<exp> and ?b=<exp> in the URL to compare two experiments.');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const resp = await fetchExperimentCompare(projectId, aId, bId);
            setData(resp);
        } catch (err: any) {
            setError(
                err?.response?.data?.detail
                    || err?.message
                    || 'Failed to load comparison',
            );
        } finally {
            setLoading(false);
        }
    }, [projectId, aId, bId]);

    useEffect(() => {
        void load();
    }, [load]);

    const handleRerun = useCallback(async () => {
        if (!data) return;
        const winnerId = data.winner === 'a'
            ? data.a.experiment_id
            : data.winner === 'b'
                ? data.b.experiment_id
                : data.a.experiment_id;  // tie/unknown → default to A's config
        setRerunning(true);
        try {
            const exp = await rerunExperimentFromManifest(projectId, winnerId, {
                runName: `Fix-the-gap rollback from exp ${data.a.experiment_id} vs ${data.b.experiment_id}`,
                description: `Rerun of exp ${winnerId} via E3 comparison "Fix the gap" CTA.`,
            });
            toast.success(`Rollback launched as experiment #${exp.id}.`);
            navigate(`/project/${projectId}/training-config`);
        } catch (err: any) {
            const detail = err?.response?.data?.detail;
            if (detail === 'manifest_not_captured') {
                toast.error(
                    `Experiment #${winnerId} has no training manifest — only completed runs can be rolled back to.`,
                );
            } else {
                toast.error(detail || err?.message || 'Rerun failed.');
            }
        } finally {
            setRerunning(false);
        }
    }, [data, navigate, projectId]);

    if (loading) {
        return (
            <div className="workspace-page eval-compare" data-testid="eval-compare-loading">
                <p>Loading comparison…</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="workspace-page eval-compare" data-testid="eval-compare-error">
                <p className="eval-compare__error">{error}</p>
            </div>
        );
    }

    if (!data) return null;

    return (
        <div className="workspace-page eval-compare" data-testid="eval-compare">
            <header className="eval-compare__header">
                <h2 className="workspace-page-title">Experiment comparison</h2>
                <p className="workspace-page-subtitle">
                    Side-by-side eval of two experiments. Regressed signals + new failure
                    clusters bubble to the top; the "Fix the gap" button reruns the
                    winner's config when B is worse.
                </p>
            </header>

            <section className="eval-compare__summary" data-testid="eval-compare-summary">
                <SideCard label="A" side={data.a} winner={data.winner === 'a'} />
                <div className={`eval-compare__verdict eval-compare__verdict--${data.regressed ? 'regressed' : data.winner}`}>
                    <span className="eval-compare__verdict-label">
                        {data.winner === 'tie' && 'Tied'}
                        {data.winner === 'unknown' && 'No verdict'}
                        {data.winner === 'a' && (data.regressed ? 'B regressed' : 'A wins')}
                        {data.winner === 'b' && 'B wins'}
                    </span>
                </div>
                <SideCard label="B" side={data.b} winner={data.winner === 'b'} />
            </section>

            {data.regressed && (
                <section
                    className="eval-compare__fix-gap"
                    data-testid="eval-compare-fix-gap"
                >
                    <div>
                        <strong>Fix the gap.</strong> Experiment B's pass-rate is below A's.
                        Rerun A's config + dataset version to roll back to the working setup.
                    </div>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleRerun}
                        disabled={rerunning}
                        data-testid="eval-compare-rerun"
                    >
                        {rerunning ? 'Launching rerun…' : `Rerun experiment #${data.a.experiment_id}`}
                    </button>
                </section>
            )}

            <section className="eval-compare__metrics" data-testid="eval-compare-metrics">
                <h3>Metric deltas</h3>
                {data.metric_deltas.length === 0 ? (
                    <p className="eval-compare__empty">
                        Neither eval reported any metrics yet. Run an eval against each
                        experiment first.
                    </p>
                ) : (
                    <table className="eval-compare__table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>A</th>
                                <th>B</th>
                                <th>Δ</th>
                                <th>Direction</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.metric_deltas.map((row) => (
                                <tr
                                    key={row.metric_id}
                                    data-testid={`eval-compare-metric-${row.metric_id}`}
                                    data-direction={row.direction}
                                >
                                    <td><code>{row.metric_id}</code></td>
                                    <td>{formatNumeric(row.a_value)}</td>
                                    <td>{formatNumeric(row.b_value)}</td>
                                    <td>{formatDelta(row.delta, false)}</td>
                                    <td>
                                        <span className={`eval-compare__pill eval-compare__pill--${row.direction}`}>
                                            {row.direction}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </section>

            <section className="eval-compare__clusters" data-testid="eval-compare-clusters">
                <h3>Failure clusters</h3>
                <p className="eval-compare__cluster-totals">
                    A: <strong>{data.cluster_diff.a_total}</strong> failures grouped ·
                    {' '}
                    B: <strong>{data.cluster_diff.b_total}</strong> failures grouped.
                </p>
                <ClusterDiffBlock
                    title="New in B (regressions)"
                    rows={data.cluster_diff.only_in_b}
                    valueKey="failure_count"
                    tone="regressed"
                    testId="eval-compare-only-b"
                />
                <ClusterDiffBlock
                    title="Resolved in B (fixed)"
                    rows={data.cluster_diff.only_in_a}
                    valueKey="failure_count"
                    tone="improved"
                    testId="eval-compare-only-a"
                />
                <ClusterDiffBlock
                    title="Shared clusters"
                    rows={data.cluster_diff.shared}
                    valueKey="delta"
                    deltaMode
                    tone="neutral"
                    testId="eval-compare-shared"
                />
            </section>

            <section className="eval-compare__config" data-testid="eval-compare-config">
                <h3>Config diff</h3>
                <table className="eval-compare__table">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>A</th>
                            <th>B</th>
                            <th>Changed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.config_diff.map((row) => (
                            <ConfigDiffRowEl key={row.field} row={row} />
                        ))}
                    </tbody>
                </table>
            </section>
        </div>
    );
}


function SideCard({
    label, side, winner,
}: {
    label: string;
    side: CompareResponse['a'];
    winner: boolean;
}) {
    return (
        <div
            className={`eval-compare__card ${winner ? 'eval-compare__card--winner' : ''}`}
            data-testid={`eval-compare-side-${label.toLowerCase()}`}
        >
            <div className="eval-compare__card-label">
                {label}{winner ? ' ★' : ''}
            </div>
            <div className="eval-compare__card-name">{side.name}</div>
            <div className="eval-compare__card-meta">
                <span>{side.base_model}</span>
                <span>· {side.training_mode}</span>
            </div>
            <div className="eval-compare__card-pass">
                pass_rate:{' '}
                <strong>{formatPassRate(side.eval_pass_rate)}</strong>
            </div>
        </div>
    );
}


function ClusterDiffBlock({
    title,
    rows,
    valueKey,
    deltaMode = false,
    tone,
    testId,
}: {
    title: string;
    rows: any[];
    valueKey: string;
    deltaMode?: boolean;
    tone: 'regressed' | 'improved' | 'neutral';
    testId: string;
}) {
    if (rows.length === 0) {
        return (
            <div className={`eval-compare__cluster-block eval-compare__cluster-block--${tone}`} data-testid={testId}>
                <h4>{title}</h4>
                <p className="eval-compare__empty">None.</p>
            </div>
        );
    }
    return (
        <div className={`eval-compare__cluster-block eval-compare__cluster-block--${tone}`} data-testid={testId}>
            <h4>{title}</h4>
            <ul>
                {rows.map((row, idx) => {
                    const value = (row as Record<string, unknown>)[valueKey];
                    const numericValue = typeof value === 'number' ? value : 0;
                    const sign = deltaMode && numericValue > 0 ? '+' : '';
                    return (
                        <li
                            key={idx}
                            data-testid={`${testId}-row-${idx}`}
                            data-reason-code={row.reason_code}
                        >
                            <span className={`eval-compare__pill eval-compare__pill--reason-${row.reason_code}`}>
                                {row.reason_code}
                            </span>
                            <code>{row.output_pattern}</code>
                            <strong>
                                {sign}
                                {numericValue}
                                {deltaMode ? '' : ' failures'}
                            </strong>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}


function ConfigDiffRowEl({ row }: { row: ConfigDiffRow }) {
    return (
        <tr
            data-testid={`eval-compare-config-${row.field}`}
            data-changed={row.changed ? 'true' : 'false'}
        >
            <td><code>{row.field}</code>{row.primary ? null : <em> (other)</em>}</td>
            <td>{formatNumeric(row.a_value)}</td>
            <td>{formatNumeric(row.b_value)}</td>
            <td>{row.changed ? 'yes' : 'no'}</td>
        </tr>
    );
}
