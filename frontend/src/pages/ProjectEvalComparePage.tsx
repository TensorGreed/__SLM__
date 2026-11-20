/**
 * Side-by-side eval comparison page (E3).
 *
 * Rendered in Results → Compare (``/project/:projectId/results?tab=compare&a=<exp>&b=<exp>``).
 * Renders:
 *   - Row-paired evidence (Epic W1): is B's change more than noise? Rows
 *     better / worse / same, 95% range, verdict — or why A and B are not
 *     comparable (answer key / judge / scoring) with a re-score offer.
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
import { Star } from 'lucide-react';
import LoadingState from '../components/shared/LoadingState';
import CompareRowEvidence from '../components/evaluation/CompareRowEvidence';
import { useLocation, useNavigate, useOutletContext } from 'react-router-dom';

import type {
    CompareResponse,
    ComparePin,
    ConfigDiffRow,
} from '../api/experimentCompare';
import {
    fetchExperimentCompare,
    rerunExperimentFromManifest,
    verdictLabel,
} from '../api/experimentCompare';
import { toast } from '../stores/toastStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import { trainSectionPath } from '../utils/workspaceRoutes';
import { crossVariantBaseLabels, evalDatasetLabel, evaluatedSetLabels, listEvalSets } from '../api/evalSets';
import type { EvalSetLastEval } from '../api/evalSets';
import { EXACT_MATCH_ALIAS_MODES } from '../components/evaluation/verdictScore';
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


interface ProjectEvalComparePageProps {
    /** Rendered inside Results → Compare: no page header. */
    embedded?: boolean;
}

export default function ProjectEvalComparePage({ embedded = false }: ProjectEvalComparePageProps = {}) {
    const navigate = useNavigate();
    const location = useLocation();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const [data, setData] = useState<CompareResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [rerunning, setRerunning] = useState(false);
    // Optional: compare on one eval set (``?set=eval_set:<slug>@v<N>[+<variant>]``); default = built-in sets.
    const [evalOn, setEvalOn] = useState<string>(() => new URLSearchParams(location.search).get('set') ?? '');
    const [setLabels, setSetLabels] = useState<string[]>([]);
    const [lastEvals, setLastEvals] = useState<Record<string, EvalSetLastEval>>({});
    // Opt-in, off by default: the base model's eval on another prompt-format variant of the same version.
    const [baseOn, setBaseOn] = useState<string>('');

    const { aId, bId } = useMemo(() => {
        const params = new URLSearchParams(location.search);
        const a = Number(params.get('a'));
        const b = Number(params.get('b'));
        return {
            aId: Number.isFinite(a) && a > 0 ? a : null,
            bId: Number.isFinite(b) && b > 0 ? b : null,
        };
    }, [location.search]);
    // ``?eval_a=&eval_b=`` (the Verdict card's "Open side by side"): the exact two results its lift line
    // cites. Picking another set in the picker drops the pin.
    const [pin, setPin] = useState<ComparePin | null>(() => {
        const params = new URLSearchParams(location.search);
        const ea = Number(params.get('eval_a'));
        const eb = Number(params.get('eval_b'));
        const evalA = Number.isFinite(ea) && ea > 0 ? ea : null;
        const evalB = Number.isFinite(eb) && eb > 0 ? eb : null;
        return evalA || evalB ? { evalA, evalB, crossVariant: params.get('cross') === '1' } : null;
    });

    useEffect(() => {
        if (!aId || !bId) return undefined;
        let alive = true;
        listEvalSets(projectId)
            .then((sets) => {
                if (!alive) return;
                setSetLabels(evaluatedSetLabels(sets));
                setLastEvals(Object.assign({}, ...sets.map((s) => s.last_evals ?? {})));
            })
            .catch(() => { if (alive) { setSetLabels([]); setLastEvals({}); } });
        return () => { alive = false; };
    }, [projectId, aId, bId]);

    const load = useCallback(async () => {
        if (!aId || !bId) {
            setLoading(false);
            setError('Pass ?a=<exp> and ?b=<exp> in the URL to compare two experiments.');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const resp = pin
                ? await fetchExperimentCompare(projectId, aId, bId, evalOn || null, null, pin)
                : evalOn && baseOn
                    ? await fetchExperimentCompare(projectId, aId, bId, evalOn, baseOn)
                    : evalOn
                        ? await fetchExperimentCompare(projectId, aId, bId, evalOn)
                        : await fetchExperimentCompare(projectId, aId, bId);
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
    }, [projectId, aId, bId, evalOn, baseOn, pin]);

    useEffect(() => {
        void load();
    }, [load]);

    // Drop the pin (also from the URL, so a reload does not re-pin): each run's newest result on the picked set.
    const unpin = useCallback(() => {
        setPin(null);
        const params = new URLSearchParams(location.search);
        params.delete('eval_a');
        params.delete('eval_b');
        params.delete('cross');
        navigate({ pathname: location.pathname, search: params.toString() ? `?${params.toString()}` : '' }, { replace: true });
    }, [location.pathname, location.search, navigate]);

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
            navigate(trainSectionPath(projectId, 'plan'));
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
                <LoadingState label="Loading comparison" lines={3} card />
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

    // A cross-variant pair (pinned with ``cross=1`` or picked as "Base model on"): the model AND the prompt differ.
    const crossVariant = data.row_evidence?.cross_variant ?? data.row_evidence?.comparability?.cross_variant ?? null;

    // O2: in the modes where ``exact_match`` stores the mode's own pass rate under its legacy name (tool call,
    // structured plan, multiple options, remediation), that row moves under "Other stored metrics", still one
    // click away. Elsewhere (spans, findings, JSON fields) ``exact_match`` means "every span / field right on
    // the row" and stays in place.
    const taskMode = [data.a.provenance?.scoring?.scoring_mode, data.b.provenance?.scoring?.scoring_mode]
        .find((mode) => !!mode && EXACT_MATCH_ALIAS_MODES.includes(mode)) ?? null;
    const asideDeltas = taskMode ? data.metric_deltas.filter((row) => row.metric_id === 'exact_match') : [];
    const mainDeltas = data.metric_deltas.filter((row) => !asideDeltas.includes(row));
    // O1: on a base vs fine-tuned pair (A = base model) the cluster diff reads as the fine-tune's effect.
    const onlyInBTitle = data.lift ? 'Fails only with the fine-tune' : 'New in B (regressions)';
    const onlyInATitle = data.lift ? 'Fixed by the fine-tune' : 'Resolved in B (fixed)';

    return (
        <div className="workspace-page eval-compare" data-testid="eval-compare">
            {!embedded && (
            <header className="eval-compare__header">
                <h2 className="workspace-page-title">Run comparison</h2>
                <p className="workspace-page-subtitle">
                    Side-by-side eval of two experiments. Regressed signals + new failure
                    clusters bubble to the top; the "Fix the gap" button reruns the
                    winner's config when B is worse.
                </p>
            </header>
            )}

            {(setLabels.length > 0 || evalOn) && (
                <div className="eval-compare__evaluated-on" data-testid="eval-compare-set-picker">
                    <label htmlFor="eval-compare-set">Evaluated on</label>
                    <select
                        id="eval-compare-set"
                        className="input u-w-auto"
                        value={evalOn}
                        onChange={(e) => { setEvalOn(e.target.value); setBaseOn(''); setPin(null); }}
                        data-testid="eval-compare-set-select"
                    >
                        <option value="">Newest eval on a built-in set</option>
                        {[...new Set([...setLabels, ...(evalOn ? [evalOn] : [])])].map((label) => (
                            <option key={label} value={label}>{evalDatasetLabel(label)}</option>
                        ))}
                    </select>
                    {/* Hidden while a result pin is active: the pin decides both sides, so this choice would be ignored. */}
                    {!pin && data.lift && evalOn && crossVariantBaseLabels(lastEvals, evalOn).length > 0 && (
                        <label className="u-meta" data-testid="eval-compare-base-variant">
                            Base model on{' '}
                            <select className="input u-w-auto" value={baseOn || evalOn}
                                aria-label="Base model's prompt format"
                                data-testid="eval-compare-base-variant-select"
                                onChange={(e) => setBaseOn(e.target.value === evalOn ? '' : e.target.value)}>
                                <option value={evalOn}>the same prompt format</option>
                                {crossVariantBaseLabels(lastEvals, evalOn).map((o) => (
                                    <option key={o} value={o}>{evalDatasetLabel(o)} (prompt differs too)</option>
                                ))}
                            </select>
                        </label>
                    )}
                </div>
            )}

            {pin && (
                <p className="u-meta" data-testid="eval-compare-pinned">
                    Comparing exactly result #{pin.evalA ?? 'newest'} (A) and result #{pin.evalB ?? 'newest'} (B), the two the Verdict
                    card cites{crossVariant ? `, on two prompt formats of one set version (${crossVariant.label})` : ''}
                    {' · '}
                    <button type="button" className="btn btn-link" onClick={unpin} data-testid="eval-compare-unpin">
                        unpin
                    </button>
                    {' '}to compare each run&apos;s newest result{evalOn ? ` on ${evalDatasetLabel(evalOn)}` : ' on a built-in set'} instead.
                </p>
            )}

            {data.row_evidence && (
                <CompareRowEvidence
                    projectId={projectId}
                    evidence={data.row_evidence}
                    a={data.a.provenance}
                    b={data.b.provenance}
                />
            )}

            {data.lift && (
                <p className="eval-compare__lift-note" data-testid="eval-compare-lift">
                    Fine-tuned vs base model: A is the base model (baseline run #{data.a.experiment_id}), B the fine-tuned
                    run #{data.b.experiment_id}, so B above A is the fine-tune&apos;s lift
                    {crossVariant ? ` together with the prompt-format change (${crossVariant.label}): the model and the prompt both differ` : ''}.
                    {data.lift.swapped ? ' The two were picked the other way round; the base model is shown on A.' : ''}
                </p>
            )}

            <section className="eval-compare__summary" data-testid="eval-compare-summary">
                <SideCard label="A" side={data.a} winner={data.winner === 'a'} role={data.lift ? 'base model' : undefined} />
                <div className={`eval-compare__verdict eval-compare__verdict--${data.regressed ? 'regressed' : data.winner}`}>
                    <span className="eval-compare__verdict-label" data-testid="eval-compare-verdict">
                        {verdictLabel(data)}
                    </span>
                </div>
                <SideCard label="B" side={data.b} winner={data.winner === 'b'} role={data.lift ? 'fine-tuned' : undefined} />
            </section>

            {data.regressed && (
                <section
                    className="eval-compare__fix-gap"
                    data-testid="eval-compare-fix-gap"
                >
                    <div>
                        <strong>Fix the gap.</strong> Run B's pass-rate is below A's.
                        Rerun A's config + dataset version to roll back to the working setup.
                    </div>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleRerun}
                        disabled={rerunning}
                        data-testid="eval-compare-rerun"
                    >
                        {rerunning ? 'Launching rerun…' : `Rerun run #${data.a.experiment_id}'s config`}
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
                            {mainDeltas.map((row) => (
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
                {asideDeltas.length > 0 && (
                    <details className="eval-compare__aside" data-testid="eval-compare-string-match">
                        <summary>Other stored metrics (same pass, legacy name) ({asideDeltas.length})</summary>
                        <p className="eval-compare__empty">
                            Scored as {taskMode?.replace(/_/g, ' ')}: <code>exact_match</code> stores the mode's own pass rate
                            under its legacy name; the rows above are the score of record.
                        </p>
                        <table className="eval-compare__table">
                            <tbody>
                                {asideDeltas.map((row) => (
                                    <tr key={row.metric_id} data-testid={`eval-compare-metric-${row.metric_id}`} data-direction={row.direction}>
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
                    </details>
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
                    title={onlyInBTitle}
                    rows={data.cluster_diff.only_in_b}
                    valueKey="failure_count"
                    tone="regressed"
                    testId="eval-compare-only-b"
                />
                <ClusterDiffBlock
                    title={onlyInATitle}
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
    label, side, winner, role,
}: {
    label: string;
    side: CompareResponse['a'];
    winner: boolean;
    /** "base model" / "fine-tuned" on a base vs fine-tuned pair. */
    role?: string;
}) {
    return (
        <div
            className={`eval-compare__card ${winner ? 'eval-compare__card--winner' : ''}`}
            data-testid={`eval-compare-side-${label.toLowerCase()}`}
        >
            <div className="eval-compare__card-label">
                {label}{role ? ` · ${role}` : ''}{winner && <> <Star className="u-icon" aria-label="winner" /></>}
            </div>
            <div className="eval-compare__card-name">{side.name}</div>
            <div className="eval-compare__card-meta">
                <span>{side.base_model}</span>
                <span>· {side.training_mode}</span>
            </div>
            {side.dataset_name && (
                <div className="eval-compare__card-meta" data-testid={`eval-compare-dataset-${label.toLowerCase()}`}>
                    <span>
                        {evalDatasetLabel(side.dataset_name)}
                        {side.eval_result_id != null ? ` · result #${side.eval_result_id}` : ''}
                        {typeof side.provenance?.eval_set?.rows === 'number' ? ` · ${side.provenance.eval_set.rows} rows` : ''}
                    </span>
                </div>
            )}
            {side.provenance?.scoring && (
                <div className="eval-compare__card-meta" data-testid={`eval-compare-format-${label.toLowerCase()}`}>
                    <span>prompt rendering v{side.provenance.scoring.prompt_format_version ?? 1}</span>
                    {side.provenance.scoring.null_as_absent && <span> · null counts as absent</span>}
                </div>
            )}
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
