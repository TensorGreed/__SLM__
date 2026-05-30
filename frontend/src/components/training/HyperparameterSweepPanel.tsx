/**
 * HyperparameterSweepPanel — Track 1, Epic C.
 *
 * Launches a LoRA-rank × learning-rate grid bake-off where each cell is a real
 * Experiment, then renders the cells on a quality-vs-cost Pareto scatter (the
 * backend annotates `pareto_optimal` / `dominated_by`, so this just renders).
 * Cells stream in as they finish training (poll), and the best completed cell
 * can be opened in the experiment view.
 *
 * The cost axis is user-pickable: wall-clock seconds (default, the honest one),
 * LoRA rank (adapter footprint proxy), or base-model parameter count (useful
 * when the sweep varies base_model). Switching the picker re-fetches; the
 * frontier annotation is recomputed against whichever axis the user chose.
 *
 * Self-contained: takes projectId + base model + a navigate callback as props.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api from '../../api/client';

type CostKind = 'wall_clock_seconds' | 'lora_r' | 'base_params_m';

interface SweepCell {
    label: string;
    experiment_id?: number;
    lora_r: number;
    learning_rate: number | null;
    base_model: string;
    status: string;
    final_train_loss: number | null;
    final_eval_loss: number | null;
    quality_score: number | null;
    quality_source?: string;
    cost_score: number | null;
    cost_source?: string;
    pareto_optimal?: boolean;
    dominated_by?: string[];
    cancelled_by_target?: boolean;
    // null = not measurable yet (no eval results / no pack);
    // true/false = pack ran and (cleared / didn't clear) the gate.
    gate_passed?: boolean | null;
    gate_failed_ids?: string[];
}

type Verdict = 'promote' | 'inconclusive' | 'pending';

interface SweepResponse {
    sweep_id: string;
    cell_count: number;
    completed_count: number;
    cells: SweepCell[];
    best_label: string | null;
    best_experiment_id: number | null;
    cost_kind: CostKind;
    supported_cost_kinds: CostKind[];
    quality_target?: number | null;
    target_hit?: boolean;
    target_hit_label?: string | null;
    cancelled_by_target?: string[];
    verdict?: Verdict;
    verdict_reason?: string;
    gate_summary?: {
        pack_id?: string | null;
        task_profile?: string | null;
        measurable_count?: number;
        any_cell_cleared?: boolean;
    };
}

interface PreflightBudget {
    cell_count: number;
    seconds_per_cell: number;
    estimated_seconds: number;
    basis: 'same_base_and_recipe' | 'same_base_model' | 'project_default' | 'no_history';
    sample_size: number;
}

interface SweepHistoryEntry {
    sweep_id: string;
    cell_count: number;
    requested_cells: number;
    base_model: string;
    recipe_id: string | null;
    quality_target: number | null;
    created_at: string | null;
    axes: { lora_r?: number[]; learning_rate?: number[]; base_model?: string[] } | null;
}

interface SweepHistoryResponse {
    project_id: number;
    sweep_count: number;
    sweeps: SweepHistoryEntry[];
}

interface HyperparameterSweepPanelProps {
    projectId: number;
    baseModel: string;
    baseConfig?: Record<string, unknown>;
    onOpenExperiment?: (experimentId: number) => void;
}

// Display metadata for each cost axis. Keep the axis label terse — the
// scatter is tight on horizontal space — but include the unit so the
// reader doesn't have to guess what "60" means.
const COST_AXIS_META: Record<CostKind, { label: string; unit: string; format: (n: number) => string }> = {
    wall_clock_seconds: {
        label: 'Wall-clock (seconds)',
        unit: 's',
        format: (n) => (n >= 60 ? `${(n / 60).toFixed(1)}m` : `${n.toFixed(0)}s`),
    },
    lora_r: {
        label: 'LoRA rank',
        unit: '',
        format: (n) => `r${n.toFixed(0)}`,
    },
    base_params_m: {
        label: 'Base params (M)',
        unit: 'M',
        format: (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}B` : `${n.toFixed(0)}M`),
    },
};

// Explanation chip — what each cost-source value means when cost_score is null.
// The backend sends honest reasons rather than guessing; surface them so a
// reader can tell "cell isn't finished yet" from "we don't know that model's
// size" without opening devtools.
const COST_SOURCE_HINT: Record<string, string> = {
    pending: 'cost not measured yet',
    missing_lora_r: 'cell has no LoRA rank',
    unknown_base_model: 'base model not in catalog',
    invalid: 'cost signal invalid',
};

function parseNumbers(text: string): number[] {
    return text
        .split(',')
        .map((t) => Number(t.trim()))
        .filter((n) => Number.isFinite(n) && n > 0);
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

// Render a seconds total as "12m", "1.3h", etc. Picks a precision that
// keeps the user honest about "this is an estimate, not a stopwatch
// reading" — single-decimal hours, integer minutes.
function formatDuration(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds <= 0) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

// Relative-time formatter for the history sidebar. Sweeps that
// completed an hour ago vs three days ago need different wording; the
// sidebar isn't the place for an exact timestamp tooltip. The browser's
// Intl.RelativeTimeFormat would be cleaner but it's overkill here.
function formatAgo(iso: string | null): string {
    if (!iso) return '—';
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return '—';
    const delta = (Date.now() - t) / 1000;
    if (delta < 60) return 'just now';
    if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
    return `${Math.round(delta / 86400)}d ago`;
}

// Human-friendly basis label. The "no_history" case is special — surface
// it as "rough" so the user knows the number is a default, not measured.
const BUDGET_BASIS_LABEL: Record<PreflightBudget['basis'], string> = {
    same_base_and_recipe: 'based on same base + recipe',
    same_base_model: 'based on same base model',
    project_default: 'based on this project’s sweeps',
    no_history: 'rough estimate, no prior runs',
};

// Coerce a quality-target string from the input. Accepts decimal (0.85)
// or percent (85). Returns null when the field is blank or unparseable,
// which is treated as "no target — run the full grid".
function parseQualityTarget(text: string): number | null {
    const trimmed = text.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    if (!Number.isFinite(n) || n <= 0) return null;
    if (n > 1.0 && n <= 100.0) return n / 100.0;
    if (n > 0 && n <= 1.0) return n;
    return null;
}

const W = 480;
const H = 280;
const PAD = { top: 18, right: 22, bottom: 42, left: 52 };

export default function HyperparameterSweepPanel({
    projectId,
    baseModel,
    baseConfig,
    onOpenExperiment,
}: HyperparameterSweepPanelProps) {
    const [ranksText, setRanksText] = useState('8, 16');
    const [lrsText, setLrsText] = useState('2e-4, 3e-4');
    const [qualityTargetText, setQualityTargetText] = useState('');
    const [launching, setLaunching] = useState(false);
    const [error, setError] = useState('');
    const [sweepId, setSweepId] = useState('');
    const [sweep, setSweep] = useState<SweepResponse | null>(null);
    const [costKind, setCostKind] = useState<CostKind>('wall_clock_seconds');
    const [budget, setBudget] = useState<PreflightBudget | null>(null);
    const [history, setHistory] = useState<SweepHistoryEntry[]>([]);
    const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // History sidebar — list past sweeps for the project. Re-fetches
    // whenever the current sweepId changes (so launching a new sweep
    // pulls it into the list) and on initial mount. Cheap call —
    // backend hits the Sweep table directly.
    const refreshHistory = useCallback(async () => {
        try {
            const res = await api.get<SweepHistoryResponse>(
                `/projects/${projectId}/training/sweeps`,
            );
            setHistory(res.data?.sweeps || []);
        } catch {
            // History sidebar is non-critical — silent failure is fine.
            setHistory([]);
        }
    }, [projectId]);
    useEffect(() => {
        void refreshHistory();
    }, [refreshHistory, sweepId]);

    const cellEstimate = useMemo(() => {
        const ranks = parseNumbers(ranksText).length;
        const lrs = parseNumbers(lrsText).length;
        return ranks * lrs;
    }, [ranksText, lrsText]);

    // Pre-flight budget — refetch whenever the planned grid shape changes.
    // Debounced so a user typing "8, 16, 32" doesn't fire three requests
    // mid-typing. 500ms is long enough to avoid the typing case and short
    // enough that the chip feels live when they paste a value.
    useEffect(() => {
        const ranks = parseNumbers(ranksText).map((n) => Math.round(n));
        const lrs = parseNumbers(lrsText);
        if (!ranks.length || !lrs.length) {
            setBudget(null);
            return;
        }
        let cancelled = false;
        const handle = setTimeout(async () => {
            try {
                const res = await api.post<PreflightBudget>(
                    `/projects/${projectId}/training/sweeps/preflight-budget`,
                    {
                        base_model: baseModel,
                        lora_r_values: ranks,
                        learning_rate_values: lrs,
                    },
                );
                if (!cancelled) setBudget(res.data);
            } catch {
                if (!cancelled) setBudget(null);
            }
        }, 500);
        return () => {
            cancelled = true;
            clearTimeout(handle);
        };
    }, [ranksText, lrsText, baseModel, projectId]);

    const fetchSweep = useCallback(async (id: string, kind: CostKind) => {
        try {
            const res = await api.get<SweepResponse>(
                `/projects/${projectId}/training/sweeps/${id}`,
                { params: { cost_kind: kind } },
            );
            setSweep(res.data);
            return res.data;
        } catch (err) {
            setError(errorDetail(err, 'Failed to load sweep results.'));
            return null;
        }
    }, [projectId]);

    // Poll while any cell is still training. Re-runs whenever costKind
    // changes — switching the axis forces a fresh fetch so the frontier
    // matches what's being plotted.
    useEffect(() => {
        if (!sweepId) return;
        let cancelled = false;
        const tick = async () => {
            const data = await fetchSweep(sweepId, costKind);
            if (cancelled || !data) return;
            if (data.completed_count < data.cell_count) {
                pollRef.current = setTimeout(tick, 4000);
            }
        };
        void tick();
        return () => {
            cancelled = true;
            if (pollRef.current) clearTimeout(pollRef.current);
        };
    }, [sweepId, costKind, fetchSweep]);

    const launch = async () => {
        const ranks = parseNumbers(ranksText).map((n) => Math.round(n));
        const lrs = parseNumbers(lrsText);
        if (!ranks.length || !lrs.length) {
            setError('Provide at least one LoRA rank and one learning rate.');
            return;
        }
        setLaunching(true);
        setError('');
        setSweep(null);
        const qualityTarget = parseQualityTarget(qualityTargetText);
        try {
            const res = await api.post<{ sweep_id: string; dispatched_cells: number }>(
                `/projects/${projectId}/training/sweeps`,
                {
                    base_model: baseModel,
                    base_config: baseConfig || {},
                    lora_r_values: ranks,
                    learning_rate_values: lrs,
                    quality_target: qualityTarget,
                },
            );
            setSweepId(res.data.sweep_id);
        } catch (err) {
            setError(errorDetail(err, 'Failed to launch sweep.'));
        } finally {
            setLaunching(false);
        }
    };

    const cells = sweep?.cells || [];
    const axisMeta = COST_AXIS_META[costKind];

    // Only cells with BOTH quality AND cost can land on the 2D plot.
    // Cells missing either signal sit out — surfaced in the list below
    // with their cost_source explanation rather than fabricated.
    const plottable = useMemo(
        () => cells.filter((c) => c.quality_score != null && c.cost_score != null),
        [cells],
    );

    const scales = useMemo(() => {
        const costs = plottable.map((c) => c.cost_score as number);
        const quals = plottable.map((c) => c.quality_score as number);
        const cMin = costs.length ? Math.min(...costs) : 0;
        const cMax = costs.length ? Math.max(...costs) : 1;
        const qMin = quals.length ? Math.min(...quals) : 0;
        const qMax = quals.length ? Math.max(...quals) : 1;
        const cSpan = cMax - cMin || 1;
        const qSpan = qMax - qMin || 1;
        return {
            x: (c: number) => PAD.left + ((c - cMin) / cSpan) * (W - PAD.left - PAD.right),
            y: (q: number) => PAD.top + (1 - (q - qMin) / qSpan) * (H - PAD.top - PAD.bottom),
        };
    }, [plottable]);

    const frontierPath = useMemo(() => {
        const pts = plottable
            .filter((c) => c.pareto_optimal)
            .map((c) => ({ cx: scales.x(c.cost_score as number), cy: scales.y(c.quality_score as number) }))
            .sort((a, b) => a.cx - b.cx);
        if (pts.length < 2) return '';
        return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`).join(' ');
    }, [plottable, scales]);

    return (
        <div className="hp-sweep" data-testid="hp-sweep">
            <div className="hp-sweep__head">
                <strong>Hyperparameter bake-off</strong>
                <span className="hp-sweep__hint">
                    Each cell is a real Experiment on <code>{baseModel}</code>. Pareto = quality vs {axisMeta.label.toLowerCase()}.
                </span>
            </div>

            {history.length > 0 && (
                <div
                    className="hp-sweep__history"
                    data-testid="hp-history"
                    aria-label="Past sweeps for this project"
                >
                    <span className="hp-sweep__history-label">Past sweeps:</span>
                    <ul className="hp-sweep__history-list">
                        {history.slice(0, 5).map((entry) => {
                            const isCurrent = entry.sweep_id === sweepId;
                            return (
                                <li
                                    key={entry.sweep_id}
                                    className={`hp-sweep__history-item ${isCurrent ? 'is-current' : ''}`}
                                >
                                    <button
                                        type="button"
                                        className="hp-sweep__history-button"
                                        data-testid={`hp-history-${entry.sweep_id}`}
                                        data-current={isCurrent ? 'true' : 'false'}
                                        onClick={() => setSweepId(entry.sweep_id)}
                                        title={
                                            entry.created_at
                                                ? `${entry.sweep_id} · ${entry.base_model}`
                                                : `${entry.sweep_id} · legacy sweep`
                                        }
                                    >
                                        <span className="hp-sweep__history-id">{entry.sweep_id.slice(0, 8)}</span>
                                        <span className="hp-sweep__history-meta">
                                            {entry.cell_count}/{entry.requested_cells || entry.cell_count} cells · {formatAgo(entry.created_at)}
                                            {entry.quality_target != null && (
                                                <> · target {(entry.quality_target * 100).toFixed(0)}%</>
                                            )}
                                        </span>
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}

            <div className="hp-sweep__controls">
                <label className="hp-sweep__field">
                    LoRA ranks
                    <input className="input" value={ranksText} onChange={(e) => setRanksText(e.target.value)} placeholder="8, 16, 32" />
                </label>
                <label className="hp-sweep__field">
                    Learning rates
                    <input className="input" value={lrsText} onChange={(e) => setLrsText(e.target.value)} placeholder="2e-4, 3e-4" />
                </label>
                <label className="hp-sweep__field" title="Stop the rest of the sweep once any cell's eval pass-rate clears this. Blank = run every cell to completion.">
                    Quality target
                    <input
                        className="input"
                        value={qualityTargetText}
                        onChange={(e) => setQualityTargetText(e.target.value)}
                        placeholder="0.85"
                        data-testid="hp-quality-target"
                    />
                </label>
                <button type="button" className="btn btn-primary" onClick={() => void launch()} disabled={launching}>
                    {launching ? 'Launching…' : `Run sweep (${cellEstimate} cells)`}
                </button>
            </div>

            {budget && (
                <div className="hp-sweep__preflight" data-testid="hp-preflight">
                    <span className="hp-sweep__preflight-headline">
                        Estimated runtime: <strong>{formatDuration(budget.estimated_seconds)}</strong>
                        {' '}for {budget.cell_count} cell{budget.cell_count === 1 ? '' : 's'}
                    </span>
                    <span className="hp-sweep__preflight-basis">
                        {budget.basis === 'no_history'
                            ? BUDGET_BASIS_LABEL[budget.basis]
                            : `${BUDGET_BASIS_LABEL[budget.basis]} (${budget.sample_size} prior cell${budget.sample_size === 1 ? '' : 's'})`}
                        {' · ~'}{formatDuration(budget.seconds_per_cell)}{' per cell'}
                    </span>
                </div>
            )}

            {error && <div className="hp-sweep__error">{error}</div>}

            {sweep && (
                <>
                    <div className="hp-sweep__status">
                        {sweep.completed_count}/{sweep.cell_count} cells complete
                        {sweep.completed_count < sweep.cell_count && <span className="hp-sweep__spinner"> · training…</span>}
                        {sweep.best_label && (
                            <span className="hp-sweep__best"> · best: <strong>{sweep.best_label}</strong></span>
                        )}
                    </div>

                    {sweep.target_hit && sweep.target_hit_label && (
                        <div className="hp-sweep__target-hit" data-testid="hp-target-hit">
                            Target {(sweep.quality_target != null) ? `${(sweep.quality_target * 100).toFixed(0)}%` : 'reached'}
                            {' · winner: '}
                            <strong>{sweep.target_hit_label}</strong>
                            {sweep.cancelled_by_target && sweep.cancelled_by_target.length > 0 && (
                                <span className="hp-sweep__target-hit-cancelled">
                                    {' · cancelled '}{sweep.cancelled_by_target.length}{' remaining cell'}
                                    {sweep.cancelled_by_target.length === 1 ? '' : 's'}
                                </span>
                            )}
                        </div>
                    )}

                    {sweep.verdict && (
                        <div
                            className={`hp-sweep__verdict hp-sweep__verdict--${sweep.verdict}`}
                            data-testid="hp-verdict"
                            data-verdict={sweep.verdict}
                        >
                            <strong className="hp-sweep__verdict-label">
                                {sweep.verdict === 'promote' && '✓ Winner cleared the gate'}
                                {sweep.verdict === 'inconclusive' && 'Inconclusive — nobody cleared the gate'}
                                {sweep.verdict === 'pending' && 'Gate verdict pending'}
                            </strong>
                            {sweep.gate_summary?.pack_id && (
                                <span className="hp-sweep__verdict-pack">
                                    {' · '}{sweep.gate_summary.pack_id}
                                </span>
                            )}
                            {sweep.verdict_reason && (
                                <div className="hp-sweep__verdict-reason">{sweep.verdict_reason}</div>
                            )}
                            {sweep.verdict === 'inconclusive' && sweep.gate_summary?.measurable_count !== 0 && (
                                <div className="hp-sweep__verdict-handoff">
                                    Open the <a className="inline-link" href="#failure-clusters">Failure clusters</a> panel to see why each cell missed.
                                </div>
                            )}
                        </div>
                    )}

                    <div
                        className="hp-sweep__cost-picker"
                        aria-label={`Cost axis — currently ${axisMeta.label}`}
                        data-testid="hp-cost-picker"
                    >
                        <span className="hp-sweep__cost-picker-label">Cost axis:</span>
                        {(sweep.supported_cost_kinds || []).map((kind) => {
                            const selected = costKind === kind;
                            return (
                                <button
                                    key={kind}
                                    type="button"
                                    className={`hp-sweep__cost-option ${selected ? 'is-selected' : ''}`}
                                    data-testid={`hp-cost-option-${kind}`}
                                    data-selected={selected ? 'true' : 'false'}
                                    aria-label={`Cost axis: ${COST_AXIS_META[kind].label}${selected ? ' (selected)' : ''}`}
                                    onClick={() => setCostKind(kind)}
                                >
                                    {COST_AXIS_META[kind].label}
                                </button>
                            );
                        })}
                    </div>

                    <svg
                        className="hp-sweep__chart"
                        viewBox={`0 0 ${W} ${H}`}
                        role="img"
                        aria-label={`Quality versus ${axisMeta.label} Pareto scatter`}
                    >
                        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} className="hp-sweep__axis" />
                        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} className="hp-sweep__axis" />
                        <text
                            x={(PAD.left + W - PAD.right) / 2}
                            y={H - 8}
                            className="hp-sweep__axis-label"
                            textAnchor="middle"
                            data-testid="hp-x-axis-label"
                        >
                            {axisMeta.label} — smaller is cheaper →
                        </text>
                        <text x={13} y={(PAD.top + H - PAD.bottom) / 2} className="hp-sweep__axis-label" textAnchor="middle" transform={`rotate(-90 13 ${(PAD.top + H - PAD.bottom) / 2})`}>
                            Quality — higher ↑
                        </text>
                        {frontierPath && <path d={frontierPath} className="hp-sweep__frontier" fill="none" />}
                        {plottable.map((c) => (
                            <circle
                                key={c.label}
                                cx={scales.x(c.cost_score as number)}
                                cy={scales.y(c.quality_score as number)}
                                r={c.label === sweep.best_label ? 8 : 6}
                                className={`hp-sweep__point ${c.pareto_optimal ? 'is-optimal' : 'is-dominated'}`}
                                data-testid={`hp-point-${c.label}`}
                            >
                                <title>{`${c.label} — quality ${(c.quality_score as number).toFixed(3)}, ${axisMeta.label.toLowerCase()} ${axisMeta.format(c.cost_score as number)}${c.pareto_optimal ? ' — Pareto-optimal' : ''}`}</title>
                            </circle>
                        ))}
                    </svg>

                    <ul className="hp-sweep__list">
                        {cells.map((c) => {
                            const costPending = c.cost_score == null;
                            const costHint = costPending && c.cost_source
                                ? (COST_SOURCE_HINT[c.cost_source] || c.cost_source)
                                : null;
                            return (
                                <li
                                    key={c.label}
                                    data-testid={`hp-row-${c.label}`}
                                    className={`hp-sweep__row ${c.pareto_optimal ? 'is-optimal' : ''} ${c.label === sweep.best_label ? 'is-best' : ''}`}
                                    onClick={() => c.experiment_id && onOpenExperiment?.(c.experiment_id)}
                                >
                                    <span className="hp-sweep__row-name">
                                        {c.label}
                                        {c.pareto_optimal && <span className="hp-sweep__badge">frontier</span>}
                                        {c.label === sweep.best_label && <span className="hp-sweep__badge hp-sweep__badge--best">best</span>}
                                        {c.cancelled_by_target && (
                                            <span
                                                className="hp-sweep__badge hp-sweep__badge--cancelled"
                                                data-testid={`hp-row-${c.label}-cancelled`}
                                                title="Stopped early because another cell hit the quality target"
                                            >
                                                cancelled · target hit
                                            </span>
                                        )}
                                        {c.gate_passed === true && (
                                            <span
                                                className="hp-sweep__badge hp-sweep__badge--gate-pass"
                                                data-testid={`hp-row-${c.label}-gate-pass`}
                                                title="Cleared the project's evaluation gate"
                                            >
                                                gate ✓
                                            </span>
                                        )}
                                        {c.gate_passed === false && (
                                            <span
                                                className="hp-sweep__badge hp-sweep__badge--gate-fail"
                                                data-testid={`hp-row-${c.label}-gate-fail`}
                                                title={
                                                    c.gate_failed_ids && c.gate_failed_ids.length
                                                        ? `Failed: ${c.gate_failed_ids.join(', ')}`
                                                        : 'Did not clear the project gate'
                                                }
                                            >
                                                gate ✗{c.gate_failed_ids && c.gate_failed_ids.length
                                                    ? ` · ${c.gate_failed_ids[0]}${c.gate_failed_ids.length > 1 ? '…' : ''}`
                                                    : ''}
                                            </span>
                                        )}
                                    </span>
                                    <span className="hp-sweep__row-metrics">
                                        {c.status === 'completed' ? (
                                            <>
                                                {`loss ${c.final_train_loss != null ? c.final_train_loss.toFixed(3) : '—'}`}
                                                {' · '}
                                                {`q ${c.quality_score != null ? c.quality_score.toFixed(3) : '—'}`}
                                                {' · '}
                                                {costPending
                                                    ? <span className="hp-sweep__cost-pending" title={costHint || undefined}>cost pending</span>
                                                    : `${axisMeta.label.toLowerCase()} ${axisMeta.format(c.cost_score as number)}`}
                                            </>
                                        ) : c.status}
                                    </span>
                                </li>
                            );
                        })}
                    </ul>
                </>
            )}
        </div>
    );
}
