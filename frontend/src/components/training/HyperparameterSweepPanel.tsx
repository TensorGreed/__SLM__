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
}

interface SweepResponse {
    sweep_id: string;
    cell_count: number;
    completed_count: number;
    cells: SweepCell[];
    best_label: string | null;
    best_experiment_id: number | null;
    cost_kind: CostKind;
    supported_cost_kinds: CostKind[];
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
    const [launching, setLaunching] = useState(false);
    const [error, setError] = useState('');
    const [sweepId, setSweepId] = useState('');
    const [sweep, setSweep] = useState<SweepResponse | null>(null);
    const [costKind, setCostKind] = useState<CostKind>('wall_clock_seconds');
    const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const cellEstimate = useMemo(() => {
        const ranks = parseNumbers(ranksText).length;
        const lrs = parseNumbers(lrsText).length;
        return ranks * lrs;
    }, [ranksText, lrsText]);

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
        try {
            const res = await api.post<{ sweep_id: string; dispatched_cells: number }>(
                `/projects/${projectId}/training/sweeps`,
                {
                    base_model: baseModel,
                    base_config: baseConfig || {},
                    lora_r_values: ranks,
                    learning_rate_values: lrs,
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

            <div className="hp-sweep__controls">
                <label className="hp-sweep__field">
                    LoRA ranks
                    <input className="input" value={ranksText} onChange={(e) => setRanksText(e.target.value)} placeholder="8, 16, 32" />
                </label>
                <label className="hp-sweep__field">
                    Learning rates
                    <input className="input" value={lrsText} onChange={(e) => setLrsText(e.target.value)} placeholder="2e-4, 3e-4" />
                </label>
                <button type="button" className="btn btn-primary" onClick={() => void launch()} disabled={launching}>
                    {launching ? 'Launching…' : `Run sweep (${cellEstimate} cells)`}
                </button>
            </div>

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
