/**
 * HyperparameterSweepPanel — Track 1, Epic C.
 *
 * Launches a LoRA-rank × learning-rate grid bake-off where each cell is a real
 * Experiment, then renders the cells on a quality-vs-rank Pareto scatter (the
 * backend annotates `pareto_optimal` / `dominated_by`, so this just renders).
 * Cells stream in as they finish training (poll), and the best completed cell
 * can be opened in the experiment view.
 *
 * Self-contained: takes projectId + base model + a navigate callback as props.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api from '../../api/client';

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
}

interface HyperparameterSweepPanelProps {
    projectId: number;
    baseModel: string;
    baseConfig?: Record<string, unknown>;
    onOpenExperiment?: (experimentId: number) => void;
}

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
    const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const cellEstimate = useMemo(() => {
        const ranks = parseNumbers(ranksText).length;
        const lrs = parseNumbers(lrsText).length;
        return ranks * lrs;
    }, [ranksText, lrsText]);

    const fetchSweep = useCallback(async (id: string) => {
        try {
            const res = await api.get<SweepResponse>(`/projects/${projectId}/training/sweeps/${id}`);
            setSweep(res.data);
            return res.data;
        } catch (err) {
            setError(errorDetail(err, 'Failed to load sweep results.'));
            return null;
        }
    }, [projectId]);

    // Poll while any cell is still training.
    useEffect(() => {
        if (!sweepId) return;
        let cancelled = false;
        const tick = async () => {
            const data = await fetchSweep(sweepId);
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
    }, [sweepId, fetchSweep]);

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
    const scales = useMemo(() => {
        const scored = cells.filter((c) => c.quality_score != null);
        const ranks = cells.map((c) => c.lora_r);
        const quals = scored.map((c) => c.quality_score as number);
        const rMin = Math.min(...ranks, 0);
        const rMax = Math.max(...ranks, 1);
        const qMin = Math.min(...quals, 0);
        const qMax = Math.max(...quals, 1);
        const rSpan = rMax - rMin || 1;
        const qSpan = qMax - qMin || 1;
        return {
            x: (r: number) => PAD.left + ((r - rMin) / rSpan) * (W - PAD.left - PAD.right),
            y: (q: number) => PAD.top + (1 - (q - qMin) / qSpan) * (H - PAD.top - PAD.bottom),
        };
    }, [cells]);

    const frontierPath = useMemo(() => {
        const pts = cells
            .filter((c) => c.pareto_optimal && c.quality_score != null)
            .map((c) => ({ cx: scales.x(c.lora_r), cy: scales.y(c.quality_score as number) }))
            .sort((a, b) => a.cx - b.cx);
        if (pts.length < 2) return '';
        return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`).join(' ');
    }, [cells, scales]);

    return (
        <div className="hp-sweep" data-testid="hp-sweep">
            <div className="hp-sweep__head">
                <strong>Hyperparameter bake-off</strong>
                <span className="hp-sweep__hint">
                    Each cell is a real Experiment on <code>{baseModel}</code>. Pareto = quality vs LoRA rank.
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

                    <svg className="hp-sweep__chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Quality versus LoRA rank Pareto scatter">
                        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} className="hp-sweep__axis" />
                        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} className="hp-sweep__axis" />
                        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="hp-sweep__axis-label" textAnchor="middle">
                            LoRA rank — smaller is cheaper →
                        </text>
                        <text x={13} y={(PAD.top + H - PAD.bottom) / 2} className="hp-sweep__axis-label" textAnchor="middle" transform={`rotate(-90 13 ${(PAD.top + H - PAD.bottom) / 2})`}>
                            Quality — higher ↑
                        </text>
                        {frontierPath && <path d={frontierPath} className="hp-sweep__frontier" fill="none" />}
                        {cells.filter((c) => c.quality_score != null).map((c) => (
                            <circle
                                key={c.label}
                                cx={scales.x(c.lora_r)}
                                cy={scales.y(c.quality_score as number)}
                                r={c.label === sweep.best_label ? 8 : 6}
                                className={`hp-sweep__point ${c.pareto_optimal ? 'is-optimal' : 'is-dominated'}`}
                                data-testid={`hp-point-${c.label}`}
                            >
                                <title>{`${c.label} — quality ${(c.quality_score as number).toFixed(3)}, rank ${c.lora_r}${c.pareto_optimal ? ' — Pareto-optimal' : ''}`}</title>
                            </circle>
                        ))}
                    </svg>

                    <ul className="hp-sweep__list">
                        {cells.map((c) => (
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
                                    {c.status === 'completed'
                                        ? `loss ${c.final_train_loss != null ? c.final_train_loss.toFixed(3) : '—'} · q ${c.quality_score != null ? c.quality_score.toFixed(3) : '—'}`
                                        : c.status}
                                </span>
                            </li>
                        ))}
                    </ul>
                </>
            )}
        </div>
    );
}
