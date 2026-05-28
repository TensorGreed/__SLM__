/**
 * ParetoComparisonPanel — Track 1, Epic C.
 *
 * Renders the model bake-off sweep as a quality-vs-cost scatter so the user can
 * read the trade-off at a glance: Pareto-optimal configs (the frontier) are
 * highlighted and connected, dominated configs are dimmed, and any frontier
 * point can be promoted to the project's base model in one click.
 *
 * Self-contained: takes the sweep matrix + a promote callback as props and
 * computes the frontier client-side, so the cost axis can be toggled
 * (latency / VRAM / params) without a round-trip.
 */

import { useMemo, useState } from 'react';

export interface ParetoRow {
  model_id?: string;
  params_b?: number;
  estimated_min_vram_gb?: number;
  estimated_quality_score?: number;
  estimated_accuracy_percent?: number;
  estimated_latency_ms?: number;
  estimated_throughput_tps?: number;
  fits_available_vram?: boolean | null;
  pareto_optimal?: boolean;
  dominated_by?: string[];
  suggested_defaults?: Record<string, unknown>;
}

type CostAxis = 'estimated_latency_ms' | 'estimated_min_vram_gb' | 'params_b';

const COST_AXES: { key: CostAxis; label: string; unit: string }[] = [
  { key: 'estimated_latency_ms', label: 'Latency', unit: 'ms' },
  { key: 'estimated_min_vram_gb', label: 'VRAM', unit: 'GB' },
  { key: 'params_b', label: 'Size', unit: 'B params' },
];

const QUALITY_KEY = 'estimated_quality_score' as const;

function num(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Pure dominance: a row is on the frontier unless another row is >= on quality
 * AND <= on cost with at least one strict inequality. Mirrors the backend
 * ``annotate_pareto_frontier`` so axis toggles recompute consistently.
 */
export function computeParetoOptimal(rows: ParetoRow[], costKey: CostAxis): Set<string> {
  const optimal = new Set<string>();
  rows.forEach((row) => {
    const id = String(row.model_id || '');
    if (!id) return;
    const q = num(row[QUALITY_KEY]);
    const c = num(row[costKey]);
    const dominated = rows.some((other) => {
      if (other === row) return false;
      const oq = num(other[QUALITY_KEY]);
      const oc = num(other[costKey]);
      return oq >= q && oc <= c && (oq > q || oc < c);
    });
    if (!dominated) optimal.add(id);
  });
  return optimal;
}

interface ParetoComparisonPanelProps {
  matrix: ParetoRow[];
  currentBaseModel?: string;
  bestBalanceModelId?: string;
  onPromote: (row: ParetoRow) => void;
}

const W = 520;
const H = 300;
const PAD = { top: 20, right: 24, bottom: 44, left: 56 };

export default function ParetoComparisonPanel({
  matrix,
  currentBaseModel,
  bestBalanceModelId,
  onPromote,
}: ParetoComparisonPanelProps) {
  const [costAxis, setCostAxis] = useState<CostAxis>('estimated_latency_ms');
  const [selectedId, setSelectedId] = useState<string>('');

  const rows = useMemo(() => matrix.filter((r) => String(r.model_id || '').trim()), [matrix]);
  const axis = COST_AXES.find((a) => a.key === costAxis) || COST_AXES[0];
  const optimalIds = useMemo(() => computeParetoOptimal(rows, costAxis), [rows, costAxis]);

  const scales = useMemo(() => {
    const costs = rows.map((r) => num(r[costAxis]));
    const quals = rows.map((r) => num(r[QUALITY_KEY]));
    const cMin = Math.min(...costs, 0);
    const cMax = Math.max(...costs, 1);
    const qMin = Math.min(...quals, 0);
    const qMax = Math.max(...quals, 1);
    const cSpan = cMax - cMin || 1;
    const qSpan = qMax - qMin || 1;
    const x = (c: number) => PAD.left + ((c - cMin) / cSpan) * (W - PAD.left - PAD.right);
    // Quality up = better → invert y so higher quality sits higher on screen.
    const y = (q: number) => PAD.top + (1 - (q - qMin) / qSpan) * (H - PAD.top - PAD.bottom);
    return { x, y };
  }, [rows, costAxis]);

  const frontierPath = useMemo(() => {
    const pts = rows
      .filter((r) => optimalIds.has(String(r.model_id || '')))
      .map((r) => ({ cx: scales.x(num(r[costAxis])), cy: scales.y(num(r[QUALITY_KEY])) }))
      .sort((a, b) => a.cx - b.cx);
    if (pts.length < 2) return '';
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`).join(' ');
  }, [rows, optimalIds, scales, costAxis]);

  const selectedRow = rows.find((r) => String(r.model_id || '') === selectedId) || null;
  // Promote target: explicit selection, else the best-balance winner if it's on
  // the frontier, else the highest-quality frontier point.
  const promoteRow = useMemo(() => {
    if (selectedRow) return selectedRow;
    const frontier = rows.filter((r) => optimalIds.has(String(r.model_id || '')));
    const balance = frontier.find((r) => String(r.model_id || '') === bestBalanceModelId);
    if (balance) return balance;
    return frontier.sort((a, b) => num(b[QUALITY_KEY]) - num(a[QUALITY_KEY]))[0] || null;
  }, [selectedRow, rows, optimalIds, bestBalanceModelId]);

  if (rows.length === 0) {
    return (
      <div className="pareto-panel pareto-panel--empty" data-testid="pareto-panel">
        Run a benchmark sweep to see the quality-vs-cost frontier.
      </div>
    );
  }

  return (
    <div className="pareto-panel" data-testid="pareto-panel">
      <div className="pareto-panel__head">
        <div className="pareto-panel__title">Quality vs cost — Pareto frontier</div>
        <div className="pareto-panel__axis-toggle" role="group" aria-label="Cost axis">
          {COST_AXES.map((a) => (
            <button
              type="button"
              key={a.key}
              className={`pareto-panel__axis-btn ${a.key === costAxis ? 'is-active' : ''}`}
              onClick={() => setCostAxis(a.key)}
            >
              {a.label}
            </button>
          ))}
        </div>
      </div>

      <svg
        className="pareto-panel__chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Quality versus ${axis.label} scatter with Pareto frontier`}
      >
        {/* axes */}
        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} className="pareto-panel__axis-line" />
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} className="pareto-panel__axis-line" />
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 8} className="pareto-panel__axis-label" textAnchor="middle">
          {axis.label} ({axis.unit}) — lower is better →
        </text>
        <text x={14} y={(PAD.top + H - PAD.bottom) / 2} className="pareto-panel__axis-label" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}>
          Quality — higher is better ↑
        </text>

        {frontierPath && <path d={frontierPath} className="pareto-panel__frontier" fill="none" />}

        {rows.map((row) => {
          const id = String(row.model_id || '');
          const isOptimal = optimalIds.has(id);
          const isSelected = id === selectedId;
          const cx = scales.x(num(row[costAxis]));
          const cy = scales.y(num(row[QUALITY_KEY]));
          return (
            <g key={id} className="pareto-panel__point-group">
              <circle
                cx={cx}
                cy={cy}
                r={isSelected ? 8 : 6}
                className={`pareto-panel__point ${isOptimal ? 'is-optimal' : 'is-dominated'} ${isSelected ? 'is-selected' : ''}`}
                onClick={() => setSelectedId(isSelected ? '' : id)}
                data-testid={`pareto-point-${id}`}
              >
                <title>
                  {`${id} — quality ${num(row[QUALITY_KEY]).toFixed(3)}, ${axis.label} ${num(row[costAxis]).toFixed(1)}${axis.unit}${
                    isOptimal ? ' — Pareto-optimal' : ` — dominated by ${(row.dominated_by || []).join(', ')}`
                  }`}
                </title>
              </circle>
            </g>
          );
        })}
      </svg>

      <div className="pareto-panel__legend">
        <span className="pareto-panel__legend-item"><span className="pareto-panel__swatch is-optimal" /> Pareto-optimal</span>
        <span className="pareto-panel__legend-item"><span className="pareto-panel__swatch is-dominated" /> dominated</span>
      </div>

      <ul className="pareto-panel__list">
        {rows.map((row) => {
          const id = String(row.model_id || '');
          const isOptimal = optimalIds.has(id);
          const isCurrent = id === String(currentBaseModel || '');
          return (
            <li
              key={id}
              className={`pareto-panel__row ${isOptimal ? 'is-optimal' : 'is-dominated'} ${id === selectedId ? 'is-selected' : ''}`}
              onClick={() => setSelectedId(id === selectedId ? '' : id)}
            >
              <span className="pareto-panel__row-name">
                {id}
                {isOptimal && <span className="pareto-panel__badge">frontier</span>}
                {isCurrent && <span className="pareto-panel__badge pareto-panel__badge--current">current</span>}
              </span>
              <span className="pareto-panel__row-metrics">
                q {num(row[QUALITY_KEY]).toFixed(2)} · {num(row.estimated_latency_ms).toFixed(0)}ms · {num(row.estimated_min_vram_gb).toFixed(1)}GB
              </span>
            </li>
          );
        })}
      </ul>

      {promoteRow && (
        <div className="pareto-panel__promote">
          <span>
            {selectedRow ? 'Selected' : 'Recommended winner'}:{' '}
            <strong>{String(promoteRow.model_id || '')}</strong>
            {!optimalIds.has(String(promoteRow.model_id || '')) && (
              <span className="pareto-panel__warn"> (dominated — not on the frontier)</span>
            )}
          </span>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={String(promoteRow.model_id || '') === String(currentBaseModel || '')}
            onClick={() => onPromote(promoteRow)}
          >
            {String(promoteRow.model_id || '') === String(currentBaseModel || '')
              ? 'Already the base model'
              : `Promote ${String(promoteRow.model_id || '')}`}
          </button>
        </div>
      )}
    </div>
  );
}
