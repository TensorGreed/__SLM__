/**
 * SeedShipTable — on a multi-seed run's Eval summary: every seed with its
 * VALIDATION metric (what the pick uses), its test score when the lift check
 * has one (shown, never used to pick), the picked seed marked, and
 * "Ship this seed" → the normal export of that seed (Hugging Face), whose
 * manifest / release notes / model card say how it was picked.
 */

import { useEffect, useState } from 'react';

import { fetchSeedPick, shipSeed, type SeedPick, type ShipSeedResponse } from '../../api/seedGroups';
import type { EvalSummaryHeadline } from '../../api/evalSummary';
import { metricDisplayName } from './liftEvidence';
import { toast } from '../../stores/toastStore';

interface SeedShipTableProps {
    projectId: number;
    leaderId: number;
    /** The summary's per-seed lift headlines (test examples), by run id. */
    seeds?: Array<{ experiment_id: number; seed_value: number | null; headline: EvalSummaryHeadline | null }> | null;
}

function fmt(value: number | null | undefined): string {
    return typeof value === 'number' ? (Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3)) : '—';
}

function errorText(err: unknown, fallback: string): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    const detail = data?.message ?? data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

export default function SeedShipTable({ projectId, leaderId, seeds }: SeedShipTableProps) {
    const [pick, setPick] = useState<SeedPick | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [shipping, setShipping] = useState(false);
    const [shipped, setShipped] = useState<ShipSeedResponse | null>(null);

    useEffect(() => {
        let alive = true;
        setPick(null);
        setError(null);
        setShipped(null);
        fetchSeedPick(projectId, leaderId)
            .then((next) => { if (alive) setPick(next); })
            .catch((err) => { if (alive) setError(errorText(err, 'Could not read the seeds of this run.')); });
        return () => { alive = false; };
    }, [projectId, leaderId]);

    if (error) {
        return <p className="eval-summary__message" data-testid="seed-ship-error">{error}</p>;
    }
    if (!pick) return null;

    const testById = new Map((seeds ?? []).map((s) => [s.experiment_id, s.headline]));
    const testMetric = (seeds ?? []).find((s) => s.headline)?.headline?.metric_id ?? null;
    const recorded = pick.recorded;

    const ship = async () => {
        setShipping(true);
        try {
            const out = await shipSeed(projectId, leaderId);
            setShipped(out);
            if (out.export.status === 'completed') {
                toast.success(`Export #${out.export.id}: seed ${out.picked_seed_value ?? '?'} (run #${out.picked_experiment_id}) packaged.`, 6000);
            } else {
                toast.error(`Export #${out.export.id} ${out.export.status}${out.export.error ? `: ${out.export.error}` : ''}`);
            }
        } catch (err) {
            toast.error(errorText(err, 'Could not ship this seed.'));
        } finally {
            setShipping(false);
        }
    };

    return (
        <div className="eval-summary__seed-ship" data-testid="seed-ship">
            <p className="eval-summary__judge">
                <strong>Ship one seed</strong> — picked by {pick.rule}.
            </p>
            <table className="eval-summary__regime-table" data-testid="seed-ship-table">
                <thead>
                    <tr>
                        <th scope="col">Seed</th>
                        <th scope="col">Run</th>
                        <th scope="col">Validation {pick.metric_label}</th>
                        {testMetric && <th scope="col">Test {metricDisplayName(testMetric)} (not used to pick)</th>}
                    </tr>
                </thead>
                <tbody>
                    {pick.children.map((c) => (
                        <tr key={c.experiment_id} data-testid={`seed-ship-row-${c.experiment_id}`} data-picked={c.picked ? 'true' : 'false'}>
                            <th scope="row">
                                {c.seed_value ?? '—'}
                                {c.picked && <span className="eval-summary__seed-picked" data-testid="seed-ship-picked"> ★ picked</span>}
                            </th>
                            <td>#{c.experiment_id}</td>
                            <td title={c.skipped_reason ?? undefined}>{c.value != null ? fmt(c.value) : '— (no value)'}</td>
                            {testMetric && <td>{fmt(testById.get(c.experiment_id)?.trained_value)}</td>}
                        </tr>
                    ))}
                </tbody>
            </table>
            {pick.picked_experiment_id != null ? (
                <p className="eval-summary__judge">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void ship()}
                        disabled={shipping}
                        data-testid="seed-ship-button"
                        title="Exports the picked seed in Hugging Face format. The model card / manifest / release notes say which seed was picked, of how many, by which validation metric."
                    >
                        {shipping ? 'Exporting…' : `Ship this seed (run #${pick.picked_experiment_id})`}
                    </button>
                    {recorded && !shipped && recorded.export_id != null && (
                        <span data-testid="seed-ship-recorded">
                            {' '}Last shipped: run #{recorded.picked_experiment_id} as export #{recorded.export_id}.
                        </span>
                    )}
                </p>
            ) : (
                <p className="eval-summary__message">No seed recorded a validation {pick.metric_label}; nothing to pick by.</p>
            )}
            {shipped && (
                <p className="eval-summary__judge" data-testid="seed-ship-result">
                    Export #{shipped.export.id} ({shipped.export.format}, {shipped.export.status}) of seed{' '}
                    {shipped.picked_seed_value ?? '?'} (run #{shipped.picked_experiment_id})
                    {shipped.export.output_path ? <> · <code>{shipped.export.output_path}</code></> : null}
                    {shipped.export.error ? ` — ${shipped.export.error}` : ''}
                    {shipped.evaluated_before_ship?.ran ? ' · evaluated first (it had no evaluation on record)' : ''}
                    {' '}· <a href={`/project/${projectId}/pipeline/export`} data-testid="seed-ship-export-link">Open Export →</a>
                </p>
            )}
        </div>
    );
}
