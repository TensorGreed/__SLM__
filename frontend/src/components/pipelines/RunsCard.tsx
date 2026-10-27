/**
 * RunsCard — start a pipeline run (a background Job) and list past runs.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    listPipelineRuns, parsePipelineError, percent, runStatusLabel, startPipelineRun, type RunListItem,
} from '../../api/pipelines';
import { toast } from '../../stores/toastStore';
import './pipelines.css';

interface Props { pipelineId: string; selectedRun: string | null; onSelect: (runId: string) => void; refreshKey?: number }

export default function RunsCard({ pipelineId, selectedRun, onSelect, refreshKey = 0 }: Props) {
    const [runs, setRuns] = useState<RunListItem[]>([]);
    const [maxRows, setMaxRows] = useState('50');
    const [busy, setBusy] = useState(false);
    const [started, setStarted] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(() => {
        listPipelineRuns(pipelineId).then(setRuns).catch(() => setRuns([]));
    }, [pipelineId]);
    useEffect(load, [load, refreshKey]);

    const run = async () => {
        setBusy(true);
        setError(null);
        try {
            const job = await startPipelineRun(pipelineId, Number(maxRows) || undefined);
            setStarted(`Run started as job #${job.id}. Progress and the outcome appear in the notification bell.`);
            toast.info('Pipeline run started');
            load();
        } catch (e) {
            setError(parsePipelineError(e).message);
        } finally {
            setBusy(false);
        }
    };

    return (
        <section className="pipelines__section" data-testid="pipeline-runs">
            <h3>Runs</h3>
            <div className="pipelines__row">
                <label className="pipelines__field">Max rows (1 to 500)
                    <input className="input" type="number" min={1} max={500} value={maxRows} data-testid="run-max-rows"
                        onChange={(e) => setMaxRows(e.target.value)} />
                </label>
                <button type="button" className="btn btn-primary" disabled={busy} data-testid="run-start" onClick={() => void run()}>
                    {busy ? 'Starting...' : 'Run pipeline'}
                </button>
                <button type="button" className="btn btn-ghost" onClick={load}>Refresh</button>
            </div>
            {started && <p className="pipelines__note" data-testid="run-started">{started}</p>}
            {error && <p className="pipelines__error" role="alert">{error}</p>}
            {runs.length === 0 ? <p className="pipelines__note">No runs yet.</p> : (
                <table className="pipelines__table" data-testid="runs-table">
                    <thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Rows</th><th>Fully correct</th><th></th></tr></thead>
                    <tbody>
                        {runs.map((r) => (
                            <tr key={r.id} className={r.id === selectedRun ? 'is-selected' : undefined}>
                                <td><code>{r.id}</code></td>
                                <td data-testid={`run-status-${r.id}`} title={r.error ?? undefined}>{runStatusLabel(r.status)}</td>
                                <td>{r.started_at ?? 'n/a'}</td>
                                <td>{r.rows ?? 'n/a'}{r.truncated ? ' (truncated)' : ''}</td>
                                <td>{percent(r.share_fully_correct)}</td>
                                <td><button type="button" className="btn btn-sm btn-secondary" data-testid={`run-open-${r.id}`}
                                    onClick={() => onSelect(r.id)}>{(r.status ?? 'done') === 'done' ? 'View results' : 'View'}</button></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </section>
    );
}
