/**
 * PipelineResults — one run: an end-to-end headline, a card per stage (both
 * views side by side for stage 2 and later, with the picked project / model /
 * experiment run), the first-wrong attribution chart and a per-row table.
 */

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import {
    percent, unscoredReasonLabel, type FinishedPipelineRun, type RunStage, type ViewMetrics,
} from '../../api/pipelines';
import './pipelines.css';

const COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)', 'var(--chart-5)', 'var(--chart-6)'];

function reasons(counts: Record<string, number> | undefined, skip: string[] = []): string {
    return Object.entries(counts ?? {})
        .filter(([k, v]) => v > 0 && !skip.includes(k))
        .map(([k, v]) => `${v} ${unscoredReasonLabel(k)}`)
        .join(' · ');
}

export function attributionData(run: FinishedPipelineRun): { name: string; count: number }[] {
    const rows = Object.entries(run.attribution?.first_wrong_stage ?? {}).map(([name, count]) => ({ name, count }));
    return [...rows, { name: 'correct', count: run.attribution?.correct ?? 0 }];
}

function View({ label, m, testid }: { label: string; m: ViewMetrics | null; testid: string }) {
    if (!m) return null;
    const why = reasons(m.unscored_by_reason);
    const keys = [
        m.lost_to_error ? `${m.lost_to_error} lost to an upstream error` : '',
        m.missing_item_key ? `${m.missing_item_key} missing item key` : '',
        m.duplicate_item_key ? `${m.duplicate_item_key} duplicate item key` : '',
        m.skipped_rows ? `${m.skipped_rows} rows skipped (no gold for this stage or the one before)` : '',
    ].filter(Boolean).join(' · ');
    return (
        <div className="pipelines__view" data-testid={testid}>
            <strong>{label}</strong>
            <div>Accuracy {percent(m.accuracy)} ({m.correct}/{m.scored} scored)</div>
            <div>{m.unscored} unscored{why ? ` (${why})` : ''} · {m.spurious} spurious · {m.missed_upstream} missed upstream · {m.errors} errors</div>
            {keys && <div>{keys}</div>}
        </div>
    );
}

/** Judge / scorer status on a stage card: what decided the units, and what could not be scored. */
function StageScoring({ s }: { s: RunStage }) {
    const notes: string[] = [];
    if (s.judge) {
        const what = s.judge.kind === 'rubric' ? 'Rubric judge' : 'Answer judge';
        notes.push(s.judge.ran
            ? `${what} ${s.judge.label ?? ''} graded ${s.judge.units} output${s.judge.units === 1 ? '' : 's'}`.replace(/\s+/g, ' ')
            : `${what} applies but did not run (${s.judge.reason ?? 'unknown'}): outputs it would decide are unscored`);
    }
    if (s.scoring_error) notes.push(`Scoring failed: ${s.scoring_error}. Those outputs are unscored, not wrong.`);
    if (s.item_compare) {
        const c = s.item_compare_counts;
        notes.push(s.item_compare === 'deep_equal'
            ? 'Items checked against the gold items'
            : `Items checked against gold ${c?.deep_equal ?? 0} times, by key only ${c?.key_only ?? 0} times (no gold item)`);
    }
    if (!notes.length) return null;
    return (
        <ul className={s.judge_not_run || s.scoring_error ? 'pipelines__list pipelines__warn' : 'pipelines__list'} data-testid={`result-scoring-${s.name}`}>
            {notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
    );
}

export default function PipelineResults({ run }: { run: FinishedPipelineRun }) {
    const data = attributionData(run);
    const e2e = run.end_to_end;
    return (
        <section className="pipelines__section" data-testid="pipeline-results">
            <h3>Results · run {run.run_id}</h3>
            <p className="pipelines__headline" data-testid="results-headline">
                {percent(e2e.share_fully_correct)} fully correct end to end ({e2e.fully_correct} of {e2e.gold_final_items - (e2e.unscored_items ?? 0)} scored gold final items)
                {' '}over {run.rows ?? 0} row{run.rows === 1 ? '' : 's'} and {run.calls ?? 0} model calls.
            </p>
            {(e2e.unscored_items ?? 0) > 0 && (
                <p className="pipelines__warn" data-testid="results-unscored-items">
                    {e2e.unscored_items} gold final item{e2e.unscored_items === 1 ? '' : 's'} left out ({reasons(e2e.unscored_by_reason)}):
                    {' '}a stage on their path could not be scored, so they count neither as correct nor against a stage.
                </p>
            )}
            {run.truncated && (
                <p className="pipelines__warn" role="status" data-testid="results-truncated">
                    {run.truncated_message ?? 'This run stopped early at the model-call cap, so later held-out rows were not evaluated.'}
                </p>
            )}
            {run.gold_mismatch && Object.keys(run.gold_mismatch).length > 0 && (
                <p className="pipelines__warn" data-testid="results-gold-mismatch">
                    Held-out rows left out because their gold no longer fits the pipeline:{' '}
                    {Object.entries(run.gold_mismatch).map(([k, ids]) => `${k.replace(/_/g, ' ')} (${ids.join(', ')})`).join(' · ')}
                </p>
            )}
            {Object.keys(run.errors_by_reason ?? {}).length > 0 && (
                <p className="pipelines__note">
                    Errors: {Object.entries(run.errors_by_reason ?? {}).map(([k, v]) => `${k} ${v}`).join(' · ')}
                </p>
            )}

            <div className="pipelines__cards">
                {run.stages.map((s, i) => (
                    <article className="pipelines__card" key={s.name} data-testid={`result-stage-${s.name}`}>
                        <h4>{i + 1}. {s.name}</h4>
                        <p className="pipelines__note" data-testid={`result-provenance-${s.name}`}>
                            project #{s.project_id} {s.project_name} · model {s.model_id}
                            {s.run_id !== null && s.run_id !== undefined ? ` · run #${s.run_id}` : ''}
                            {(s.scoring_mode || s.handler) ? ` · ${s.scoring_mode ?? s.handler}` : ''}
                        </p>
                        <StageScoring s={s} />
                        <div className="pipelines__views">
                            <View label={s.isolated ? 'End to end' : 'Stage 1'} m={s.end_to_end} testid={`view-e2e-${s.name}`} />
                            <View label="Isolated (gold input)" m={s.isolated} testid={`view-isolated-${s.name}`} />
                        </div>
                    </article>
                ))}
            </div>

            <h4>Where the first wrong output happens</h4>
            <div className="pipelines__chart" data-testid="attribution-chart">
                <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                        <XAxis dataKey="name" stroke="var(--chart-axis)" tick={{ fontSize: 12 }} />
                        <YAxis allowDecimals={false} stroke="var(--chart-axis)" tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="count">
                            {data.map((d, i) => <Cell key={d.name} fill={d.name === 'correct' ? 'var(--chart-2)' : COLORS[(i + 2) % COLORS.length]} />)}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <ul className="pipelines__list" data-testid="attribution-list">
                {data.map((d) => <li key={d.name}>{d.name}: {d.count}</li>)}
            </ul>
            {Object.keys(run.attribution?.missed_upstream_at ?? {}).length > 0 && (
                <p className="pipelines__note">
                    Missed upstream at: {Object.entries(run.attribution.missed_upstream_at).map(([k, v]) => `${k} ${v}`).join(' · ')}
                </p>
            )}

            <h4>Rows ({run.rows_detail.length} shown)</h4>
            <table className="pipelines__table" data-testid="rows-table">
                <thead><tr><th>Row</th><th>Stages</th><th>Final items</th></tr></thead>
                <tbody>
                    {run.rows_detail.map((r) => (
                        <tr key={r.id}>
                            <td><code>{r.id}</code></td>
                            <td>
                                {r.stages.map((s) => {
                                    const ok = s.outputs.filter((o) => o.correct === true).length;
                                    const bad = s.outputs.filter((o) => o.correct === false).length;
                                    const err = s.outputs.filter((o) => o.error).length;
                                    const uns = s.outputs.filter((o) => o.unscored_reason).length;
                                    return <div key={s.name}>{s.name}: {s.inputs_count} in · {ok} correct · {bad} wrong{uns ? ` · ${uns} unscored` : ''}{err ? ` · ${err} error` : ''}</div>;
                                })}
                            </td>
                            <td>
                                {r.final.length === 0 ? 'n/a' : r.final.map((f, i) => (
                                    <div key={i}>
                                        {typeof f.gold_item_key === 'string' ? f.gold_item_key : JSON.stringify(f.gold_item_key)}
                                        {': '}{f.unscored_reason ? `unscored (${unscoredReasonLabel(f.unscored_reason)})`
                                            : f.fully_correct ? 'fully correct' : `wrong at ${f.first_wrong_stage ?? 'unknown'}`}
                                    </div>
                                ))}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </section>
    );
}
