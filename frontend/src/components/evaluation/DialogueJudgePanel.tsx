/**
 * DialogueJudgePanel — summary of the per-turn replay judge.
 *
 * Every assistant turn of a multi-turn row is generated from the gold history
 * up to it and judged against that turn's reference. Shows the mean score, the
 * worst-turn mean and a by-turn-index table. A dash means undefined.
 */

import './DialogueJudgePanel.css';

export interface DialogueJudgeSummary {
    rows?: number;
    mean?: number | null;
    worst_turn_mean?: number | null;
    by_turn_index?: Array<{ turn: number; mean: number | null; n: number }>;
    /** Set instead of the numbers when the judge did not run, e.g. "no_judge_available". */
    skipped?: string;
    /** Multi-turn rows past the replay cap were left out rather than scored on their last turn alone. */
    rows_not_replayed?: number;
    replay_cap_hit?: boolean;
}

interface Props {
    summary: DialogueJudgeSummary;
    provenance: string;
}

function pct(v: number | null | undefined): string {
    return typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—';
}

/** The dialogue_judge block of an eval result's metrics, or null when absent. */
export function findDialogueJudge(metrics: unknown): DialogueJudgeSummary | null {
    const dj = (metrics as { dialogue_judge?: unknown } | null | undefined)?.dialogue_judge;
    return dj && typeof dj === 'object' ? (dj as DialogueJudgeSummary) : null;
}

/** A judge scored rows: the answer judge (``metrics.judge.judged``) or the per-turn judge (``dialogue_judge.rows``). */
export function hasJudgedRows(metrics: unknown): boolean {
    const m = metrics as { judge?: { judged?: unknown }; dialogue_judge?: { rows?: unknown } } | null | undefined;
    const judged = Number(m?.judge?.judged ?? 0);
    const rows = Number(m?.dialogue_judge?.rows ?? 0);
    return judged > 0 || rows > 0;
}

export default function DialogueJudgePanel({ summary, provenance }: Props) {
    const turns = summary.by_turn_index ?? [];
    if (summary.skipped) {
        return (
            <section className="dialogue-judge" data-testid="dialogue-judge-panel">
                <h4 className="dialogue-judge__title">Per-turn judge</h4>
                <p className="u-hint u-mt0">{provenance}</p>
                <p data-testid="dj-skipped">Per-turn judge skipped: {summary.skipped}</p>
            </section>
        );
    }
    return (
        <section className="dialogue-judge" data-testid="dialogue-judge-panel">
            <h4 className="dialogue-judge__title">Per-turn judge</h4>
            <p className="u-hint u-mt0">{provenance}</p>
            <dl className="dialogue-judge__summary">
                <div><dt>Rows</dt><dd data-testid="dj-rows">{summary.rows ?? '—'}</dd></div>
                <div><dt>Mean</dt><dd data-testid="dj-mean">{pct(summary.mean)}</dd></div>
                <div><dt>Worst-turn mean</dt><dd data-testid="dj-worst">{pct(summary.worst_turn_mean)}</dd></div>
            </dl>
            {summary.replay_cap_hit && (
                <p className="u-hint" data-testid="dj-cap-hit">
                    Replay cap reached: {summary.rows_not_replayed ?? 0} multi-turn row
                    {summary.rows_not_replayed === 1 ? ' was' : 's were'} not replayed and are not in these numbers.
                </p>
            )}
            {turns.length > 0 && (
                <table className="dialogue-judge__table" data-testid="dj-by-turn">
                    <thead>
                        <tr><th>Assistant turn</th><th>Mean</th><th>Rows</th></tr>
                    </thead>
                    <tbody>
                        {turns.map((t) => (
                            <tr key={t.turn} data-testid={`dj-turn-${t.turn}`}>
                                <td>{t.turn}</td><td>{pct(t.mean)}</td><td>{t.n}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </section>
    );
}
