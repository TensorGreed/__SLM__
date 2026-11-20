/**
 * "How it's scored" on Results › Verdict (UI simplification pass, phase 1; decision 5).
 *
 * Read-only for now: for each set the selected run was scored on (newest result), what the stored
 * result says about how it was scored — scoring mode, strict tool-call arguments, null counts as
 * absent, prompt rendering, decoding — and links to where those settings are changed today
 * (Train › More options, Score this run › Advanced). The settings themselves move here in a later phase.
 */

import { evalResultDatasetLabel } from '../../api/evalSets';
import { trainSectionPath } from '../../utils/workspaceRoutes';
import { howScoredText, scoringModeOf, verdictResultsPerSet, type VerdictResultLike } from './verdictScore';
import './VerdictCard.css';

interface HowItsScoredCardProps {
    projectId: number;
    runLabel: string;
    results: VerdictResultLike[];
}

export default function HowItsScoredCard({ projectId, runLabel, results }: HowItsScoredCardProps) {
    const shown = verdictResultsPerSet(results);
    const modes = new Set(shown.map((r) => scoringModeOf(r)));
    return (
        <section className="card how-scored" data-testid="how-scored">
            <h3 className="u-card-title u-m0">How it's scored</h3>
            {shown.length === 0 ? (
                <p className="u-hint u-m0">Nothing scored on this run yet.</p>
            ) : (
                <ul className="how-scored__list">
                    {shown.map((r) => (
                        <li key={r.id} data-testid={`how-scored-${r.id}`}>
                            {runLabel} · {evalResultDatasetLabel(r as Parameters<typeof evalResultDatasetLabel>[0])} · result #{r.id}: {howScoredText(r)}
                        </li>
                    ))}
                </ul>
            )}
            <p className="u-meta u-m0" data-testid="how-scored-links">
                Change it:{' '}
                {(modes.has('tool_call') || modes.has('structured_plan')) && (
                    <><a href={trainSectionPath(projectId, 'tool-scoring')}>Tool-call scoring</a>{' · '}</>
                )}
                <a href={trainSectionPath(projectId, 'sql-execution')}>SQL execution match</a>
                {' · '}
                <a href={trainSectionPath(projectId, 'auto-lift')}>Automatic base-vs-fine-tuned check</a>
                {' '}(Train › More options) · null counts as absent: Score this run › Advanced. New settings apply to
                evals run after the change.
            </p>
        </section>
    );
}
