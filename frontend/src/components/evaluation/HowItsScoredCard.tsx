/**
 * "How it's scored" on Results › Verdict (UI simplification pass, phase 1; decision 5).
 *
 * For each set the selected run was scored on (newest result), what the stored result says about how
 * it was scored — scoring mode, strict tool-call arguments, null counts as absent, prompt rendering,
 * decoding. Phase 4: the project's scoring settings live here too, under the collapsed
 * `AdvancedSection` "Change how it's scored" (`how-scored-settings`, `ScoringSettingsSection`); the old
 * Train › More options links (`?open=tool-scoring|structured-plan|remediation|sql-execution|auto-lift`)
 * redirect to ``?open=how-scored-settings&setting=<id>``.
 */

import { evalResultDatasetLabel } from '../../api/evalSets';
import AdvancedSection from '../shared/AdvancedSection';
import ScoringSettingsSection from './ScoringSettingsSection';
import { howScoredText, verdictResultsPerSet, type VerdictResultLike } from './verdictScore';
import './VerdictCard.css';

export const HOW_SCORED_SETTINGS_ID = 'how-scored-settings';

interface HowItsScoredCardProps {
    projectId: number;
    /** Null while no run is selected (the settings still show). */
    runLabel: string | null;
    results: VerdictResultLike[];
    /** The page's ``?open=`` request (`EvalPanel` sectionRequest). */
    openSection?: string | null;
    openNonce?: number;
}

/** ``?setting=<id>`` on the current URL: which settings card to scroll to. */
function focusFromUrl(): string | null {
    try {
        return new URLSearchParams(window.location.search).get('setting');
    } catch {
        return null;
    }
}

export default function HowItsScoredCard({ projectId, runLabel, results, openSection = null, openNonce = 0 }: HowItsScoredCardProps) {
    const shown = runLabel ? verdictResultsPerSet(results) : [];
    return (
        <section className="card how-scored" data-testid="how-scored">
            <h3 className="u-card-title u-m0">How it's scored</h3>
            {!runLabel ? (
                <p className="u-hint u-m0">No run selected.</p>
            ) : shown.length === 0 ? (
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
            <AdvancedSection
                pageKey="results-verdict"
                id={HOW_SCORED_SETTINGS_ID}
                title="Change how it's scored"
                hint="tool-call scoring, plan rubric, remediation checks, null as absent, SQL execution, automatic base-vs-fine-tuned check"
                openSection={openSection}
                openNonce={openNonce}
            >
                <ScoringSettingsSection projectId={projectId} focus={openSection === HOW_SCORED_SETTINGS_ID ? focusFromUrl() : null} />
            </AdvancedSection>
        </section>
    );
}
