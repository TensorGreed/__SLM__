/**
 * Results — the workspace's evaluation destination (Epic U, phase U3).
 *
 * Four workflows as tabs: **Verdict** (did it work: the Verdict card with the
 * score of record, base vs fine-tuned per set and the rules line, the summary
 * card, Scores, How it's scored; rule report / scorecard / runtime under
 * Details), **Failures** (where it fails: failure rows and clusters first; probe
 * pack, rule set, planner, drift queue, active learning… under Advanced), **Compare**
 * (runs side by side, benchmarks and trend, frontier, RAG, student/teacher)
 * and **Try it** (the playground and the redaction box). One evaluation
 * panel stays mounted across the tabs so the selected run persists;
 * ``?tab=`` picks a workflow and the old eval anchors map onto one;
 * ``?open=<id>`` (or an old anchor whose panel moved) opens an Advanced section.
 */

import { useMemo } from 'react';
import { useLocation, useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import EvalPanel from '../components/evaluation/EvalPanel';
import CoachNotices from '../components/coach/CoachNotices';
import PipelineMembership from '../components/pipelines/PipelineMembership';
import StepHelp from '../components/coach/StepHelp';
import ChatPlaygroundPanel from '../components/training/ChatPlaygroundPanel';
import { isResultsTab, RESULTS_OPEN_BY_HASH, RESULTS_TAB_BY_HASH, RESULTS_TABS, resultsPath, type ResultsTab } from '../utils/workspaceRoutes';
import ProjectEvalComparePage from './ProjectEvalComparePage';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectResultsPage.css';

const TAB_LABEL: Record<ResultsTab, string> = {
    verdict: 'Verdict',
    failures: 'Failures',
    compare: 'Compare',
    try: 'Try it',
};

const TAB_HINT: Record<ResultsTab, string> = {
    verdict: 'Did it work?',
    failures: 'Where does it fail?',
    compare: 'Against other runs and models',
    try: 'Chat with it, redact a text',
};

export function resolveResultsTab(search: string, hash: string): ResultsTab {
    const tab = new URLSearchParams(search).get('tab');
    if (isResultsTab(tab)) return tab;
    return RESULTS_TAB_BY_HASH[hash] ?? 'verdict';
}

/** The Advanced section to open: ``?open=<id>``, else the section an old anchor's panel moved into. */
export function resolveResultsOpen(search: string, hash: string): string | null {
    return new URLSearchParams(search).get('open') || RESULTS_OPEN_BY_HASH[hash] || null;
}

export default function ProjectResultsPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const [searchParams] = useSearchParams();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const tab = useMemo(() => resolveResultsTab(location.search, location.hash), [location.search, location.hash]);
    const hasPair = Boolean(searchParams.get('a') && searchParams.get('b'));
    const openSection = useMemo(() => resolveResultsOpen(location.search, location.hash), [location.search, location.hash]);

    const goTo = (next: ResultsTab) => {
        // Keep ?a / ?b (compare pair) and ?run (playground) across tabs.
        const keep = new URLSearchParams();
        for (const key of ['a', 'b', 'run']) {
            const value = searchParams.get(key);
            if (value) keep.set(key, value);
        }
        navigate(resultsPath(projectId, next, keep.toString()));
    };

    return (
        <div className="workspace-page results-page" data-testid="results-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Results</h2>
                    <p className="workspace-page-subtitle">Did it work, where does it fail, and how does it compare.</p>
                </div>
            </section>

            <div className="results-page__tabs" role="tablist" aria-label="Results workflows">
                {RESULTS_TABS.map((id) => (
                    <button
                        key={id}
                        type="button"
                        role="tab"
                        aria-selected={tab === id ? 'true' : 'false'}
                        className={`results-page__tab ${tab === id ? 'is-active' : ''}`}
                        onClick={() => goTo(id)}
                        data-testid={`results-tab-${id}`}
                    >
                        <span>{TAB_LABEL[id]}</span>
                        <small>{TAB_HINT[id]}</small>
                    </button>
                ))}
            </div>

            <PipelineMembership projectId={projectId} />

            <StepHelp projectId={projectId} tab="eval" sub={tab} />
            <CoachNotices projectId={projectId} stage="eval" />

            {tab === 'compare' && hasPair && (
                <div data-testid="results-compare-pair">
                    <ProjectEvalComparePage embedded />
                </div>
            )}
            {tab === 'compare' && !hasPair && (
                <p className="results-page__note" data-testid="results-compare-hint">
                    Pick a run under Verdict and press <strong>Compare to #…</strong> to see two runs side by side, or <strong>Open side by side</strong> on a
                    base vs fine-tuned line of the Verdict card; the comparisons below need no pair.
                </p>
            )}

            {tab === 'try' && (
                <div data-testid="results-playground">
                    <ChatPlaygroundPanel projectId={projectId} />
                </div>
            )}

            <EvalPanel projectId={projectId} workflow={tab} openSection={openSection} />
        </div>
    );
}
