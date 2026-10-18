/**
 * Results — the workspace's evaluation destination (Epic U, phase U3).
 *
 * Four workflows as tabs: **Verdict** (did it work: summary card, lift,
 * seeds, answer key, weak types, the pass/fail scorecard), **Failures**
 * (where it fails: failure rows, clusters, confusions, answer-key review,
 * drift queue, active learning, the rule set and the checks), **Compare**
 * (runs side by side, benchmarks and trend, frontier, RAG, student/teacher)
 * and **Try it** (the playground and the redaction box). One evaluation
 * panel stays mounted across the tabs so the selected run persists;
 * ``?tab=`` picks a workflow and the old eval anchors map onto one.
 */

import { useMemo } from 'react';
import { useLocation, useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import EvalPanel from '../components/evaluation/EvalPanel';
import CoachNotices from '../components/coach/CoachNotices';
import StepHelp from '../components/coach/StepHelp';
import ChatPlaygroundPanel from '../components/training/ChatPlaygroundPanel';
import { isResultsTab, RESULTS_TAB_BY_HASH, RESULTS_TABS, resultsPath, type ResultsTab } from '../utils/workspaceRoutes';
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

export default function ProjectResultsPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const [searchParams] = useSearchParams();
    const { projectId, pipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const tab = useMemo(() => resolveResultsTab(location.search, location.hash), [location.search, location.hash]);
    const autoGate = pipelineStatus?.auto_gate ?? null;
    const hasPair = Boolean(searchParams.get('a') && searchParams.get('b'));

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

            <StepHelp projectId={projectId} tab="eval" />
            <CoachNotices projectId={projectId} stage="eval" />

            {tab === 'verdict' && autoGate && (
                <div className="card results-page__gate" data-testid="results-auto-gate">
                    <div className="results-page__gate-head">
                        <h3>Pass/fail rules</h3>
                        <span className={`badge ${autoGate.passed ? 'badge-success' : 'badge-error'}`}>{autoGate.passed ? 'PASS' : 'FAIL'}</span>
                    </div>
                    <div className="results-page__gate-meta">
                        <span>Run <strong>#{autoGate.experiment_id}</strong></span>
                        <span>Rule set <strong>{autoGate.pack_id || 'auto'}</strong></span>
                        {autoGate.captured_at && <span>Checked <strong>{new Date(autoGate.captured_at).toLocaleString()}</strong></span>}
                    </div>
                    {autoGate.failed_gate_ids.length > 0 && (
                        <div className="results-page__gate-warn">Failed required rules: {autoGate.failed_gate_ids.join(', ')}</div>
                    )}
                    {autoGate.missing_required_metrics.length > 0 && (
                        <div className="results-page__gate-warn">Missing required metrics: {autoGate.missing_required_metrics.join(', ')}</div>
                    )}
                </div>
            )}

            {tab === 'compare' && hasPair && (
                <div data-testid="results-compare-pair">
                    <ProjectEvalComparePage embedded />
                </div>
            )}
            {tab === 'compare' && !hasPair && (
                <p className="results-page__note" data-testid="results-compare-hint">
                    Pick a run under Verdict and press <strong>Compare to #…</strong> to see two runs side by side; the comparisons below need no pair.
                </p>
            )}

            {tab === 'try' && (
                <div data-testid="results-playground">
                    <ChatPlaygroundPanel projectId={projectId} />
                </div>
            )}

            <EvalPanel projectId={projectId} workflow={tab} />
        </div>
    );
}
