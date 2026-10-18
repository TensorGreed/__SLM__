/**
 * Project Overview — the workspace's landing page (Epic U, phase U1).
 *
 * One screen: where the project is, the one next step, the latest run and
 * its verdict, the shipped model, and at most three open Coach nudges. The
 * full checklist and the one-click quickstart live behind "Full plan".
 */

import { useEffect, useMemo, useState } from 'react';
import { FlaskConical, Package } from 'lucide-react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import { fetchCoachSuggestions, type CoachStage, type CoachSuggestion } from '../api/coach';
import { fetchEvalSummary, type EvalSummary } from '../api/evalSummary';
import { latestRunText } from './overviewText';
import api from '../api/client';
import CoachSuggestionCard from '../components/coach/CoachSuggestion';
import EmptyState from '../components/shared/EmptyState';
import LoadingState from '../components/shared/LoadingState';
import QuickstartCard from '../components/guide/QuickstartCard';
import { useCoachMode } from '../components/coach/useCoachMode';
import { getPipelineStageIndex, getRecommendedAction, PIPELINE_STAGE_LABEL } from '../utils/flowGuide';
import { projectPath, stagePath } from '../utils/workspaceRoutes';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectOverviewPage.css';

interface ExportRow {
    id: number;
    status: string;
    export_format?: string;
    format?: string;
    completed_at?: string | null;
    created_at?: string;
    experiment_id?: number | null;
    hub_publish?: { repo_id: string; url: string } | null;
}

const STAGE_TO_COACH: Record<string, CoachStage> = {
    ingestion: 'data',
    cleaning: 'cleaning',
    gold_set: 'gold_set',
    synthetic: 'synthetic',
    dataset_prep: 'dataprep',
    data_adapter_preview: 'dataprep',
    tokenization: 'training',
    training: 'training',
    evaluation: 'eval',
    compression: 'export',
    export: 'export',
    completed: 'export',
};

const SEVERITY_RANK: Record<string, number> = { critical: 0, warning: 1, info: 2 };
const MAX_NUDGES = 3;

export default function ProjectOverviewPage() {
    const navigate = useNavigate();
    const { projectId, project, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const { isOn: tipsOn } = useCoachMode(projectId);
    const [summary, setSummary] = useState<EvalSummary | null>(null);
    const [exports, setExports] = useState<ExportRow[]>([]);
    const [nudges, setNudges] = useState<CoachSuggestion[]>([]);
    const [summaryLoaded, setSummaryLoaded] = useState(false);
    const [exportsLoaded, setExportsLoaded] = useState(false);
    const [planOpen, setPlanOpen] = useState(false);

    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage || 'ingestion';
    const stageIndex = getPipelineStageIndex(currentStage);
    const next = useMemo(() => getRecommendedAction(projectId, project, pipelineStatus), [projectId, project, pipelineStatus]);
    const coachStage = STAGE_TO_COACH[currentStage] ?? 'data';

    useEffect(() => {
        let cancelled = false;
        fetchEvalSummary(projectId)
            .then((data) => { if (!cancelled) { setSummary(data); setSummaryLoaded(true); } })
            .catch(() => { if (!cancelled) { setSummary(null); setSummaryLoaded(true); } });
        api.get<ExportRow[]>(`/projects/${projectId}/export/list`)
            .then((res) => { if (!cancelled) { setExports(Array.isArray(res.data) ? res.data : []); setExportsLoaded(true); } })
            .catch(() => { if (!cancelled) { setExports([]); setExportsLoaded(true); } });
        return () => { cancelled = true; };
    }, [projectId, currentStage]);

    useEffect(() => {
        if (!tipsOn) return;
        let cancelled = false;
        fetchCoachSuggestions(projectId, coachStage)
            .then((res) => {
                if (cancelled) return;
                const list = Array.isArray(res?.suggestions) ? [...res.suggestions] : [];
                list.sort((a, b) => (SEVERITY_RANK[a.severity] ?? 9) - (SEVERITY_RANK[b.severity] ?? 9));
                setNudges(list.slice(0, MAX_NUDGES));
            })
            .catch(() => { if (!cancelled) setNudges([]); });
        return () => { cancelled = true; };
    }, [projectId, coachStage, tipsOn]);

    const shipped = useMemo(() => {
        const done = exports.filter((e) => e.status === 'completed');
        done.sort((a, b) => String(b.completed_at ?? b.created_at ?? '').localeCompare(String(a.completed_at ?? a.created_at ?? '')));
        return done[0] ?? null;
    }, [exports]);

    const latest = latestRunText(summary);
    const visibleNudges = tipsOn ? nudges : [];

    const steps = [
        { id: 'data', title: 'Bring data in', path: stagePath(projectId, 'data'), complete: stageIndex >= 1 },
        { id: 'prepare', title: 'Prepare, label, split', path: stagePath(projectId, 'cleaning'), complete: stageIndex >= 7 },
        { id: 'train', title: 'Train', path: stagePath(projectId, 'training'), complete: stageIndex >= 8 },
        { id: 'results', title: 'Check the results', path: stagePath(projectId, 'eval'), complete: stageIndex >= 9 },
        { id: 'ship', title: 'Ship', path: stagePath(projectId, 'export'), complete: stageIndex >= 11 },
    ];

    return (
        <div className="workspace-page overview-page" data-testid="overview-page">
            <section className="card overview-hero">
                <div className="overview-hero__where">
                    <span className="overview-kicker">Where you are</span>
                    <strong data-testid="overview-stage">{PIPELINE_STAGE_LABEL[currentStage as keyof typeof PIPELINE_STAGE_LABEL] ?? currentStage}</strong>
                </div>
                <div className="overview-hero__next">
                    <span className="overview-kicker">Next step</span>
                    <p>{next.description}</p>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => navigate(next.path)}
                        data-testid="overview-next"
                    >
                        {next.title}
                    </button>
                </div>
            </section>

            <div className="overview-grid">
                <section className="card overview-card" data-testid="overview-latest-run">
                    <span className="overview-kicker">Latest run</span>
                    {!summaryLoaded ? (
                        <LoadingState label="Loading latest run" lines={2} />
                    ) : latest ? (
                        <>
                            <strong>{latest.title}</strong>
                            <p>{latest.detail}</p>
                            <button type="button" className="btn btn-secondary btn-sm" onClick={() => navigate(projectPath(projectId, 'results'))}>
                                Open results
                            </button>
                        </>
                    ) : (
                        <EmptyState
                            size="compact"
                            icon={<FlaskConical />}
                            title="Nothing trained yet"
                            description="Train a model and its verdict shows up here."
                            primary={{ label: 'Open Train', onClick: () => navigate(projectPath(projectId, 'train')) }}
                        />
                    )}
                </section>

                <section className="card overview-card" data-testid="overview-shipped">
                    <span className="overview-kicker">Shipped model</span>
                    {!exportsLoaded ? (
                        <LoadingState label="Loading exports" lines={2} />
                    ) : shipped ? (
                        <>
                            <strong>
                                Export #{shipped.id}
                                {shipped.experiment_id ? ` · run #${shipped.experiment_id}` : ''}
                                {` · ${shipped.export_format ?? shipped.format ?? ''}`}
                            </strong>
                            {shipped.hub_publish ? (
                                <p>
                                    Published as{' '}
                                    <a href={shipped.hub_publish.url} target="_blank" rel="noopener noreferrer">
                                        {shipped.hub_publish.repo_id}
                                    </a>
                                </p>
                            ) : (
                                <p className="overview-muted">Not published yet.</p>
                            )}
                            <button type="button" className="btn btn-secondary btn-sm" onClick={() => navigate(projectPath(projectId, 'ship'))}>
                                Open Ship
                            </button>
                        </>
                    ) : (
                        <EmptyState
                            size="compact"
                            icon={<Package />}
                            title="Nothing exported yet"
                            description="Package a trained model and it lists here."
                            primary={{ label: 'Open Ship', onClick: () => navigate(projectPath(projectId, 'ship')) }}
                        />
                    )}
                </section>
            </div>

            {visibleNudges.length > 0 && (
                <section className="overview-nudges" data-testid="overview-nudges" aria-label="Open suggestions">
                    <span className="overview-kicker">Worth a look</span>
                    {visibleNudges.map((suggestion) => (
                        <CoachSuggestionCard
                            key={suggestion.id}
                            projectId={projectId}
                            suggestion={suggestion}
                            onActionCompleted={() => { void refreshPipelineStatus(); }}
                        />
                    ))}
                </section>
            )}

            <section className="card overview-plan">
                <button
                    type="button"
                    className="overview-plan__toggle"
                    onClick={() => setPlanOpen((v) => !v)}
                    aria-expanded={planOpen ? 'true' : 'false'}
                    data-testid="overview-plan-toggle"
                >
                    {planOpen ? '▾' : '▸'} Full plan
                </button>
                {planOpen && (
                    <div className="overview-plan__body">
                        <ol className="overview-plan__steps">
                            {steps.map((step) => (
                                <li key={step.id} className={step.complete ? 'is-done' : ''}>
                                    <button type="button" className="overview-plan__step" onClick={() => navigate(step.path)}>
                                        <span className={`badge ${step.complete ? 'badge-success' : 'badge-info'}`}>{step.complete ? 'Done' : 'To do'}</span>
                                        {step.title}
                                    </button>
                                </li>
                            ))}
                        </ol>
                        <QuickstartCard
                            projectId={projectId}
                            hasBaseModel={Boolean(project.base_model_name)}
                            initialDismissedNudges={project.quickstart_tour_state?.dismissed_nudges ?? []}
                            onRefresh={() => { void refreshPipelineStatus(); }}
                        />
                    </div>
                )}
            </section>
        </div>
    );
}
