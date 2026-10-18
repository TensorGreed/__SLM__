/**
 * Ship — the workspace's release destination (Epic U, phase U5), one flow:
 *   Package  — pick the seed, export (compression as an option), the releases list
 *   Publish  — push a Hugging Face export to the Hub
 *   Deploy   — serve plan, deploy / SDK plan, the deployment assistant
 *   Monitor  — observability: timeline, failure clusters, support bundles
 * One export panel stays mounted across the stages so plans and the
 * releases list persist; ``?stage=`` picks a stage, ``?open=`` a Package
 * section (``compress``, ``manifest``).
 */

import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import CompressionPanel from '../components/compression/CompressionPanel';
import SeedShipTable from '../components/evaluation/SeedShipTable';
import ExportPanel from '../components/export/ExportPanel';
import DetailsDrawer from '../components/shared/DetailsDrawer';
import CoachNotices from '../components/coach/CoachNotices';
import StepHelp from '../components/coach/StepHelp';
import { fetchEvalSummary, type EvalSummary } from '../api/evalSummary';
import { isShipStage, SHIP_STAGES, shipPath, type ShipStageId } from '../utils/workspaceRoutes';
import ProjectDeploymentsPage from './ProjectDeploymentsPage';
import ProjectManifestPage from './ProjectManifestPage';
import ProjectObservabilityPage from './ProjectObservabilityPage';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './stagePage.css';

const STAGE_LABEL: Record<ShipStageId, string> = {
    package: 'Package',
    publish: 'Publish',
    deploy: 'Deploy',
    monitor: 'Monitor',
};

const STAGE_HINT: Record<ShipStageId, string> = {
    package: 'Pick the seed, export, compress',
    publish: 'To the Hugging Face Hub',
    deploy: 'Serve plans and deployments',
    monitor: 'Timeline, failures, bundles',
};

export function resolveShipStage(search: string): ShipStageId {
    const stage = new URLSearchParams(search).get('stage');
    return isShipStage(stage) ? stage : 'package';
}

/** The seed pick of the latest multi-seed run, when there is one. */
export function SeedPickCard({ projectId }: { projectId: number }) {
    const [summary, setSummary] = useState<EvalSummary | null>(null);
    useEffect(() => {
        let cancelled = false;
        fetchEvalSummary(projectId)
            .then((data) => { if (!cancelled) setSummary(data); })
            .catch(() => { if (!cancelled) setSummary(null); });
        return () => { cancelled = true; };
    }, [projectId]);
    const leaderId = summary?.trained?.experiment_id ?? summary?.experiment_id ?? null;
    if (!summary?.seed_group_leader || !leaderId) return null;
    return (
        <section className="card" data-testid="ship-seed-pick">
            <h3 className="u-card-title u-mb-sm">Pick the seed to ship</h3>
            <p className="u-hint u-mt0 u-mb-md">
                Run #{leaderId} trained several seeds. The one shipped is picked by its validation score only; the export below records why.
            </p>
            <SeedShipTable projectId={projectId} leaderId={leaderId} seeds={summary.seeds} />
        </section>
    );
}

export default function ProjectShipPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const [searchParams] = useSearchParams();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const stage = useMemo(() => resolveShipStage(location.search), [location.search]);
    const openSection = searchParams.get('open');

    const goTo = (next: ShipStageId) => navigate(shipPath(projectId, next));

    return (
        <div className="workspace-page stage-page ship-page" data-testid="ship-page">
            <section className="workspace-page-header stage-page__header">
                <div>
                    <h2 className="workspace-page-title">Ship</h2>
                    <p className="workspace-page-subtitle">Package the model, publish it, deploy it, watch it.</p>
                </div>
            </section>

            <div className="stage-page__steps" role="tablist" aria-label="Ship stages">
                {SHIP_STAGES.map((id, index) => (
                    <button
                        key={id}
                        type="button"
                        role="tab"
                        aria-selected={stage === id ? 'true' : 'false'}
                        className={`stage-page__step ${stage === id ? 'is-active' : ''}`}
                        onClick={() => goTo(id)}
                        title={STAGE_HINT[id]}
                        data-testid={`ship-stage-${id}`}
                    >
                        <span className="stage-page__step-index">{index + 1}</span>
                        {STAGE_LABEL[id]}
                    </button>
                ))}
            </div>

            <StepHelp projectId={projectId} tab="export" />
            <CoachNotices projectId={projectId} stage="export" />
            <div className="stage-page__content stage-page__stack">
                {stage === 'package' && <SeedPickCard projectId={projectId} />}

                <ExportPanel projectId={projectId} stage={stage} />

                {stage === 'package' && (
                    <DetailsDrawer
                        testId="ship-details"
                        title="Options"
                        hint="compress before exporting, the project manifest"
                        openSection={openSection}
                        sections={[
                            {
                                id: 'compress',
                                title: 'Compress first',
                                hint: 'GGUF, ONNX, pruning — the export then packages the compressed file',
                                render: () => <CompressionPanel projectId={projectId} />,
                            },
                            {
                                id: 'manifest',
                                title: 'Project manifest (Pipeline as Code)',
                                hint: 'import, validate, diff and apply a brewslm.yaml',
                                render: () => <ProjectManifestPage embedded />,
                            },
                        ]}
                    />
                )}

                {stage === 'deploy' && (
                    <div data-testid="ship-deployments">
                        <ProjectDeploymentsPage embedded />
                    </div>
                )}

                {stage === 'monitor' && (
                    <div data-testid="ship-observability">
                        <ProjectObservabilityPage embedded />
                    </div>
                )}
            </div>
        </div>
    );
}
