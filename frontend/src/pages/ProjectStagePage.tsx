/**
 * A stage destination of the workspace — Data, Train, Results or Ship
 * (Epic U, phase U1). Each renders its steps (the former pipeline tabs)
 * behind a small step bar; ``?step=<tab>`` picks one. The panels themselves
 * are unchanged until the later phases replace them; old ``/pipeline/<tab>``
 * URLs redirect here (``utils/workspaceRoutes``).
 */

import { useMemo } from 'react';
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import IngestionPanel from '../components/data/IngestionPanel';
import CleaningPanel from '../components/data/CleaningPanel';
import GoldSetPanel from '../components/data/GoldSetPanel';
import SyntheticPanel from '../components/data/SyntheticPanel';
import DatasetPrepPanel from '../components/data/DatasetPrepPanel';
import EvalPanel from '../components/evaluation/EvalPanel';
import CompressionPanel from '../components/compression/CompressionPanel';
import ExportPanel from '../components/export/ExportPanel';
import api from '../api/client';
import type { TabKey } from '../types';
import { useProjectStore } from '../stores/projectStore';
import {
    DESTINATION_LABEL,
    nextTab,
    resolveStep,
    stagePath,
    STEP_LABEL,
    STEPS_BY_DESTINATION,
    type StageDestination,
} from '../utils/workspaceRoutes';

/** Train is its own page (ProjectTrainPage) since U2. */
type PanelDestination = Exclude<StageDestination, 'train'>;
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectStagePage.css';

const TAB_TARGET_STAGE: Record<TabKey, string> = {
    data: 'ingestion',
    cleaning: 'cleaning',
    goldset: 'gold_set',
    synthetic: 'synthetic',
    dataprep: 'dataset_prep',
    tokenization: 'tokenization',
    training: 'training',
    eval: 'evaluation',
    compression: 'compression',
    export: 'export',
};

const DESTINATION_BLURB: Record<PanelDestination, string> = {
    data: 'Bring data in, prepare it, write the answer key, generate more, and split it.',
    results: 'Did it work, where does it fail, and how do runs compare.',
    ship: 'Package the model, publish it, and deploy it.',
};

interface RelatedLink {
    label: string;
    path: string;
}

/** Pages that belong to a destination but are not steps yet (moved in U2–U5). */
const RELATED_LINKS: Record<PanelDestination, RelatedLink[]> = {
    data: [
        { label: 'Data Studio', path: 'data-studio' },
        { label: 'Annotation', path: 'annotate' },
    ],
    results: [
        { label: 'Playground', path: 'playground' },
        { label: 'Compare runs', path: 'eval/compare' },
    ],
    ship: [
        { label: 'Deployments', path: 'deployments' },
        { label: 'Observability', path: 'observability' },
    ],
};

interface ProjectStagePageProps {
    destination: PanelDestination;
}

export default function ProjectStagePage({ destination }: ProjectStagePageProps) {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { projectId, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const setActiveTab = useProjectStore((s) => s.setActiveTab);

    const steps = STEPS_BY_DESTINATION[destination];
    const step = useMemo(() => resolveStep(destination, searchParams.get('step')), [destination, searchParams]);
    const autoGate = destination === 'results' ? pipelineStatus?.auto_gate ?? null : null;

    const goToStep = (tab: TabKey) => {
        setActiveTab(tab);
        navigate(stagePath(projectId, tab));
    };

    const goToNextTab = async () => {
        const next = nextTab(step);
        if (!next) return;
        const targetStage = TAB_TARGET_STAGE[next];
        let currentStage = pipelineStatus?.current_stage;
        let guard = 0;
        while (currentStage && currentStage !== targetStage && guard < 12) {
            try {
                const res = await api.post(`/projects/${projectId}/pipeline/advance`);
                currentStage = res.data.current_stage;
                guard += 1;
            } catch {
                break;
            }
        }
        await refreshPipelineStatus();
        goToStep(next);
    };

    const renderStep = () => {
        switch (step) {
            case 'data':
                return <IngestionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'cleaning':
                return <CleaningPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'goldset':
                return <GoldSetPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'synthetic':
                return <SyntheticPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'dataprep':
                return <DatasetPrepPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'eval':
                return <EvalPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'compression':
                return <CompressionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'export':
                return <ExportPanel projectId={projectId} />;
            default:
                return null;
        }
    };

    return (
        <div className="workspace-page stage-page" data-testid={`stage-page-${destination}`}>
            <section className="workspace-page-header stage-page__header">
                <div>
                    <h2 className="workspace-page-title">{DESTINATION_LABEL[destination]}</h2>
                    <p className="workspace-page-subtitle">{DESTINATION_BLURB[destination]}</p>
                </div>
                <nav className="stage-page__related" aria-label={`${DESTINATION_LABEL[destination]} tools`}>
                    {RELATED_LINKS[destination].map((link) => (
                        <button
                            key={link.path}
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => navigate(`/project/${projectId}/${link.path}`)}
                        >
                            {link.label}
                        </button>
                    ))}
                </nav>
            </section>

            {autoGate && (
                <div className="card stage-page__gate">
                    <div className="stage-page__gate-head">
                        <h3>Pass/fail rules</h3>
                        <span className={`badge ${autoGate.passed ? 'badge-success' : 'badge-error'}`}>
                            {autoGate.passed ? 'PASS' : 'FAIL'}
                        </span>
                    </div>
                    <div className="stage-page__gate-meta">
                        <span>Run <strong>#{autoGate.experiment_id}</strong></span>
                        <span>Rule set <strong>{autoGate.pack_id || 'auto'}</strong></span>
                        {autoGate.captured_at && (
                            <span>Checked <strong>{new Date(autoGate.captured_at).toLocaleString()}</strong></span>
                        )}
                    </div>
                    {autoGate.failed_gate_ids.length > 0 && (
                        <div className="stage-page__gate-warn">Failed required rules: {autoGate.failed_gate_ids.join(', ')}</div>
                    )}
                    {autoGate.missing_required_metrics.length > 0 && (
                        <div className="stage-page__gate-warn">Missing required metrics: {autoGate.missing_required_metrics.join(', ')}</div>
                    )}
                </div>
            )}

            {steps.length > 1 && (
                <div className="stage-page__steps" role="tablist" aria-label={`${DESTINATION_LABEL[destination]} steps`}>
                    {steps.map((tab, index) => (
                        <button
                            key={tab}
                            type="button"
                            role="tab"
                            aria-selected={tab === step ? 'true' : 'false'}
                            className={`stage-page__step ${tab === step ? 'is-active' : ''}`}
                            onClick={() => goToStep(tab)}
                            data-testid={`stage-step-${tab}`}
                        >
                            <span className="stage-page__step-index">{index + 1}</span>
                            {STEP_LABEL[tab]}
                        </button>
                    ))}
                </div>
            )}

            <div className="stage-page__content">{renderStep()}</div>
        </div>
    );
}
