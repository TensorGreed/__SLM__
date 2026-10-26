/**
 * Data — the workspace's data destination (Epic U, phase U4).
 *
 * Five steps as short forms, each with a Details drawer underneath:
 *   1 Sources   — import, upload, seed documents
 *   2 Prepare   — clean
 *   3 Labels    — task shape, conventions, answer key, review queue, human review
 *   4 Generate  — synthetic rows, refine weak types, relabel, spend
 *   5 Split     — train / val / test, versions, leakage
 * The former Data Studio panels and the annotation job list render as
 * Details sections; ``?step=<tab>&open=<section>`` addresses one.
 */

import { useMemo } from 'react';
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import IngestionPanel from '../components/data/IngestionPanel';
import CleaningPanel from '../components/data/CleaningPanel';
import GoldSetPanel from '../components/data/GoldSetPanel';
import SyntheticPanel from '../components/data/SyntheticPanel';
import DatasetPrepPanel from '../components/data/DatasetPrepPanel';
import LabellingConventionsCard, { isSpanProject } from '../components/data/LabellingConventionsCard';
import TaskShapeCard from '../components/data/TaskShapeCard';
import CoachNotices from '../components/coach/CoachNotices';
import StepHelp from '../components/coach/StepHelp';
import type { CoachStage } from '../api/coach';
import DataStudioOverviewPanel from '../components/data/DataStudioOverviewPanel';
import ProvenancePanel from '../components/data/ProvenancePanel';
import DataStudioSourcesSummaryPanel from '../components/data/DataStudioSourcesSummaryPanel';
import DataStudioAssistPanel from '../components/data/DataStudioAssistPanel';
import DataStudioMappingPreviewPanel from '../components/data/DataStudioMappingPreviewPanel';
import DataStudioDomainDetectionPanel from '../components/data/DataStudioDomainDetectionPanel';
import DataStudioQualitySafetyPanel from '../components/data/DataStudioQualitySafetyPanel';
import DataStudioGoldSetWorkbenchPanel from '../components/data/DataStudioGoldSetWorkbenchPanel';
import DataStudioReviewQueuePanel from '../components/data/DataStudioReviewQueuePanel';
import DataStudioSyntheticPlaybookCenterPanel from '../components/data/DataStudioSyntheticPlaybookCenterPanel';
import DataStudioSyntheticRecommendationsPanel from '../components/data/DataStudioSyntheticRecommendationsPanel';
import DataStudioSyntheticQualityPanel from '../components/data/DataStudioSyntheticQualityPanel';
import DataStudioPrepareDatasetPanel from '../components/data/DataStudioPrepareDatasetPanel';
import DataStudioDatasetVersionsPanel from '../components/data/DataStudioDatasetVersionsPanel';
import type { DetailSection } from '../components/shared/DetailsDrawer';
import api from '../api/client';
import type { TabKey } from '../types';
import { useProjectStore } from '../stores/projectStore';
import {
    DATA_OPEN_BY_STUDIO_SECTION,
    dataStepPath,
    isTabKey,
    nextTab,
    projectPath,
    resolveStep,
    settingsPath,
    stagePath,
    STEP_LABEL,
    STEPS_BY_DESTINATION,
} from '../utils/workspaceRoutes';
import { AnnotateListView } from './ProjectAnnotatePage';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './stagePage.css';

const STEP_HINT: Record<string, string> = {
    data: 'Upload files, import a dataset, or let a model write sample documents.',
    cleaning: 'Chunk, redact and tidy what came in.',
    goldset: 'What gets labelled, how, and the answer key every run is scored against.',
    synthetic: 'More rows where the data is thin; relabel under the current conventions.',
    dataprep: 'Cut train, validation and test; keep the test examples; watch for leakage.',
};

const COACH_STAGE_BY_STEP: Partial<Record<TabKey, CoachStage>> = {
    data: 'data', cleaning: 'cleaning', goldset: 'gold_set', synthetic: 'synthetic', dataprep: 'dataprep',
};

const TAB_TARGET_STAGE: Record<TabKey, string> = {
    data: 'ingestion', cleaning: 'cleaning', goldset: 'gold_set', synthetic: 'synthetic', dataprep: 'dataset_prep',
    tokenization: 'tokenization', training: 'training', eval: 'evaluation', compression: 'compression', export: 'export',
};

/** Where a Data Studio panel's target (a tab key, ``annotate``, ``domain…``,
 *  optionally with ``?query``) lives now. */
export function dataStudioTargetPath(projectId: number, target: string, sectionId?: string | null): string {
    const queryIdx = target.indexOf('?');
    const base = queryIdx >= 0 ? target.slice(0, queryIdx) : target;
    const query = queryIdx >= 0 ? target.slice(queryIdx + 1) : '';
    const section = sectionId ? sectionId.replace(/^#/, '').trim().toLowerCase().replace(/_/g, '-') : '';
    const landing = section ? DATA_OPEN_BY_STUDIO_SECTION[section] : undefined;
    if (landing) return dataStepPath(projectId, landing.step, landing.open, query);
    if (isTabKey(base)) return stagePath(projectId, base, query);
    if (base === 'annotate') return dataStepPath(projectId, 'goldset', 'annotation', query);
    if (base === 'domain' || base === 'domain-packs') return settingsPath(projectId, 'domain-packs');
    if (base === 'domain-profiles') return settingsPath(projectId, 'domain-profiles');
    return projectPath(projectId, 'data');
}

export default function ProjectDataPage() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { projectId, project, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const setActiveTab = useProjectStore((s) => s.setActiveTab);

    const steps = STEPS_BY_DESTINATION.data;
    const step = useMemo(() => resolveStep('data', searchParams.get('step')), [searchParams]);
    const openSection = searchParams.get('open');

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
        setActiveTab(next);
        navigate(stagePath(projectId, next));
    };

    const openTarget = (target: string, sectionId?: string | null) => navigate(dataStudioTargetPath(projectId, target, sectionId));

    const sourcesDetails: DetailSection[] = [
        { id: 'overview', title: 'Readiness overview', hint: 'task type, sources, domain, reviews, the next action', render: () => <DataStudioOverviewPanel projectId={projectId} onOpenTab={(tab) => openTarget(tab)} /> },
        { id: 'sources', title: 'Sources summary', hint: 'coverage and ingestion health per source', render: () => <DataStudioSourcesSummaryPanel projectId={projectId} /> },
        { id: 'provenance', title: 'Provenance', hint: 'licence, origin and allowed use of each source', render: () => <ProvenancePanel projectId={projectId} /> },
        { id: 'assist', title: 'Assist', hint: 'suggested fixes for the data as it is', render: () => <DataStudioAssistPanel projectId={projectId} /> },
    ];
    const prepareDetails: DetailSection[] = [
        { id: 'quality', title: 'Quality and safety', hint: 'PII, toxicity, leakage scans', render: () => <DataStudioQualitySafetyPanel projectId={projectId} onOpenTarget={openTarget} /> },
        { id: 'domain', title: 'Domain detection', hint: 'which domain the rows look like', render: () => <DataStudioDomainDetectionPanel projectId={projectId} onOpenTarget={openTarget} /> },
    ];
    const labelsDetails: DetailSection[] = [
        { id: 'workbench', title: 'Answer-key workbench', hint: 'coverage, suggested rows, authoring', render: () => <DataStudioGoldSetWorkbenchPanel projectId={projectId} onOpenGoldSet={() => goToStep('goldset')} /> },
        { id: 'review-queue', title: 'Review queue', hint: 'pending synthetic rows, triage, label noise', render: () => <DataStudioReviewQueuePanel projectId={projectId} onOpenTarget={openTarget} /> },
        { id: 'annotation', title: 'Annotation jobs', hint: 'label rows by hand, one job per task shape', render: () => <AnnotateListView projectId={projectId} embedded /> },
    ];
    const generateDetails: DetailSection[] = [
        { id: 'playbook-center', title: 'Playbook center', hint: 'every playbook and its last run', render: () => <DataStudioSyntheticPlaybookCenterPanel projectId={projectId} onOpenSynthetic={() => goToStep('synthetic')} /> },
        { id: 'recommendations', title: 'Recommendations', hint: 'what to generate next and why', render: () => <DataStudioSyntheticRecommendationsPanel projectId={projectId} onOpenTab={(tab) => openTarget(tab)} /> },
        { id: 'quality', title: 'Synthetic quality', hint: 'how the generated rows score', render: () => <DataStudioSyntheticQualityPanel projectId={projectId} onOpenTarget={openTarget} /> },
    ];
    const splitDetails: DetailSection[] = [
        { id: 'mapping', title: 'Schema mapping', hint: 'task-type-aware mapping before the split', render: () => <DataStudioMappingPreviewPanel projectId={projectId} onOpenTarget={openTarget} /> },
        { id: 'prepare', title: 'Prepare checklist', hint: 'what the split needs before it runs', render: () => <DataStudioPrepareDatasetPanel projectId={projectId} onOpenTarget={openTarget} /> },
        { id: 'versions', title: 'Dataset versions', hint: 'prepared versions, make active, retrain', render: () => <DataStudioDatasetVersionsPanel projectId={projectId} onOpenTarget={openTarget} /> },
    ];

    const renderStep = () => {
        switch (step) {
            case 'data':
                return <IngestionPanel projectId={projectId} onNextStep={goToNextTab} extraDetails={sourcesDetails} openSection={openSection} />;
            case 'cleaning':
                return <CleaningPanel projectId={projectId} onNextStep={goToNextTab} extraDetails={prepareDetails} openSection={openSection} />;
            case 'goldset':
                return (
                    <div className="stage-page__stack">
                        <TaskShapeCard projectId={projectId} />
                        {isSpanProject(project) && <LabellingConventionsCard projectId={projectId} />}
                        <GoldSetPanel projectId={projectId} onNextStep={goToNextTab} extraDetails={labelsDetails} openSection={openSection} />
                    </div>
                );
            case 'synthetic':
                return <SyntheticPanel projectId={projectId} onNextStep={goToNextTab} extraDetails={generateDetails} openSection={openSection} />;
            case 'dataprep':
                return <DatasetPrepPanel projectId={projectId} onNextStep={goToNextTab} extraDetails={splitDetails} openSection={openSection} />;
            default:
                return null;
        }
    };

    return (
        <div className="workspace-page stage-page" data-testid="data-page">
            <section className="workspace-page-header stage-page__header">
                <div>
                    <h2 className="workspace-page-title">Data</h2>
                    <p className="workspace-page-subtitle">{STEP_HINT[step]}</p>
                </div>
            </section>

            <div className="stage-page__steps" role="tablist" aria-label="Data steps">
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

            <StepHelp projectId={projectId} tab={step} />
            {COACH_STAGE_BY_STEP[step] && (
                <CoachNotices projectId={projectId} stage={COACH_STAGE_BY_STEP[step]!} onActionCompleted={() => { void refreshPipelineStatus(); }} />
            )}
            <div className="stage-page__content">{renderStep()}</div>
        </div>
    );
}
