/**
 * Train — the workspace's training destination (Epic U, phase U2).
 *
 * The Train form (page one: base model, task shape, seeds, readiness, one
 * Train button) with everything else under More options, then the runs
 * list. The pages this folds in (training configuration, base model
 * registry, task type, pipeline presets, adapter studio, tokenization)
 * render as More-options sections; ``?open=<section>`` opens one.
 */

import { useMemo } from 'react';
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import AlignmentScaffoldPanel from '../components/training/AlignmentScaffoldPanel';
import ArchetypeComparisonPanel from '../components/training/ArchetypeComparisonPanel';
import RemediationConfigCard from '../components/training/RemediationConfigCard';
import StructuredPlanConfigCard from '../components/training/StructuredPlanConfigCard';
import ToolScoringConfigCard from '../components/training/ToolScoringConfigCard';
import DocumentsQaFlowCard from '../components/training/DocumentsQaFlowCard';
import PlanRefinementCard from '../components/training/PlanRefinementCard';
import TokenizationPanel from '../components/training/TokenizationPanel';
import TrainabilityForecastPanel from '../components/training/TrainabilityForecastPanel';
import TrainingConfigGapsPanel from '../components/training/TrainingConfigGapsPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import CoachNotices from '../components/coach/CoachNotices';
import StepHelp from '../components/coach/StepHelp';
import type { TrainFormSection } from '../components/training/trainingPanelTypes';
import { recordRemediationEvent } from '../api/remediation';
import { routeForecastAction } from '../utils/forecastActionRouter';
import { projectPath } from '../utils/workspaceRoutes';
import { isTrainSectionId, taskShapeLabelFor } from './trainPageUtils';
import ProjectAdapterStudioPage from './ProjectAdapterStudioPage';
import ProjectModelsPage from './ProjectModelsPage';
import ProjectRecipePickerPage from './ProjectRecipePickerPage';
import ProjectRecipesPage from './ProjectRecipesPage';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectTrainPage() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();

    const openSection = useMemo(() => {
        const open = searchParams.get('open');
        if (isTrainSectionId(open)) return open;
        // ``?step=tokenization`` (the former Train step) and the multi-seed deep link.
        if (searchParams.get('step') === 'tokenization') return 'tokenization';
        if (searchParams.get('expand_multi_seed') === '1') return 'seeds';
        return null;
    }, [searchParams]);

    const trainUrl = projectPath(projectId, 'train');

    const isRemediation = (project.selected_recipe as { recipe_id?: string } | null | undefined)?.recipe_id === 'code-remediation';

    const isStructuredPlan = (project.selected_recipe as { recipe_id?: string } | null | undefined)?.recipe_id === 'structured-plan';

    const taskProfile = String(
        (project.dataset_adapter_preset as { task_profile?: string } | null | undefined)?.task_profile
        ?? (project.selected_recipe as { task_profile?: string } | null | undefined)?.task_profile
        ?? '',
    );
    const isToolCalling = taskProfile === 'tool_calling';

    const extraSections: TrainFormSection[] = [
        ...(isStructuredPlan || isToolCalling
            ? [{
                id: 'tool-scoring',
                title: 'Tool-call scoring',
                hint: 'which checks score tool calls against the tool catalog, and the optional read-only dry run',
                render: () => <ToolScoringConfigCard projectId={projectId} isPlan={isStructuredPlan} />,
            }]
            : []),
        ...(isStructuredPlan
            ? [{
                id: 'structured-plan',
                title: 'Plan schema and rubric',
                hint: 'JSON Schema bundle, pointer checks, rubric judge and passages that score plans',
                render: () => <StructuredPlanConfigCard projectId={projectId} />,
            }]
            : []),
        ...(isRemediation
            ? [{
                id: 'remediation',
                title: 'Remediation checks',
                hint: 'detector, code checks, judge and passages that score patches',
                render: () => <RemediationConfigCard projectId={projectId} />,
            }]
            : []),
        {
            id: 'plan',
            title: 'Plan and gaps',
            hint: 'does the plan fit the data: config gaps, trainability forecast, archetype comparison',
            render: () => (
                <div className="train-page__stack" id="training-config-gaps-panel-host">
                    <PlanRefinementCard projectId={projectId} />
                    <TrainingConfigGapsPanel projectId={projectId} />
                    <TrainabilityForecastPanel
                        projectId={projectId}
                        onActionClicked={(kind, params) => {
                            void recordRemediationEvent(projectId, { kind, params, outcome: 'clicked' });
                            const route = routeForecastAction(projectId, kind, params);
                            if (!route) return;
                            navigate(route.search ? `${route.path}?${route.search}` : route.path);
                        }}
                    />
                    <ArchetypeComparisonPanel projectId={projectId} />
                    <AlignmentScaffoldPanel projectId={projectId} />
                </div>
            ),
        },
        {
            id: 'models',
            title: 'Base model registry',
            hint: 'browse, validate and choose a base model',
            render: () => <ProjectModelsPage embedded />,
        },
        {
            id: 'task',
            title: 'Task type',
            hint: 'the task shape, adapter and scoring mode the project trains',
            render: () => <ProjectRecipePickerPage embedded returnTo={trainUrl} />,
        },
        {
            id: 'pipeline-presets',
            title: 'Pipeline presets',
            hint: 'end-to-end presets: domain defaults, workflow, training preset, adapter, rule set',
            render: () => <ProjectRecipesPage embedded />,
        },
        {
            id: 'adapters',
            title: 'Adapter Studio',
            hint: 'map custom JSON payloads onto an adapter',
            render: () => <ProjectAdapterStudioPage embedded />,
        },
        {
            id: 'tokenization',
            title: 'Tokenization',
            hint: 'sequence lengths and truncation before you train',
            render: () => <TokenizationPanel projectId={projectId} onNextStep={() => undefined} />,
        },
    ];

    return (
        <div className="workspace-page train-page" data-testid="train-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Train</h2>
                    <p className="workspace-page-subtitle">Pick a base model and train. Runs and live metrics stay here.</p>
                </div>
            </section>
            <StepHelp projectId={projectId} tab="training" />
            <CoachNotices projectId={projectId} stage="training" />
            <TrainingPanel
                projectId={projectId}
                title="Runs"
                hideStepFooter
                moreOptions={extraSections}
                openSection={openSection}
                taskShapeLabel={taskShapeLabelFor(project)}
                defaultBaseModel={project.base_model_name}
                afterForm={<DocumentsQaFlowCard projectId={projectId} />}
            />
        </div>
    );
}
