import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { contextState, panelProps, toolScoring } = vi.hoisted(() => ({
    toolScoring: { profile: null as string | null },
    contextState: {
        project: { id: 31, name: 'Tagger', selected_recipe: { recipe_id: 'span-extraction', name: 'Span extraction' } } as unknown,
    },
    panelProps: { last: null as Record<string, unknown> | null },
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({ projectId: 31, project: contextState.project, pipelineStatus: null, refreshPipelineStatus: vi.fn() }),
    };
});
vi.mock('../components/training/TrainingPanel', () => ({
    default: (props: Record<string, unknown>) => {
        panelProps.last = props;
        return <div data-testid="training-panel">{String(props.layout)} open={String(props.openSection)}{props.afterForm as React.ReactNode}</div>;
    },
}));
vi.mock('../api/toolScoring', () => ({
    getToolScoringConfig: vi.fn(async () => ({ config: {}, effective: null, catalog: null,
        effective_task_profile: toolScoring.profile, row_context_reads: [] })),
}));
vi.mock('../components/training/ToolScoringConfigCard', () => ({ default: () => <div data-testid="tool-scoring" /> }));
vi.mock('../components/training/DocumentsQaFlowCard', () => ({ default: () => <div data-testid="docs-qa" /> }));

vi.mock('../components/coach/CoachNotices', () => ({ default: ({ stage }: { stage: string }) => <div data-testid={`coach-notices-${stage}`} /> }));
vi.mock('../components/coach/StepHelp', () => ({ default: () => null }));
import ProjectTrainPage from './ProjectTrainPage';
import { taskShapeLabelFor } from './trainPageUtils';

function renderAt(path: string) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/project/:id/train" element={<ProjectTrainPage />} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectTrainPage', () => {
    beforeEach(() => {
        panelProps.last = null;
        toolScoring.profile = null;
    });

    it('shows tool-call scoring when the backend says the project evaluates as tool calling', async () => {
        // The recipe says span extraction and the project response has no preset, but the saved
        // dataset preset / prepared manifest make it a tool-calling project (backend-reported).
        toolScoring.profile = 'tool_calling';
        renderAt('/project/31/train');
        await waitFor(() => {
            const ids = (panelProps.last?.moreOptions as Array<{ id: string }>).map((s) => s.id);
            expect(ids[0]).toBe('tool-scoring');
        });
    });

    it('mounts the training panel in form layout with the task shape and the extra sections', () => {
        renderAt('/project/31/train');
        expect(screen.getByTestId('training-panel')).toHaveTextContent('open=null');
        expect(screen.getByTestId('docs-qa')).toBeInTheDocument();
        expect(panelProps.last?.taskShapeLabel).toBe('Span extraction (span-extraction)');
        const ids = (panelProps.last?.moreOptions as Array<{ id: string }>).map((s) => s.id);
        expect(ids).toEqual(['plan', 'models', 'task', 'pipeline-presets', 'adapters', 'tokenization', 'sql-execution', 'auto-lift']);
    });

    it.each([
        ['/project/31/train?open=models', 'models'],
        ['/project/31/train?step=tokenization', 'tokenization'],
        ['/project/31/train?expand_multi_seed=1', 'seeds'],
        ['/project/31/train?open=nope', null],
    ])('%s opens %s', (path, expected) => {
        renderAt(path);
        expect(panelProps.last?.openSection).toBe(expected);
    });

    it('taskShapeLabelFor reads the recipe snapshot', () => {
        expect(taskShapeLabelFor({ selected_recipe: null })).toBe('Not set');
        expect(taskShapeLabelFor({ selected_recipe: { recipe_id: 'qa-sft' } })).toBe('qa-sft');
        expect(taskShapeLabelFor({ selected_recipe: { recipe_id: 'qa-sft', name: 'Q&A' } })).toBe('Q&A (qa-sft)');
    });
});
