import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { contextState, panelProps } = vi.hoisted(() => ({
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
    });

    it('mounts the training panel in form layout with the task shape and the extra sections', () => {
        renderAt('/project/31/train');
        expect(screen.getByTestId('training-panel')).toHaveTextContent('open=null');
        expect(screen.getByTestId('docs-qa')).toBeInTheDocument();
        expect(panelProps.last?.taskShapeLabel).toBe('Span extraction (span-extraction)');
        const ids = (panelProps.last?.moreOptions as Array<{ id: string }>).map((s) => s.id);
        expect(ids).toEqual(['plan', 'models', 'task', 'pipeline-presets', 'adapters', 'tokenization']);
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
