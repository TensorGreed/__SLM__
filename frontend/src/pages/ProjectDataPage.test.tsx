import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { contextState, seen, panel } = vi.hoisted(() => ({
    contextState: {
        project: { id: 31, name: 'Tagger', selected_recipe: { recipe_id: 'span-extraction', scoring_mode: 'span_set' } } as unknown,
    },
    seen: { props: {} as Record<string, Record<string, unknown>> },
    panel: (id: string) => ({
        default: (props: Record<string, unknown>) => {
            seen.props[id] = props;
            return <section data-testid={id}>{String(props.layout)} open={String(props.openSection)}</section>;
        },
    }),
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({ projectId: 31, project: contextState.project, pipelineStatus: null, refreshPipelineStatus: vi.fn() }),
    };
});
vi.mock('../stores/projectStore', () => ({
    useProjectStore: (selector: (s: { setActiveTab: () => void }) => unknown) => selector({ setActiveTab: vi.fn() }),
}));
vi.mock('../components/data/IngestionPanel', () => panel('panel-ingestion'));
vi.mock('../components/data/CleaningPanel', () => panel('panel-cleaning'));
vi.mock('../components/data/GoldSetPanel', () => panel('panel-goldset'));
vi.mock('../components/data/SyntheticPanel', () => panel('panel-synthetic'));
vi.mock('../components/data/DatasetPrepPanel', () => panel('panel-dataprep'));
vi.mock('../components/data/TaskShapeCard', () => ({ default: () => <div data-testid="task-shape" /> }));
vi.mock('../components/data/LabellingConventionsCard', () => ({
    default: () => <div data-testid="conventions" />,
    isSpanProject: (p: { selected_recipe?: { scoring_mode?: string } | null }) => p.selected_recipe?.scoring_mode === 'span_set',
}));
vi.mock('./ProjectAnnotatePage', () => ({ AnnotateListView: () => <div data-testid="annotation-list" /> }));

vi.mock('../components/coach/CoachNotices', () => ({ default: ({ stage }: { stage: string }) => <div data-testid={`coach-notices-${stage}`} /> }));
vi.mock('../components/coach/StepHelp', () => ({ default: () => null }));
import ProjectDataPage, { dataStudioTargetPath } from './ProjectDataPage';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderAt(path: string) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/project/:id/data" element={<><ProjectDataPage /><LocationProbe /></>} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectDataPage', () => {
    beforeEach(() => {
        seen.props = {};
    });

    it('renders five steps and the Sources panel in stepper layout with the studio details', () => {
        renderAt('/project/31/data');
        expect(screen.getAllByRole('tab')).toHaveLength(5);
        expect(screen.getByTestId('panel-ingestion')).toHaveTextContent('open=null');
        const ids = (seen.props['panel-ingestion'].extraDetails as Array<{ id: string }>).map((s) => s.id);
        expect(ids).toEqual(['overview', 'sources', 'assist']);
    });

    it('Labels shows the task shape, the conventions for a span project, and the answer key panel with review + annotation details', () => {
        renderAt('/project/31/data?step=goldset&open=annotation');
        expect(screen.getByTestId('task-shape')).toBeInTheDocument();
        expect(screen.getByTestId('conventions')).toBeInTheDocument();
        expect(screen.getByTestId('panel-goldset')).toHaveTextContent('open=annotation');
        const ids = (seen.props['panel-goldset'].extraDetails as Array<{ id: string }>).map((s) => s.id);
        expect(ids).toEqual(['workbench', 'review-queue', 'annotation']);
    });

    it('hides the conventions for a non-span project', () => {
        contextState.project = { id: 31, name: 'FAQ', selected_recipe: { recipe_id: 'qa-sft', scoring_mode: 'field_match' } };
        renderAt('/project/31/data?step=goldset');
        expect(screen.queryByTestId('conventions')).toBeNull();
        contextState.project = { id: 31, name: 'Tagger', selected_recipe: { recipe_id: 'span-extraction', scoring_mode: 'span_set' } };
    });

    it('the step bar navigates between steps', async () => {
        const user = userEvent.setup();
        renderAt('/project/31/data');
        await user.click(screen.getByTestId('stage-step-dataprep'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/31/data?step=dataprep');
        expect(screen.getByTestId('panel-dataprep')).toBeInTheDocument();
    });

    it('dataStudioTargetPath maps the studio targets onto steps and sections', () => {
        expect(dataStudioTargetPath(7, 'dataprep')).toBe('/project/7/data?step=dataprep');
        expect(dataStudioTargetPath(7, 'training')).toBe('/project/7/train');
        expect(dataStudioTargetPath(7, 'synthetic?focus_synth_source=x')).toBe('/project/7/data?focus_synth_source=x&step=synthetic');
        expect(dataStudioTargetPath(7, 'annotate')).toBe('/project/7/data?open=annotation&step=goldset');
        expect(dataStudioTargetPath(7, 'data', 'dataset-versions')).toBe('/project/7/data?open=versions&step=dataprep');
        expect(dataStudioTargetPath(7, 'domain-packs')).toBe('/project/7/settings?open=domain-packs');
    });
});
