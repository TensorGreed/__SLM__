import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { navigateMock, setActiveTabMock, routeState, contextState } = vi.hoisted(() => ({
    navigateMock: vi.fn(),
    setActiveTabMock: vi.fn(),
    routeState: {
        tabKey: 'data',
    },
    contextState: {
        projectId: 77,
    },
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useNavigate: () => navigateMock,
        useParams: () => ({ tabKey: routeState.tabKey }),
        useOutletContext: () => ({
            projectId: contextState.projectId,
            pipelineStatus: null,
            refreshPipelineStatus: vi.fn(),
        }),
    };
});

vi.mock('../stores/projectStore', () => ({
    useProjectStore: () => ({
        activeTab: 'data',
        setActiveTab: setActiveTabMock,
    }),
}));

vi.mock('../components/dashboard/PipelineProgress', () => ({
    default: () => <div data-testid="pipeline-progress" />,
}));

vi.mock('../components/shared/GettingStartedWizard', () => ({
    default: () => <div data-testid="getting-started-wizard" />,
}));

vi.mock('../components/video/TabVideoLink', () => ({
    default: () => <div data-testid="tab-video-link" />,
}));

vi.mock('../components/data/IngestionPanel', () => ({
    default: () => <section data-testid="panel-ingestion">Ingestion</section>,
}));

vi.mock('../components/data/CleaningPanel', () => ({
    default: () => <section data-testid="panel-cleaning">Cleaning</section>,
}));

vi.mock('../components/data/GoldSetPanel', () => ({
    default: () => <section data-testid="panel-goldset-tab">Gold Set Tab</section>,
}));

vi.mock('../components/data/SyntheticPanel', () => ({
    default: () => <section data-testid="panel-synthetic-tab">Synthetic Tab</section>,
}));

vi.mock('../components/data/DatasetPrepPanel', () => ({
    default: () => <section data-testid="panel-dataprep-tab">Dataset Prep Tab</section>,
}));

vi.mock('../components/training/TokenizationPanel', () => ({
    default: () => <section data-testid="panel-tokenization">Tokenization</section>,
}));

vi.mock('../components/training/TrainingPanel', () => ({
    default: () => <section data-testid="panel-training">Training</section>,
}));

vi.mock('../components/evaluation/EvalPanel', () => ({
    default: () => <section data-testid="panel-eval">Eval</section>,
}));

vi.mock('../components/compression/CompressionPanel', () => ({
    default: () => <section data-testid="panel-compression">Compression</section>,
}));

vi.mock('../components/export/ExportPanel', () => ({
    default: () => <section data-testid="panel-export">Export</section>,
}));

import ProjectPipelinePage from './ProjectPipelinePage';

describe('ProjectPipelinePage data tab', () => {
    beforeEach(() => {
        navigateMock.mockReset();
        setActiveTabMock.mockReset();
        routeState.tabKey = 'data';
        contextState.projectId = 77;
    });

    it('keeps source ingestion in the Data tab without mounting Data Studio workbenches', () => {
        render(<ProjectPipelinePage />);

        expect(screen.getByTestId('panel-ingestion')).toBeInTheDocument();
        expect(screen.queryByTestId('panel-coach')).not.toBeInTheDocument();
        expect(screen.queryByText('Data Studio sections')).not.toBeInTheDocument();
    });
});
