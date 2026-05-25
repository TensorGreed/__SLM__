import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { navigateMock, setActiveTabMock, routeState, contextState } = vi.hoisted(() => ({
    navigateMock: vi.fn(),
    setActiveTabMock: vi.fn(),
    routeState: {
        hash: '',
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
        useLocation: () => ({
            pathname: `/project/${contextState.projectId}/data-studio`,
            search: '',
            hash: routeState.hash,
            state: null,
            key: 'test',
        }),
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

vi.mock('../components/data/DataStudioCoachRailPanel', () => ({
    default: ({ onOpenTarget }: { onOpenTarget: (target: string, sectionId?: string) => void }) => (
        <section data-testid="panel-coach">
            Coach
            <button type="button" onClick={() => onOpenTarget('data', 'review_queue')}>
                Open review from coach
            </button>
        </section>
    ),
}));

vi.mock('../components/data/DataStudioOverviewPanel', () => ({
    default: () => <section data-testid="panel-overview">Overview</section>,
}));

vi.mock('../components/data/DataStudioSourcesSummaryPanel', () => ({
    default: () => <section data-testid="panel-sources">Sources</section>,
}));

vi.mock('../components/data/DataStudioMappingPreviewPanel', () => ({
    default: () => <section data-testid="panel-mapping">Mapping</section>,
}));

vi.mock('../components/data/DataStudioDomainDetectionPanel', () => ({
    default: () => <section data-testid="panel-domain">Domain</section>,
}));

vi.mock('../components/data/DataStudioAssistPanel', () => ({
    default: () => <section data-testid="panel-assist">Assist</section>,
}));

vi.mock('../components/data/DataStudioGoldSetWorkbenchPanel', () => ({
    default: () => <section data-testid="panel-gold">Gold Set</section>,
}));

vi.mock('../components/data/DataStudioSyntheticPlaybookCenterPanel', () => ({
    default: () => <section data-testid="panel-synthetic-playbooks">Synthetic Playbooks</section>,
}));

vi.mock('../components/data/DataStudioSyntheticRecommendationsPanel', () => ({
    default: () => <section data-testid="panel-synthetic-recommendations">Synthetic Recommendations</section>,
}));

vi.mock('../components/data/DataStudioReviewQueuePanel', () => ({
    default: () => <section data-testid="panel-review-queue">Review Queue</section>,
}));

vi.mock('../components/data/DataStudioPrepareDatasetPanel', () => ({
    default: () => <section data-testid="panel-prepare-dataset">Prepare Dataset</section>,
}));

vi.mock('../components/data/DataStudioDatasetVersionsPanel', () => ({
    default: () => <section data-testid="panel-dataset-versions">Dataset Versions</section>,
}));

import ProjectDataStudioPage from './ProjectDataStudioPage';

describe('ProjectDataStudioPage', () => {
    beforeEach(() => {
        navigateMock.mockReset();
        setActiveTabMock.mockReset();
        window.localStorage.clear();
        window.HTMLElement.prototype.scrollIntoView = vi.fn();
        routeState.hash = '';
        contextState.projectId = 77;
    });

    it('keeps the coach visible and collapses long-tail Data Studio panels by default', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        expect(screen.getByTestId('panel-coach')).toBeInTheDocument();
        expect(screen.getByTestId('panel-overview')).toBeInTheDocument();
        expect(screen.getByTestId('panel-sources')).toBeInTheDocument();
        expect(screen.getByTestId('panel-mapping')).toBeInTheDocument();
        expect(screen.queryByTestId('panel-domain')).not.toBeInTheDocument();
        expect(screen.queryByTestId('panel-ingestion')).not.toBeInTheDocument();

        const domainToggle = within(screen.getByTestId('data-studio-section-domain'))
            .getByRole('button', { name: /Domain detection/i });
        expect(domainToggle).toHaveAttribute('aria-expanded', 'false');

        await user.click(domainToggle);
        expect(screen.getByTestId('panel-domain')).toBeInTheDocument();
        expect(domainToggle).toHaveAttribute('aria-expanded', 'true');

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));
        expect(screen.getByTestId('panel-coach')).toBeInTheDocument();
        expect(screen.queryByTestId('panel-overview')).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Expand all/i }));
        expect(screen.queryByTestId('panel-ingestion')).not.toBeInTheDocument();
        expect(screen.getByTestId('panel-dataset-versions')).toBeInTheDocument();
    });

    it('persists expanded sections per project', async () => {
        const user = userEvent.setup();
        const firstRender = render(<ProjectDataStudioPage />);

        await user.click(
            within(screen.getByTestId('data-studio-section-domain'))
                .getByRole('button', { name: /Domain detection/i }),
        );
        expect(screen.getByTestId('panel-domain')).toBeInTheDocument();
        expect(window.localStorage.getItem('brewslm:data-studio:expanded-sections:77')).toContain('domain');

        firstRender.unmount();
        const secondRender = render(<ProjectDataStudioPage />);
        expect(screen.getByTestId('panel-domain')).toBeInTheDocument();

        contextState.projectId = 88;
        secondRender.unmount();
        render(<ProjectDataStudioPage />);
        expect(screen.queryByTestId('panel-domain')).not.toBeInTheDocument();
    });

    it('persists an intentionally collapsed Data Studio page', async () => {
        const user = userEvent.setup();
        const firstRender = render(<ProjectDataStudioPage />);

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));
        expect(screen.queryByTestId('panel-overview')).not.toBeInTheDocument();
        expect(window.localStorage.getItem('brewslm:data-studio:expanded-sections:77')).toBe('[]');

        firstRender.unmount();
        render(<ProjectDataStudioPage />);
        expect(screen.queryByTestId('panel-overview')).not.toBeInTheDocument();
    });

    it('filters the section index and opens workbenches through hash navigation', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        expect(within(index).getByText('Start')).toBeInTheDocument();
        expect(within(index).getByText('Shape')).toBeInTheDocument();
        expect(within(index).getByText('Examples')).toBeInTheDocument();
        expect(within(index).getByText('Release')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));
        expect(screen.queryByTestId('panel-synthetic-recommendations')).not.toBeInTheDocument();

        await user.type(
            within(index).getByRole('searchbox', { name: /Search Data Studio sections/i }),
            'synthetic',
        );

        expect(within(index).getByRole('button', { name: /Synthetic recommendations/i })).toBeInTheDocument();
        expect(within(index).queryByRole('button', { name: /Domain detection/i })).not.toBeInTheDocument();

        await user.click(within(index).getByRole('button', { name: /Synthetic recommendations/i }));
        expect(screen.getByTestId('panel-synthetic-recommendations')).toBeInTheDocument();
        expect(navigateMock).toHaveBeenCalledWith('/project/77/data-studio#synthetic-recommendations', { replace: false });
    });

    it('shows an empty section index state for unmatched searches', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        await user.type(
            within(index).getByRole('searchbox', { name: /Search Data Studio sections/i }),
            'nope-no-match',
        );

        expect(within(index).getByText('No matching sections.')).toBeInTheDocument();
    });

    it('routes compact workflow handoff chips from their relevant sections', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));

        const sources = screen.getByTestId('data-studio-section-sources');
        const domain = screen.getByTestId('data-studio-section-domain');
        const gold = screen.getByTestId('data-studio-section-gold-set');
        const synthetic = screen.getByTestId('data-studio-section-synthetic-playbooks');
        const review = screen.getByTestId('data-studio-section-review-queue');
        const prepare = screen.getByTestId('data-studio-section-prepare-dataset');
        const versions = screen.getByTestId('data-studio-section-dataset-versions');

        await user.click(within(sources).getByRole('button', { name: /^Source Ingestion$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('data');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/data');
        expect(screen.queryByTestId('panel-sources')).not.toBeInTheDocument();

        await user.click(within(domain).getByRole('button', { name: /^Domain Managers$/i }));
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/domain');

        await user.click(within(gold).getByRole('button', { name: /^Gold Set$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('goldset');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/goldset');

        await user.click(within(synthetic).getByRole('button', { name: /^Synthetic$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('synthetic');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/synthetic');

        await user.click(within(review).getByRole('button', { name: /^Review$/i }));
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/annotate');

        await user.click(within(prepare).getByRole('button', { name: /^Dataset Prep$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('dataprep');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/dataprep');

        await user.click(within(versions).getByRole('button', { name: /^Training$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('training');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/training');

        await user.click(within(versions).getByRole('button', { name: /^Eval$/i }));
        expect(setActiveTabMock).toHaveBeenLastCalledWith('eval');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/eval');
    });

    it('opens Data Studio sections from hashes and Coach actions', async () => {
        const user = userEvent.setup();
        routeState.hash = '#dataset-versions';
        render(<ProjectDataStudioPage />);

        expect(await screen.findByTestId('panel-dataset-versions')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));
        expect(screen.queryByTestId('panel-review-queue')).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Open review from coach/i }));
        expect(screen.getByTestId('panel-review-queue')).toBeInTheDocument();
        expect(navigateMock).toHaveBeenCalledWith('/project/77/data-studio#review-queue', { replace: false });
    });
});
