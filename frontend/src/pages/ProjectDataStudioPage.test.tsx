import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { navigateMock, setActiveTabMock, routeState, contextState, coachRailPayload } = vi.hoisted(() => ({
    navigateMock: vi.fn(),
    setActiveTabMock: vi.fn(),
    routeState: {
        hash: '',
    },
    contextState: {
        projectId: 77,
    },
    coachRailPayload: {
        project_id: 77,
        verdict: 'attention',
        read_only: true,
        auto_apply: false,
        source_of_truth: 'deterministic_data_studio_checks',
        summary: {
            blocker_count: 2,
            warning_count: 3,
            info_count: 0,
            section_count: 13,
            ready_section_count: 4,
            empty_section_count: 1,
            next_action_target: 'annotate',
        },
        next_action: {
            id: 'review_queue:pending',
            section_id: 'review_queue',
            section_label: 'Review Queue',
            severity: 'blocker',
            priority: 'high',
            title: 'Review pending rows',
            message: 'Synthetic rows need review before training.',
            action_label: 'Open review',
            target_tab: 'data',
            requires_user_confirmation: true,
        },
        next_steps: [],
        checks: [
            { id: 'sources', status: 'blocked', label: 'Sources', target_tab: 'data', message: 'Sources need attention.', blocker_count: 1, warning_count: 0, info_count: 0 },
            { id: 'mapping', status: 'ready', label: 'Mapping', target_tab: 'dataprep', message: 'Mapping is ready.', blocker_count: 0, warning_count: 0, info_count: 0 },
            { id: 'domain', status: 'attention', label: 'Domain', target_tab: 'domain', message: 'Domain needs confirmation.', blocker_count: 0, warning_count: 1, info_count: 0 },
            { id: 'quality_safety', status: 'attention', label: 'Quality & Safety', target_tab: 'dataprep', message: 'Quality warnings need review.', blocker_count: 0, warning_count: 1, info_count: 0 },
            { id: 'gold_set', status: 'ready', label: 'Gold Set', target_tab: 'goldset', message: 'Gold Set is ready.', blocker_count: 0, warning_count: 0, info_count: 0 },
            { id: 'synthetic_playbooks', status: 'attention', label: 'Synthetic Playbooks', target_tab: 'synthetic', message: 'Synthetic prerequisites need review.', blocker_count: 0, warning_count: 1, info_count: 0 },
            { id: 'synthetic_recommendations', status: 'ready', label: 'Synthetic Recommendations', target_tab: 'synthetic', message: 'Recommendations are ready.', blocker_count: 0, warning_count: 0, info_count: 0 },
            { id: 'synthetic_quality', status: 'attention', label: 'Synthetic Quality', target_tab: 'synthetic', message: 'Synthetic quality needs review.', blocker_count: 0, warning_count: 1, info_count: 0 },
            { id: 'review_queue', status: 'blocked', label: 'Review Queue', target_tab: 'annotate', message: 'Review is blocking prepare.', blocker_count: 1, warning_count: 0, info_count: 0 },
            { id: 'prepare_dataset', status: 'attention', label: 'Prepare Dataset', target_tab: 'dataprep', message: 'Splits need review.', blocker_count: 0, warning_count: 1, info_count: 0 },
            { id: 'dataset_versions', status: 'ready', label: 'Dataset Versions', target_tab: 'training', message: 'A prepared version is ready.', blocker_count: 0, warning_count: 0, info_count: 0 },
        ],
        issues: [],
        entry_points: [],
        power_details: {},
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

vi.mock('../components/data/DataStudioCoachRailPanel', async () => {
    const React = await vi.importActual<typeof import('react')>('react');
    return {
        default: ({
            onOpenTarget,
            onCoachLoaded,
        }: {
            onOpenTarget: (target: string, sectionId?: string) => void;
            onCoachLoaded?: (coach: typeof coachRailPayload) => void;
        }) => {
            React.useEffect(() => {
                onCoachLoaded?.(coachRailPayload);
            }, [onCoachLoaded]);
            return (
                <section data-testid="panel-coach">
                    Coach
                    <button type="button" onClick={() => onOpenTarget('data', 'review_queue')}>
                        Open review from coach
                    </button>
                </section>
            );
        },
    };
});

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

vi.mock('../components/data/DataStudioQualitySafetyPanel', () => ({
    default: () => <section data-testid="panel-quality-safety">Quality & Safety</section>,
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

vi.mock('../components/data/DataStudioSyntheticQualityPanel', () => ({
    default: () => <section data-testid="panel-synthetic-quality">Synthetic Quality</section>,
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
        const search = within(index).getByRole('searchbox', { name: /Search Data Studio sections/i });
        await user.type(
            search,
            'nope-no-match',
        );

        expect(within(index).getByText('No matching sections.')).toBeInTheDocument();
        const resetButtons = within(index).getAllByRole('button', { name: /Reset/i });
        await user.click(resetButtons[resetButtons.length - 1]);
        expect(search).toHaveValue('');
        expect(within(index).getByRole('button', { name: /Overview readiness/i })).toBeInTheDocument();
    });

    it('filters the section index by Coach triage signals while preserving hash navigation', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        const filters = within(index).getByRole('group', { name: /Data Studio triage filters/i });
        const blockerFilter = await within(filters).findByRole('button', { name: /^Blockers\s+7$/i });

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));
        await user.click(blockerFilter);

        expect(blockerFilter).toHaveAttribute('aria-pressed', 'true');
        expect(within(index).getByRole('button', { name: /Sources summary.*Blocker/i })).toBeInTheDocument();
        expect(within(index).getByRole('button', { name: /Review Queue.*Blocker/i })).toBeInTheDocument();
        expect(within(index).queryByRole('button', { name: /Domain detection/i })).not.toBeInTheDocument();
        expect(within(index).queryByRole('button', { name: /Schema mapping/i })).not.toBeInTheDocument();

        await user.type(
            within(index).getByRole('searchbox', { name: /Search Data Studio sections/i }),
            'sources',
        );

        const sourcesJump = within(index).getByRole('button', { name: /Sources summary.*Blocker/i });
        expect(sourcesJump).toHaveClass('is-filter-match');
        expect(within(index).queryByRole('button', { name: /Review Queue/i })).not.toBeInTheDocument();

        await user.click(sourcesJump);
        expect(screen.getByTestId('panel-sources')).toBeInTheDocument();
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/data-studio#sources', { replace: false });
    });

    it('explains why active triage filters match visible sections from Coach messages', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        const filters = within(index).getByRole('group', { name: /Data Studio triage filters/i });
        await user.click(await within(filters).findByRole('button', { name: /^Blockers\s+7$/i }));

        const explanation = within(index).getByRole('status');
        expect(within(explanation).getByText('Blocker view')).toBeInTheDocument();
        expect(within(explanation).getByText(/7 sections match current search and filter/i)).toBeInTheDocument();
        expect(within(explanation).getByText('Overview readiness')).toBeInTheDocument();
        expect(within(explanation).getAllByText(/Source Ingestion: Sources need attention/i).length).toBeGreaterThan(0);
        expect(within(explanation).getByText(/Additional matching sections are listed below/i)).toBeInTheDocument();

        await user.type(
            within(index).getByRole('searchbox', { name: /Search Data Studio sections/i }),
            'ingestion',
        );

        expect(within(explanation).getByText(/1 section matches current search and filter/i)).toBeInTheDocument();
        expect(within(explanation).queryByText('Overview readiness')).not.toBeInTheDocument();
        expect(within(explanation).getByText('Sources summary')).toBeInTheDocument();
        expect(within(explanation).getByText(/Sources: Sources need attention/i)).toBeInTheDocument();
    });

    it('shows a compact status legend and supports keyboard-friendly index reset', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        const legend = within(index).getByRole('list', { name: /Status legend/i });
        expect(within(legend).getByText('Blocker')).toBeInTheDocument();
        expect(within(legend).getByText('Attention')).toBeInTheDocument();
        expect(within(legend).getByText('Ready')).toBeInTheDocument();

        const filters = within(index).getByRole('group', { name: /Data Studio triage filters/i });
        const readyFilter = await within(filters).findByRole('button', { name: /^Ready\s+9$/i });
        await user.click(readyFilter);
        expect(readyFilter).toHaveAttribute('aria-pressed', 'true');

        await user.click(readyFilter);
        expect(readyFilter).toHaveAttribute('aria-pressed', 'false');
        expect(within(filters).getByRole('button', { name: /^All\s+13$/i })).toHaveAttribute('aria-pressed', 'true');

        await user.click(await within(filters).findByRole('button', { name: /^Blockers\s+7$/i }));
        const search = within(index).getByRole('searchbox', { name: /Search Data Studio sections/i });
        await user.type(search, 'no-match');
        expect(within(index).getByText('No matching sections.')).toBeInTheDocument();

        await user.keyboard('{Escape}');
        expect(search).toHaveValue('');
        expect(within(filters).getByRole('button', { name: /^All\s+13$/i })).toHaveAttribute('aria-pressed', 'true');
        expect(within(index).queryByText('No matching sections.')).not.toBeInTheDocument();
        expect(within(index).queryByRole('status')).not.toBeInTheDocument();
    });

    it('highlights matching handoff chips for active triage filters without disabling navigation', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        const index = screen.getByRole('region', { name: /Data Studio section index/i });
        const filters = within(index).getByRole('group', { name: /Data Studio triage filters/i });
        await user.click(await within(filters).findByRole('button', { name: /^Ready\s+9$/i }));

        const review = screen.getByTestId('data-studio-section-review-queue');
        const readyGoldChip = await within(review).findByRole('button', { name: /^Gold Set\s+Ready$/i });
        const blockerReviewChip = await within(review).findByRole('button', { name: /^Review\s+Blocker$/i });
        const attentionSyntheticChip = await within(review).findByRole('button', { name: /^Synthetic\s+Attention$/i });

        expect(readyGoldChip).toHaveClass('is-filter-match');
        expect(blockerReviewChip).toHaveClass('is-filter-muted');
        expect(attentionSyntheticChip).toHaveClass('is-filter-muted');

        await user.click(blockerReviewChip);
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/annotate');
    });

    it('routes compact workflow handoff chips from their relevant sections', async () => {
        const user = userEvent.setup();
        render(<ProjectDataStudioPage />);

        await user.click(screen.getByRole('button', { name: /Collapse all/i }));

        const sources = screen.getByTestId('data-studio-section-sources');
        const domain = screen.getByTestId('data-studio-section-domain');
        const gold = screen.getByTestId('data-studio-section-gold-set');
        const synthetic = screen.getByTestId('data-studio-section-synthetic-playbooks');
        const syntheticQuality = screen.getByTestId('data-studio-section-synthetic-quality');
        const review = screen.getByTestId('data-studio-section-review-queue');
        const prepare = screen.getByTestId('data-studio-section-prepare-dataset');
        const versions = screen.getByTestId('data-studio-section-dataset-versions');

        const sourceChip = await within(sources).findByRole('button', { name: /^Source Ingestion\s+Blocker$/i });
        expect(sourceChip).toHaveAttribute('data-status', 'blocker');
        expect(within(sourceChip).getByText('Blocker')).toBeInTheDocument();

        const domainChip = await within(domain).findByRole('button', { name: /^Domain Managers\s+Attention$/i });
        expect(domainChip).toHaveAttribute('data-status', 'attention');
        expect(within(domainChip).getByText('Attention')).toBeInTheDocument();

        const goldChip = await within(gold).findByRole('button', { name: /^Gold Set\s+Ready$/i });
        expect(goldChip).toHaveAttribute('data-status', 'ready');
        expect(within(goldChip).getByText('Ready')).toBeInTheDocument();

        const syntheticChip = await within(synthetic).findByRole('button', { name: /^Synthetic\s+Attention$/i });
        expect(syntheticChip).toHaveAttribute('data-status', 'attention');

        const syntheticQualityChip = await within(syntheticQuality).findByRole('button', { name: /^Synthetic\s+Attention$/i });
        expect(syntheticQualityChip).toHaveAttribute('data-status', 'attention');

        const reviewChip = await within(review).findByRole('button', { name: /^Review\s+Blocker$/i });
        expect(reviewChip).toHaveAttribute('data-status', 'blocker');

        const prepareChip = await within(prepare).findByRole('button', { name: /^Dataset Prep\s+Attention$/i });
        expect(prepareChip).toHaveAttribute('data-status', 'attention');

        const trainingChip = await within(versions).findByRole('button', { name: /^Training\s+Ready$/i });
        expect(trainingChip).toHaveAttribute('data-status', 'ready');

        await user.click(sourceChip);
        expect(setActiveTabMock).toHaveBeenLastCalledWith('data');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/data');
        expect(screen.queryByTestId('panel-sources')).not.toBeInTheDocument();

        await user.click(domainChip);
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/domain');

        await user.click(goldChip);
        expect(setActiveTabMock).toHaveBeenLastCalledWith('goldset');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/goldset');

        await user.click(syntheticChip);
        expect(setActiveTabMock).toHaveBeenLastCalledWith('synthetic');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/synthetic');

        await user.click(reviewChip);
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/annotate');

        await user.click(prepareChip);
        expect(setActiveTabMock).toHaveBeenLastCalledWith('dataprep');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/dataprep');

        await user.click(trainingChip);
        expect(setActiveTabMock).toHaveBeenLastCalledWith('training');
        expect(navigateMock).toHaveBeenLastCalledWith('/project/77/pipeline/training');

        const evalChip = await within(versions).findByRole('button', { name: /^Eval\s+Ready$/i });
        expect(evalChip).toHaveAttribute('data-status', 'ready');
        await user.click(evalChip);
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
