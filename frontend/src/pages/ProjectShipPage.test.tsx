import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, seen } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    seen: { stages: [] as string[] },
}));

vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({ projectId: 31, project: { id: 31, name: 'Tagger' }, pipelineStatus: null, refreshPipelineStatus: vi.fn() }),
    };
});
vi.mock('../components/export/ExportPanel', () => ({
    default: ({ stage }: { stage: string }) => {
        seen.stages.push(stage);
        return <div data-testid="export-panel">{stage}</div>;
    },
}));
vi.mock('../components/compression/CompressionPanel', () => ({ default: () => <div data-testid="compression" /> }));
vi.mock('../components/evaluation/SeedShipTable', () => ({ default: ({ leaderId }: { leaderId: number }) => <div data-testid="seed-table">leader {leaderId}</div> }));
vi.mock('./ProjectDeploymentsPage', () => ({ default: () => <div data-testid="deployments" /> }));
vi.mock('./ProjectObservabilityPage', () => ({ default: () => <div data-testid="observability" /> }));
vi.mock('./ProjectManifestPage', () => ({ default: () => <div data-testid="manifest" /> }));

vi.mock('../components/coach/CoachNotices', () => ({ default: ({ stage }: { stage: string }) => <div data-testid={`coach-notices-${stage}`} /> }));
vi.mock('../components/coach/StepHelp', () => ({ default: () => null }));
import ProjectShipPage, { resolveShipStage } from './ProjectShipPage';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderAt(path: string) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/project/:id/ship" element={<><ProjectShipPage /><LocationProbe /></>} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectShipPage', () => {
    beforeEach(() => {
        seen.stages = [];
        apiMock.get.mockReset();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/evaluation/summary')) {
                return { data: { project_id: 31, experiment_id: 142, verdict: 'evaluated', headline: null, failures: [], seed_group_leader: true, seeds: [{ experiment_id: 143, seed_value: 45, headline: null }] } };
            }
            return { data: {} };
        });
    });

    it('opens on Package with four stages, the seed pick, the export panel and the options drawer', async () => {
        renderAt('/project/31/ship');
        expect(screen.getAllByRole('tab')).toHaveLength(4);
        expect(screen.getByTestId('ship-stage-package')).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByTestId('export-panel')).toHaveTextContent('package');
        expect(await screen.findByTestId('seed-table')).toHaveTextContent('leader 142');
        expect(screen.getByTestId('ship-details')).toBeInTheDocument();
        expect(screen.queryByTestId('compression')).toBeNull();
        expect(screen.queryByTestId('deployments')).toBeNull();
    });

    it('?open=compress opens the compression option; ?open=manifest the manifest', () => {
        renderAt('/project/31/ship?open=compress');
        expect(screen.getByTestId('compression')).toBeInTheDocument();
        renderAt('/project/31/ship?open=manifest');
        expect(screen.getByTestId('manifest')).toBeInTheDocument();
    });

    it('Deploy and Monitor mount their pages; the export panel follows the stage', async () => {
        const user = userEvent.setup();
        renderAt('/project/31/ship?stage=deploy');
        expect(screen.getByTestId('deployments')).toBeInTheDocument();
        expect(screen.getByTestId('export-panel')).toHaveTextContent('deploy');
        expect(screen.queryByTestId('ship-seed-pick')).toBeNull();
        await user.click(screen.getByTestId('ship-stage-monitor'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/31/ship?stage=monitor');
        expect(screen.getByTestId('observability')).toBeInTheDocument();
        expect(screen.getByTestId('export-panel')).toHaveTextContent('monitor');
    });

    it('hides the seed pick when the latest run is not a seed group', async () => {
        apiMock.get.mockImplementation(async () => ({ data: { project_id: 31, experiment_id: 5, verdict: 'better', headline: null, failures: [] } }));
        renderAt('/project/31/ship');
        await screen.findByTestId('export-panel');
        expect(screen.queryByTestId('ship-seed-pick')).toBeNull();
    });

    it('resolveShipStage defaults to package', () => {
        expect(resolveShipStage('')).toBe('package');
        expect(resolveShipStage('?stage=publish')).toBe('publish');
        expect(resolveShipStage('?stage=nope')).toBe('package');
    });
});
