import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { contextState, setActiveTabMock, panel } = vi.hoisted(() => ({
    contextState: { projectId: 77, pipelineStatus: null as unknown },
    setActiveTabMock: vi.fn(),
    panel: (id: string, text: string) => ({ default: () => <section data-testid={id}>{text}</section> }),
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({
            projectId: contextState.projectId,
            pipelineStatus: contextState.pipelineStatus,
            refreshPipelineStatus: vi.fn(),
        }),
    };
});

vi.mock('../stores/projectStore', () => ({
    useProjectStore: (selector: (s: { setActiveTab: typeof setActiveTabMock }) => unknown) =>
        selector({ setActiveTab: setActiveTabMock }),
}));

vi.mock('../components/data/IngestionPanel', () => panel('panel-ingestion', 'Ingestion'));
vi.mock('../components/data/CleaningPanel', () => panel('panel-cleaning', 'Cleaning'));
vi.mock('../components/data/GoldSetPanel', () => panel('panel-goldset', 'Answer key'));
vi.mock('../components/data/SyntheticPanel', () => panel('panel-synthetic', 'Synthetic'));
vi.mock('../components/data/DatasetPrepPanel', () => panel('panel-dataprep', 'Split'));
vi.mock('../components/compression/CompressionPanel', () => panel('panel-compression', 'Compression'));
vi.mock('../components/export/ExportPanel', () => panel('panel-export', 'Export'));

import ProjectStagePage from './ProjectStagePage';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderAt(path: string, destination: 'data' | 'ship') {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route
                    path="/project/:id/*"
                    element={
                        <>
                            <ProjectStagePage destination={destination} />
                            <LocationProbe />
                        </>
                    }
                />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectStagePage', () => {
    beforeEach(() => {
        setActiveTabMock.mockReset();
        contextState.pipelineStatus = null;
    });

    it('renders the first step of Data by default and the step bar', () => {
        renderAt('/project/77/data', 'data');
        expect(screen.getByTestId('panel-ingestion')).toBeInTheDocument();
        expect(screen.getAllByRole('tab')).toHaveLength(5);
        expect(screen.getByTestId('stage-step-data')).toHaveAttribute('aria-selected', 'true');
    });

    it('picks the step from ?step= and ignores a step of another destination', () => {
        renderAt('/project/77/data?step=goldset', 'data');
        expect(screen.getByTestId('panel-goldset')).toBeInTheDocument();
        renderAt('/project/77/data?step=training', 'data');
        expect(screen.getAllByTestId('panel-ingestion').length).toBeGreaterThan(0);
    });

    it('navigates between steps with the step bar', async () => {
        const user = userEvent.setup();
        renderAt('/project/77/ship', 'ship');
        expect(screen.getByTestId('panel-compression')).toBeInTheDocument();
        await user.click(screen.getByTestId('stage-step-export'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/77/ship?step=export');
        expect(setActiveTabMock).toHaveBeenCalledWith('export');
    });
});
