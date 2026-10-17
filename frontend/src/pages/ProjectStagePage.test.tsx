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

vi.mock('../components/compression/CompressionPanel', () => panel('panel-compression', 'Compression'));
vi.mock('../components/export/ExportPanel', () => panel('panel-export', 'Export'));

import ProjectStagePage from './ProjectStagePage';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderAt(path: string, destination: 'ship') {
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

    it('renders the first step of Ship by default and the step bar', () => {
        renderAt('/project/77/ship', 'ship');
        expect(screen.getByTestId('panel-compression')).toBeInTheDocument();
        expect(screen.getAllByRole('tab')).toHaveLength(2);
        expect(screen.getByTestId('stage-step-compression')).toHaveAttribute('aria-selected', 'true');
    });

    it('ignores a step of another destination', () => {
        renderAt('/project/77/ship?step=training', 'ship');
        expect(screen.getByTestId('panel-compression')).toBeInTheDocument();
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
