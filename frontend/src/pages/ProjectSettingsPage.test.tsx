import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, contextState, setActiveProjectMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    setActiveProjectMock: vi.fn(),
    contextState: {
        project: { id: 31, name: 'Tagger', beginner_mode: true } as unknown,
    },
}));

vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({ projectId: 31, project: contextState.project, pipelineStatus: null, refreshPipelineStatus: vi.fn() }),
    };
});
vi.mock('../stores/projectStore', () => ({
    useProjectStore: (selector: (s: { setActiveProject: typeof setActiveProjectMock }) => unknown) =>
        selector({ setActiveProject: setActiveProjectMock }),
}));
vi.mock('../components/manifest/ManifestExportButton', () => ({ default: () => <button type="button">Export manifest</button> }));
vi.mock('./ProjectDomainPacksPage', () => ({ default: () => <div data-testid="domain-packs" /> }));
vi.mock('./ProjectDomainProfilesPage', () => ({ default: () => <div data-testid="domain-profiles" /> }));
vi.mock('./ProjectWorkflowPage', () => ({ default: () => <div data-testid="workflow" /> }));
vi.mock('./ProjectExtensionStudioPage', () => ({ default: () => <div data-testid="extensions" /> }));
vi.mock('./ProjectAutopilotPage', () => ({ default: () => <div data-testid="autopilot" /> }));

import ProjectSettingsPage from './ProjectSettingsPage';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../stores/coachModeStore';
import { _resetThemeStoreForTests, useThemeStore } from '../stores/themeStore';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname}</div>;
}

function renderSettings(path = '/project/31/settings') {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/project/:id/settings" element={<><ProjectSettingsPage /><LocationProbe /></>} />
                <Route path="*" element={<LocationProbe />} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectSettingsPage', () => {
    beforeEach(() => {
        _resetCoachModeStoreForTests();
        apiMock.put.mockReset();
        setActiveProjectMock.mockReset();
    });

    it('Appearance picks the theme and applies it to <html>', async () => {
        const user = userEvent.setup();
        renderSettings();
        expect(screen.getByTestId('settings-theme-light')).toBeChecked();
        await user.click(screen.getByTestId('settings-theme-dark'));
        expect(document.documentElement.dataset.theme).toBe('dark');
        expect(useThemeStore.getState().preference).toBe('dark');
        _resetThemeStoreForTests();
    });

    it('toggles inline tips for the project', async () => {
        const user = userEvent.setup();
        renderSettings();
        const tips = screen.getByTestId('settings-inline-tips');
        expect(tips).toBeChecked();
        await user.click(tips);
        expect(useCoachModeStore.getState().isOn(31)).toBe(false);
    });

    it('saves the beginner flag through PUT /projects/{id}', async () => {
        const user = userEvent.setup();
        apiMock.put.mockResolvedValue({ data: { id: 31, name: 'Tagger', beginner_mode: false } });
        renderSettings();
        await user.click(screen.getByTestId('settings-beginner-mode'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31', { beginner_mode: false }));
        expect(setActiveProjectMock).toHaveBeenCalledWith({ id: 31, name: 'Tagger', beginner_mode: false });
    });

    it('holds the long tail as sections that load only when opened; ?open= opens one', async () => {
        const user = userEvent.setup();
        renderSettings();
        expect(screen.queryByTestId('domain-packs')).toBeNull();
        await user.click(screen.getByTestId('settings-details-toggle'));
        await user.click(screen.getByTestId('settings-details-section-domain-packs'));
        expect(screen.getByTestId('domain-packs')).toBeInTheDocument();
        expect(screen.queryByTestId('workflow')).toBeNull();
        renderSettings('/project/31/settings?open=extensions');
        expect(screen.getByTestId('extensions')).toBeInTheDocument();
    });

    it('links the tools that live elsewhere', async () => {
        const user = userEvent.setup();
        renderSettings();
        await user.click(screen.getByRole('button', { name: /Adapter Studio/ }));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/31/train');
    });
});
