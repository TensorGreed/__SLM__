import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import CoachNotices from './CoachNotices';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../../stores/coachModeStore';

const nav = (label: string, target: string) => ({ kind: 'navigate', label, params: { target } });

function installSuggestions(list: Array<Record<string, unknown>>) {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/coach/')) return { data: { project_id: 7, stage: 'data', handler_available: true, suggestions: list } };
        return { data: {} };
    });
}

function renderNotices() {
    return render(
        <MemoryRouter>
            <CoachNotices projectId={7} stage="data" maxVisible={2} />
        </MemoryRouter>,
    );
}

describe('CoachNotices', () => {
    beforeEach(() => {
        window.localStorage.clear();
        _resetCoachModeStoreForTests();
        installSuggestions([
            { id: 'i', rule_id: 'rule-i', title: 'Info one', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
            { id: 'c', rule_id: 'rule-c', title: 'Critical one', body: '', severity: 'critical', action: nav('Go', 'recipe-picker') },
            { id: 'w', rule_id: 'rule-w', title: 'Warning one', body: '', severity: 'warning', action: nav('Go', 'recipe-picker') },
        ]);
    });

    it('shows the most severe first, limits to maxVisible with a Show more link', async () => {
        const user = userEvent.setup();
        renderNotices();
        const box = await screen.findByTestId('coach-notices-data');
        expect(box).toHaveTextContent('Critical one');
        expect(box).toHaveTextContent('Warning one');
        expect(screen.queryByText('Info one')).toBeNull();
        expect(box.textContent!.indexOf('Critical one')).toBeLessThan(box.textContent!.indexOf('Warning one'));
        await user.click(screen.getByTestId('coach-notices-more'));
        expect(screen.getByText('Info one')).toBeInTheDocument();
    });

    it('hides a notice on × and remembers it for the project and stage', async () => {
        const user = userEvent.setup();
        const { unmount } = renderNotices();
        await screen.findByText('Critical one');
        await user.click(screen.getByTestId('coach-notice-dismiss-c'));
        expect(screen.queryByText('Critical one')).toBeNull();
        expect(screen.getByText('Info one')).toBeInTheDocument();
        unmount();
        renderNotices();
        await screen.findByText('Warning one');
        expect(screen.queryByText('Critical one')).toBeNull();
    });

    it('renders nothing when inline tips are off or there are no suggestions', async () => {
        useCoachModeStore.getState().setOverride(7, 'off');
        const { container, unmount } = renderNotices();
        await waitFor(() => expect(container.firstChild).toBeNull());
        expect(apiMock.get).not.toHaveBeenCalled();
        unmount();
        useCoachModeStore.getState().setOverride(7, 'on');
        installSuggestions([]);
        const second = renderNotices();
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(second.container.firstChild).toBeNull();
    });
});
