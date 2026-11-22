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
            <CoachNotices projectId={7} stage="data" />
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

    it('shows ONE notice, the most severe, and folds the rest under "N more tips" (phase 6)', async () => {
        const user = userEvent.setup();
        renderNotices();
        const box = await screen.findByTestId('coach-notices-data');
        expect(box).toHaveTextContent('Critical one');
        expect(screen.queryByText('Warning one')).toBeNull();
        expect(screen.queryByText('Info one')).toBeNull();
        expect(screen.getByTestId('coach-notices-more')).toHaveTextContent('2 more tips');
        await user.click(screen.getByTestId('coach-notices-more'));
        expect(screen.getByText('Warning one')).toBeInTheDocument();
        expect(screen.getByText('Info one')).toBeInTheDocument();
        expect(box.textContent!.indexOf('Warning one')).toBeLessThan(box.textContent!.indexOf('Info one'));
        expect(screen.getByTestId('coach-notices-more')).toHaveTextContent('Fewer tips');
        await user.click(screen.getByTestId('coach-notices-more'));
        expect(screen.queryByText('Warning one')).toBeNull();
        expect(screen.getByTestId('coach-notices-more')).toHaveTextContent('2 more tips');
    });

    it('the toggle is one button: its label switches and keyboard focus stays on it', async () => {
        const user = userEvent.setup();
        renderNotices();
        const toggle = await screen.findByTestId('coach-notices-more');
        toggle.focus();
        await user.keyboard('{Enter}');
        expect(screen.getByText('Warning one')).toBeInTheDocument();
        expect(screen.getByTestId('coach-notices-more')).toBe(toggle);
        expect(toggle).toHaveTextContent('Fewer tips');
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(toggle).toHaveFocus();
        await user.keyboard('{Enter}');
        expect(screen.queryByText('Warning one')).toBeNull();
        expect(toggle).toHaveTextContent('2 more tips');
        expect(toggle).toHaveFocus();
    });

    it('outside a step (forStep false) info-only tips do not say "for this step"', async () => {
        installSuggestions([
            { id: 'i', rule_id: 'rule-i', title: 'Info one', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
            { id: 'i2', rule_id: 'rule-i2', title: 'Info two', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
        ]);
        render(
            <MemoryRouter>
                <CoachNotices projectId={7} stage="data" forStep={false} />
            </MemoryRouter>,
        );
        const more = await screen.findByTestId('coach-notices-more');
        expect(more).toHaveTextContent(/^2 tips$/);
        expect(screen.getByTestId('coach-notices-data')).toHaveAttribute('aria-label', 'Suggestions');
    });

    it('never opens an info-level tip by default: only info tips = a "N tips" link', async () => {
        const user = userEvent.setup();
        installSuggestions([
            { id: 'i', rule_id: 'rule-i', title: 'Info one', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
            { id: 'i2', rule_id: 'rule-i2', title: 'Info two', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
        ]);
        renderNotices();
        const more = await screen.findByTestId('coach-notices-more');
        expect(more).toHaveTextContent('2 tips for this step');
        expect(screen.queryByText('Info one')).toBeNull();
        await user.click(more);
        expect(screen.getByText('Info one')).toBeInTheDocument();
        expect(screen.getByText('Info two')).toBeInTheDocument();
    });

    it('a warning opens when there is no critical one; one fold line names the count', async () => {
        installSuggestions([
            { id: 'i', rule_id: 'rule-i', title: 'Info one', body: '', severity: 'info', action: nav('Go', 'recipe-picker') },
            { id: 'w', rule_id: 'rule-w', title: 'Warning one', body: '', severity: 'warning', action: nav('Go', 'recipe-picker') },
        ]);
        renderNotices();
        await screen.findByText('Warning one');
        expect(screen.queryByText('Info one')).toBeNull();
        expect(screen.getByTestId('coach-notices-more')).toHaveTextContent('1 more tip');
    });

    it('hides a notice on × and remembers it for the project and stage', async () => {
        const user = userEvent.setup();
        const { unmount } = renderNotices();
        await screen.findByText('Critical one');
        await user.click(screen.getByTestId('coach-notice-dismiss-c'));
        expect(screen.queryByText('Critical one')).toBeNull();
        // The next most severe one takes its place; info stays folded.
        expect(screen.getByText('Warning one')).toBeInTheDocument();
        expect(screen.queryByText('Info one')).toBeNull();
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
