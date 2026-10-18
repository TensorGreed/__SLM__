import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../video/TabVideoLink', () => ({ default: ({ tabKey }: { tabKey: string }) => <div data-testid={`video-${tabKey}`} /> }));

import StepHelp from './StepHelp';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../../stores/coachModeStore';

describe('StepHelp', () => {
    beforeEach(() => {
        window.localStorage.clear();
        _resetCoachModeStoreForTests();
    });

    it('shows the step tip with its video and remembers Got it', async () => {
        const user = userEvent.setup();
        const { unmount } = render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
        expect(screen.getByTestId('video-cleaning')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Dismiss tip' }));
        expect(screen.queryByTestId('step-help-cleaning')).toBeNull();
        unmount();
        render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.queryByTestId('step-help-cleaning')).toBeNull();
    });

    it('follows the step and respects the tips switch', () => {
        const { rerender } = render(<StepHelp projectId={7} tab="data" />);
        expect(screen.getByTestId('step-help-data')).toBeInTheDocument();
        rerender(<StepHelp projectId={7} tab="goldset" />);
        expect(screen.getByTestId('step-help-goldset')).toBeInTheDocument();
        useCoachModeStore.getState().setOverride(7, 'off');
        rerender(<StepHelp projectId={7} tab="goldset" />);
        expect(screen.queryByTestId('step-help-goldset')).toBeNull();
    });
});
