import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../video/TabVideoLink', () => ({
    default: ({ tabKey, subStep }: { tabKey: string; subStep?: string }) => <div data-testid={`video-${tabKey}${subStep ? `-${subStep}` : ''}`} />,
}));

import StepHelp, { STEP_HELP_SEEN_AFTER_MS } from './StepHelp';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../../stores/coachModeStore';

describe('StepHelp', () => {
    beforeEach(() => {
        window.localStorage.clear();
        _resetCoachModeStoreForTests();
        vi.useFakeTimers();
    });
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    const shownLongEnough = () => act(() => { vi.advanceTimersByTime(STEP_HELP_SEEN_AFTER_MS); });

    it('shows the full tip with its video on the first visit, a "?" line on the next (phase 6)', () => {
        const first = render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
        expect(screen.getByTestId('video-cleaning')).toBeInTheDocument();
        shownLongEnough();
        first.unmount();
        render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).not.toHaveTextContent('Tidy the data');
        expect(screen.queryByTestId('video-cleaning')).toBeNull();
        fireEvent.click(screen.getByTestId('step-help-cleaning-open'));
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
        expect(screen.getByTestId('video-cleaning')).toBeInTheDocument();
    });

    it('a tip unmounted before it was on screen long enough is shown in full again', () => {
        const first = render(<StepHelp projectId={7} tab="cleaning" />);
        act(() => { vi.advanceTimersByTime(STEP_HELP_SEEN_AFTER_MS - 100); });
        first.unmount();
        act(() => { vi.advanceTimersByTime(STEP_HELP_SEEN_AFTER_MS); });
        render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
    });

    it('Got it collapses the tip to the "?" line and remembers it', () => {
        const { unmount } = render(<StepHelp projectId={7} tab="cleaning" />);
        fireEvent.click(screen.getByRole('button', { name: 'Dismiss tip' }));
        expect(screen.getByTestId('step-help-cleaning-open')).toBeInTheDocument();
        expect(screen.getByTestId('step-help-cleaning')).not.toHaveTextContent('Tidy the data');
        unmount();
        render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning-open')).toBeInTheDocument();
        // Another project's first visit is still a full tip.
        render(<StepHelp projectId={8} tab="cleaning" />);
        expect(screen.getAllByTestId('step-help-cleaning')[1]).toHaveTextContent('Tidy the data');
    });

    it('follows the step and respects the tips switch', () => {
        const { rerender } = render(<StepHelp projectId={7} tab="data" />);
        expect(screen.getByTestId('step-help-data')).toHaveTextContent('Bring in your raw material');
        shownLongEnough();
        rerender(<StepHelp projectId={7} tab="goldset" />);
        expect(screen.getByTestId('step-help-goldset')).toHaveTextContent('Build a small, trusted answer key');
        // Back to a step already shown once: the "?" line.
        rerender(<StepHelp projectId={7} tab="data" />);
        expect(screen.getByTestId('step-help-data-open')).toBeInTheDocument();
        useCoachModeStore.getState().setOverride(7, 'off');
        rerender(<StepHelp projectId={7} tab="goldset" />);
        expect(screen.queryByTestId('step-help-goldset')).toBeNull();
    });

    it('follows the Results tab / Ship stage it is given', () => {
        const { rerender } = render(<StepHelp projectId={7} tab="eval" sub="verdict" />);
        expect(screen.getByTestId('step-help-eval')).toHaveTextContent('One card with the answer');
        rerender(<StepHelp projectId={7} tab="eval" sub="failures" />);
        expect(screen.getByTestId('step-help-eval')).toHaveTextContent('See where the model goes wrong');
        expect(screen.getByTestId('video-eval-failures')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Dismiss tip' }));
        expect(screen.getByTestId('step-help-eval-open')).toBeInTheDocument();
        rerender(<StepHelp projectId={7} tab="eval" sub="compare" />);
        expect(screen.getByTestId('step-help-eval')).toHaveTextContent('Put runs side by side');
        rerender(<StepHelp projectId={7} tab="export" sub="monitor" />);
        expect(screen.getByTestId('step-help-export')).toHaveTextContent('Watch the model after it ships');
        expect(screen.getByTestId('step-help-export')).not.toHaveTextContent('You made it');
    });

    it('shows the full tip every time when storage is blocked', () => {
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        const first = render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
        first.unmount();
        render(<StepHelp projectId={7} tab="cleaning" />);
        expect(screen.getByTestId('step-help-cleaning')).toHaveTextContent('Tidy the data');
    });
});
