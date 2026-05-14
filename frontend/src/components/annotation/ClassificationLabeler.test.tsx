/**
 * ClassificationLabeler contract.
 *
 * Pins:
 * - Press number key → onSubmit called with the indexed label.
 * - Click a label button → onSubmit called with that label.
 * - Press 'esc' → onSkip called.
 * - When disabled, keyboard + click handlers are no-ops.
 * - Number keys past labels.length are ignored.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import ClassificationLabeler from './ClassificationLabeler';

const LABELS = ['positive', 'negative', 'neutral'];

describe('ClassificationLabeler', () => {
    it('submits the label at index 1 when "2" is pressed', async () => {
        const onSubmit = vi.fn();
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <ClassificationLabeler
                text="sample text"
                labels={LABELS}
                onSubmit={onSubmit}
                onSkip={onSkip}
            />,
        );
        await user.keyboard('2');
        expect(onSubmit).toHaveBeenCalledTimes(1);
        expect(onSubmit).toHaveBeenCalledWith('negative');
        expect(onSkip).not.toHaveBeenCalled();
    });

    it('submits via mouse click on a label button', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <ClassificationLabeler
                text="sample"
                labels={LABELS}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.click(screen.getByTestId('classification-label-neutral'));
        expect(onSubmit).toHaveBeenCalledWith('neutral');
    });

    it('skips on Escape', async () => {
        const onSubmit = vi.fn();
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <ClassificationLabeler
                text="sample"
                labels={LABELS}
                onSubmit={onSubmit}
                onSkip={onSkip}
            />,
        );
        await user.keyboard('{Escape}');
        expect(onSkip).toHaveBeenCalledTimes(1);
        expect(onSubmit).not.toHaveBeenCalled();
    });

    it('ignores number keys past labels.length', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <ClassificationLabeler
                text="sample"
                labels={LABELS}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('9');
        expect(onSubmit).not.toHaveBeenCalled();
    });

    it('ignores keyboard + click when disabled', async () => {
        const onSubmit = vi.fn();
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <ClassificationLabeler
                text="sample"
                labels={LABELS}
                onSubmit={onSubmit}
                onSkip={onSkip}
                disabled
            />,
        );
        await user.keyboard('1');
        await user.click(screen.getByTestId('classification-label-positive'));
        expect(onSubmit).not.toHaveBeenCalled();
        expect(onSkip).not.toHaveBeenCalled();
    });
});
