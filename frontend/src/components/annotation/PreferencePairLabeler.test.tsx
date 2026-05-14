/**
 * PreferencePairLabeler contract.
 *
 * Pins:
 * - Pressing ← submits { chosen: 'A', tie: false, both_bad: false }.
 * - Pressing → submits { chosen: 'B', tie: false, both_bad: false }.
 * - Pressing '=' submits { chosen: null, tie: true, both_bad: false }.
 * - Pressing 'r' submits { chosen: null, tie: false, both_bad: true }.
 * - Pressing 'esc' calls onSkip.
 * - Typing a comment then submitting includes the comment in the payload.
 * - Shortcuts are suppressed while the comment textarea is focused
 *   so the reviewer can use arrow keys inside the field.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import PreferencePairLabeler from './PreferencePairLabeler';

const ROW = {
    prompt: 'Translate "hello" to French.',
    completionA: 'Bonjour',
    completionB: 'Salut',
};

describe('PreferencePairLabeler', () => {
    it('submits A on ArrowLeft', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('{ArrowLeft}');
        expect(onSubmit).toHaveBeenCalledTimes(1);
        expect(onSubmit).toHaveBeenCalledWith({
            chosen: 'A',
            tie: false,
            both_bad: false,
        });
    });

    it('submits B on ArrowRight', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('{ArrowRight}');
        expect(onSubmit).toHaveBeenCalledWith({
            chosen: 'B',
            tie: false,
            both_bad: false,
        });
    });

    it('marks tie on "=" with chosen=null', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('=');
        expect(onSubmit).toHaveBeenCalledWith({
            chosen: null,
            tie: true,
            both_bad: false,
        });
    });

    it('marks both-bad on "r"', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('r');
        expect(onSubmit).toHaveBeenCalledWith({
            chosen: null,
            tie: false,
            both_bad: true,
        });
    });

    it('skips on Escape', async () => {
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={() => undefined}
                onSkip={onSkip}
            />,
        );
        await user.keyboard('{Escape}');
        expect(onSkip).toHaveBeenCalledTimes(1);
    });

    it('includes the typed comment in the submitted payload', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        const textarea = screen.getByTestId('pref-comment');
        await user.click(textarea);
        await user.type(textarea, 'tone matches better');
        // Click the "Prefer B" button — keyboard shortcuts are
        // suppressed while the textarea has focus, so clicks are the
        // expected submit path after typing a comment.
        await user.click(screen.getByTestId('pref-b'));
        expect(onSubmit).toHaveBeenCalledWith({
            chosen: 'B',
            tie: false,
            both_bad: false,
            comment: 'tone matches better',
        });
    });

    it('does not fire shortcuts when the comment textarea is focused', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        const textarea = screen.getByTestId('pref-comment');
        await user.click(textarea);
        // Inside the textarea, ArrowLeft should move the caret, not
        // submit a preference choice.
        await user.keyboard('{ArrowLeft}');
        expect(onSubmit).not.toHaveBeenCalled();
    });

    it('omits comment when textarea is empty', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        await user.keyboard('{ArrowLeft}');
        const payload = onSubmit.mock.calls[0][0];
        expect(payload).not.toHaveProperty('comment');
    });

    it('does not submit when disabled', async () => {
        const onSubmit = vi.fn();
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <PreferencePairLabeler
                {...ROW}
                onSubmit={onSubmit}
                onSkip={onSkip}
                disabled
            />,
        );
        await user.keyboard('{ArrowLeft}');
        await user.click(screen.getByTestId('pref-a'));
        expect(onSubmit).not.toHaveBeenCalled();
    });
});
