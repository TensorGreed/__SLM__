/**
 * SpanLabeler contract.
 *
 * Pins:
 * - Renders pre-existing spans as <mark> elements with the right text.
 * - Clicking a mark removes that span from the working set.
 * - On mouseup over the text region with an active type, the current
 *   selection is converted to a span and added.
 * - Pressing 'j' (or Enter) submits the working spans via onSubmit.
 * - 'esc' calls onSkip.
 */

import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import SpanLabeler from './SpanLabeler';

const SPAN_TYPES = ['PERSON', 'ORG'];

describe('SpanLabeler', () => {
    it('renders initial spans as marks', () => {
        render(
            <SpanLabeler
                text="Alice works at Acme."
                spanTypes={SPAN_TYPES}
                onSubmit={() => undefined}
                onSkip={() => undefined}
                initialSpans={[
                    { start: 0, end: 5, type: 'PERSON' },
                    { start: 15, end: 19, type: 'ORG' },
                ]}
            />,
        );
        const personMark = screen.getByTestId('span-mark-0');
        const orgMark = screen.getByTestId('span-mark-1');
        expect(personMark).toHaveTextContent('Alice');
        expect(orgMark).toHaveTextContent('Acme');
    });

    it('removes a span when its mark is clicked', async () => {
        const user = userEvent.setup();
        render(
            <SpanLabeler
                text="Alice works at Acme."
                spanTypes={SPAN_TYPES}
                onSubmit={() => undefined}
                onSkip={() => undefined}
                initialSpans={[{ start: 0, end: 5, type: 'PERSON' }]}
            />,
        );
        const mark = screen.getByTestId('span-mark-0');
        await user.click(mark);
        expect(screen.queryByTestId('span-mark-0')).toBeNull();
        // status reflects no spans
        expect(screen.getByTestId('span-labeler-status')).toHaveTextContent(
            '0 span(s)',
        );
    });

    it('creates a span from the current selection on mouseup', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <SpanLabeler
                text="Alice works at Acme."
                spanTypes={SPAN_TYPES}
                onSubmit={onSubmit}
                onSkip={() => undefined}
            />,
        );
        // Default activeType is the first spanType ('PERSON').
        const region = screen.getByTestId('span-labeler-text');
        // Plain segments are wrapped in <span>; walk one level deeper
        // to the actual text node, which is what window.getSelection()
        // would return as anchorNode in a real browser.
        const firstChild = region.firstChild as HTMLElement | null;
        const textNode = (firstChild?.firstChild ?? firstChild) as Node;
        const fakeSelection = {
            anchorNode: textNode,
            anchorOffset: 0,
            focusNode: textNode,
            focusOffset: 5,
            rangeCount: 1,
            removeAllRanges: vi.fn(),
        };
        const spy = vi
            .spyOn(window, 'getSelection')
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .mockReturnValue(fakeSelection as any);
        try {
            fireEvent.mouseUp(region);
        } finally {
            spy.mockRestore();
        }
        // A span chip should now exist.
        const mark = await screen.findByTestId('span-mark-0');
        expect(mark).toHaveTextContent('Alice');
        // Submit + verify payload.
        await user.click(screen.getByTestId('span-submit'));
        expect(onSubmit).toHaveBeenCalledWith([
            { start: 0, end: 5, type: 'PERSON' },
        ]);
    });

    it('changes active type when its letter key is pressed', async () => {
        const user = userEvent.setup();
        render(
            <SpanLabeler
                text="text"
                spanTypes={SPAN_TYPES}
                onSubmit={() => undefined}
                onSkip={() => undefined}
            />,
        );
        // 'b' activates the second span type (index 1 = ORG).
        await user.keyboard('b');
        expect(screen.getByTestId('span-labeler-status')).toHaveTextContent(
            'active type: ORG',
        );
    });

    it('submits the working set when "j" is pressed', async () => {
        const onSubmit = vi.fn();
        const user = userEvent.setup();
        render(
            <SpanLabeler
                text="text"
                spanTypes={SPAN_TYPES}
                onSubmit={onSubmit}
                onSkip={() => undefined}
                initialSpans={[{ start: 0, end: 4, type: 'PERSON' }]}
            />,
        );
        await user.keyboard('j');
        expect(onSubmit).toHaveBeenCalledWith([
            { start: 0, end: 4, type: 'PERSON' },
        ]);
    });

    it('skips on Escape', async () => {
        const onSkip = vi.fn();
        const user = userEvent.setup();
        render(
            <SpanLabeler
                text="text"
                spanTypes={SPAN_TYPES}
                onSubmit={() => undefined}
                onSkip={onSkip}
            />,
        );
        await user.keyboard('{Escape}');
        expect(onSkip).toHaveBeenCalledTimes(1);
    });
});
