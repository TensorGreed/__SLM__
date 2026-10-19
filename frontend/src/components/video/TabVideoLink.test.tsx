import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import TabVideoLink from './TabVideoLink';

describe('TabVideoLink', () => {
    it('renders the link with the resolved chapter label + timecode for a known tab', () => {
        render(<TabVideoLink tabKey="data" />);
        // Catalog maps 'data' → v03 chapter "Data · Sources" at 00:18.
        const button = screen.getByRole('button', {
            name: /Watch the 2-minute walkthrough for Data · Sources/i,
        });
        expect(button).toBeInTheDocument();
        expect(button.textContent).toContain('Data · Sources');
        expect(button.textContent).toContain('0:18');
    });

    it('opens the embed modal when clicked', async () => {
        render(<TabVideoLink tabKey="cleaning" />);
        const user = userEvent.setup();
        // Modal not in document before click.
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

        await user.click(
            screen.getByRole('button', {
                name: /Watch the 2-minute walkthrough/i,
            }),
        );

        const dialog = screen.getByRole('dialog');
        expect(dialog).toBeInTheDocument();
        // The Prepare chapter is in v03.
        expect(dialog.textContent).toContain('Data · Prepare');
    });

    it('renders different chapter labels per pipeline tab', () => {
        const { rerender } = render(<TabVideoLink tabKey="goldset" />);
        expect(
            screen.getByRole('button', { name: /Data · Labels/i }),
        ).toBeInTheDocument();

        rerender(<TabVideoLink tabKey="training" />);
        // Training maps to the v03 "Train" chapter.
        expect(
            screen.getByRole('button', { name: /walkthrough for Train$/i }),
        ).toBeInTheDocument();

        rerender(<TabVideoLink tabKey="compression" />);
        expect(
            screen.getByRole('button', { name: /Ship · Package/i }),
        ).toBeInTheDocument();
    });

    it('deep-links the Results tab / Ship stage chapter when given one', () => {
        const { rerender } = render(<TabVideoLink tabKey="eval" subStep="failures" />);
        expect(screen.getByRole('button', { name: /Results · Failures/i })).toBeInTheDocument();
        rerender(<TabVideoLink tabKey="eval" subStep="try" />);
        expect(screen.getByRole('button', { name: /Results · Try it/i })).toBeInTheDocument();
        rerender(<TabVideoLink tabKey="export" subStep="deploy" />);
        expect(screen.getByRole('button', { name: /Ship · Deploy/i })).toBeInTheDocument();
        rerender(<TabVideoLink tabKey="eval" />);
        expect(screen.getByRole('button', { name: /Results · Verdict/i })).toBeInTheDocument();
    });
});
