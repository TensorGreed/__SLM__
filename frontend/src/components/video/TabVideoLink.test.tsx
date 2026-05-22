import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import TabVideoLink from './TabVideoLink';

describe('TabVideoLink', () => {
    it('renders the link with the resolved chapter label + timecode for a known tab', () => {
        render(<TabVideoLink tabKey="data" />);
        // Catalog maps 'data' → v03 chapter "Data tab" at 00:15.
        const button = screen.getByRole('button', {
            name: /Watch the 2-minute walkthrough for Data tab/i,
        });
        expect(button).toBeInTheDocument();
        expect(button.textContent).toContain('Data tab');
        expect(button.textContent).toContain('0:15');
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
        // Cleaning chapter is in v03.
        expect(dialog.textContent).toContain('Cleaning');
    });

    it('renders different chapter labels per pipeline tab', () => {
        const { rerender } = render(<TabVideoLink tabKey="goldset" />);
        expect(
            screen.getByRole('button', { name: /Gold Set/i }),
        ).toBeInTheDocument();

        rerender(<TabVideoLink tabKey="training" />);
        // Training maps to v07 chapter "Training Config recap".
        expect(
            screen.getByRole('button', { name: /Training Config recap/i }),
        ).toBeInTheDocument();

        rerender(<TabVideoLink tabKey="compression" />);
        expect(
            screen.getByRole('button', { name: /Compression form/i }),
        ).toBeInTheDocument();
    });
});
