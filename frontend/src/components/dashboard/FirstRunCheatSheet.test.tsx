import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import FirstRunCheatSheet, { CHEATSHEET_DISMISSED_KEY } from './FirstRunCheatSheet';

describe('FirstRunCheatSheet', () => {
    beforeEach(() => {
        window.localStorage.clear();
    });

    afterEach(() => {
        window.localStorage.clear();
    });

    it('renders on first visit with the three orientation items', async () => {
        render(<FirstRunCheatSheet />);
        expect(
            await screen.findByRole('heading', { name: /first time here/i }),
        ).toBeInTheDocument();
        expect(screen.getByText(/Click a demo tile below/i)).toBeInTheDocument();
        expect(screen.getByText(/Press ⌘K \(or Ctrl-K\)/i)).toBeInTheDocument();
        expect(screen.getByText(/Confused by a term/i)).toBeInTheDocument();
    });

    it('renders nothing when the dismissed flag is already set', () => {
        window.localStorage.setItem(CHEATSHEET_DISMISSED_KEY, '1');
        const { container } = render(<FirstRunCheatSheet />);
        expect(container.querySelector('.first-run-cheatsheet')).toBeNull();
    });

    it('persists the dismissal to localStorage and removes the card on click', async () => {
        render(<FirstRunCheatSheet />);
        const dismiss = await screen.findByRole('button', {
            name: /dismiss the first-run cheat sheet/i,
        });

        const user = userEvent.setup();
        await user.click(dismiss);

        expect(window.localStorage.getItem(CHEATSHEET_DISMISSED_KEY)).toBe('1');
        expect(
            screen.queryByRole('heading', { name: /first time here/i }),
        ).not.toBeInTheDocument();
    });

    it('links to the docs glossary + quickstart', async () => {
        render(<FirstRunCheatSheet />);
        const glossaryLink = await screen.findByRole('link', { name: /glossary/i });
        const quickstartLink = screen.getByRole('link', { name: /quickstart/i });
        expect(glossaryLink).toHaveAttribute(
            'href',
            'http://localhost:3001/docs/concepts/glossary',
        );
        expect(quickstartLink).toHaveAttribute(
            'href',
            'http://localhost:3001/docs/getting-started/quickstart',
        );
    });
});
