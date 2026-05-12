import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import EmptyState from './EmptyState';

describe('EmptyState', () => {
    it('renders title and description', () => {
        render(
            <EmptyState
                title="No experiments yet"
                description="Train an experiment to see results here."
            />,
        );
        expect(screen.getByText('No experiments yet')).toBeInTheDocument();
        expect(
            screen.getByText(/Train an experiment to see results here/i),
        ).toBeInTheDocument();
    });

    it('renders primary and secondary action buttons + wires onClick', async () => {
        const onPrimary = vi.fn();
        const onSecondary = vi.fn();
        render(
            <EmptyState
                title="t"
                description="d"
                primary={{ label: 'Create', onClick: onPrimary }}
                secondary={{ label: 'Import', onClick: onSecondary }}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: 'Create' }));
        await user.click(screen.getByRole('button', { name: 'Import' }));
        expect(onPrimary).toHaveBeenCalledTimes(1);
        expect(onSecondary).toHaveBeenCalledTimes(1);
    });

    it('renders action as an anchor when href is passed', () => {
        render(
            <EmptyState
                title="t"
                description="d"
                primary={{ label: 'Open docs', href: 'https://example.com/docs' }}
            />,
        );
        const link = screen.getByRole('link', { name: 'Open docs' });
        expect(link).toHaveAttribute('href', 'https://example.com/docs');
        expect(link).toHaveAttribute('target', '_blank');
    });

    it('renders the docs link when docsHref is set', () => {
        render(
            <EmptyState
                title="t"
                description="d"
                docsHref="http://localhost:3001/docs/getting-started/quickstart"
            />,
        );
        const link = screen.getByRole('link', { name: /Learn more/i });
        expect(link).toHaveAttribute(
            'href',
            'http://localhost:3001/docs/getting-started/quickstart',
        );
        expect(link).toHaveAttribute('target', '_blank');
    });

    it('still accepts the legacy action prop for back-compat', () => {
        render(
            <EmptyState
                title="t"
                description="d"
                action={<button type="button">Legacy</button>}
            />,
        );
        expect(
            screen.getByRole('button', { name: 'Legacy' }),
        ).toBeInTheDocument();
    });

    it('renders an emoji icon when passed as string', () => {
        render(
            <EmptyState
                title="t"
                description="d"
                icon="📂"
            />,
        );
        // Emoji is rendered inside the icon slot; assert by querying status text.
        const status = screen.getByRole('status');
        expect(status.textContent).toContain('📂');
    });
});
