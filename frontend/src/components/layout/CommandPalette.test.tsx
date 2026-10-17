import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import CommandPalette from './CommandPalette';
import { openCommandPalette } from './commandPaletteBridge';

function renderPalette(
    projectId: number | null = 7,
    initialPath = '/project/7/training-config',
) {
    return render(
        <MemoryRouter initialEntries={[initialPath]}>
            <Routes>
                <Route
                    path="*"
                    element={
                        <>
                            <CommandPalette projectId={projectId} />
                            <span data-testid="loc">{initialPath}</span>
                        </>
                    }
                />
            </Routes>
        </MemoryRouter>,
    );
}

describe('CommandPalette', () => {
    it('opens on Cmd-K and shows project-scoped nav actions', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        expect(
            await screen.findByRole('dialog', { name: /Command palette/i }),
        ).toBeInTheDocument();
        // Multiple sections.
        expect(screen.getByText('Navigation')).toBeInTheDocument();
        expect(screen.getByText('Workspace')).toBeInTheDocument();
        expect(screen.getByText('Overview')).toBeInTheDocument();
        // A few well-known actions.
        expect(screen.getByText('Data Studio')).toBeInTheDocument();
        expect(screen.getByText('Annotation')).toBeInTheDocument();
        expect(screen.getByText('Observability')).toBeInTheDocument();
    });

    it('also opens on Ctrl-K (for non-mac users)', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Control>}k{/Control}');
        expect(
            await screen.findByRole('dialog', { name: /Command palette/i }),
        ).toBeInTheDocument();
    });

    it('offers every tool regardless of beginner mode (nothing is hidden since Epic U)', async () => {
        renderPalette(7);
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        expect(screen.getByText('Adapter Studio')).toBeInTheDocument();
        expect(screen.getByText('Extension Studio')).toBeInTheDocument();
        expect(screen.getByText('Workflow Builder')).toBeInTheDocument();
        expect(screen.getByText('Domain Packs')).toBeInTheDocument();
    });

    it('filters by substring match', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        await user.type(
            screen.getByRole('textbox', { name: /Command query/i }),
            'observ',
        );
        expect(screen.getByText('Observability')).toBeInTheDocument();
        expect(screen.queryByText('Data Studio')).not.toBeInTheDocument();
    });

    it('shows the empty state when nothing matches', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        await user.type(
            screen.getByRole('textbox', { name: /Command query/i }),
            'zzzzzz-no-such-thing',
        );
        expect(screen.getByText(/No matches/i)).toBeInTheDocument();
    });

    it('closes on Escape', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        await user.keyboard('{Escape}');
        await waitFor(() => {
            expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        });
    });

    it('Enter selects the highlighted item and closes', async () => {
        renderPalette();
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        // Type narrow query so the first highlighted result is predictable.
        await user.type(
            screen.getByRole('textbox', { name: /Command query/i }),
            'observ',
        );
        await user.keyboard('{Enter}');
        await waitFor(() => {
            expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        });
    });

    it('opens imperatively via openCommandPalette()', async () => {
        renderPalette();
        // Initially closed.
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        openCommandPalette();
        expect(
            await screen.findByRole('dialog', { name: /Command palette/i }),
        ).toBeInTheDocument();
    });

    it('when projectId is null only the Back-to-projects action is offered', async () => {
        renderPalette(null);
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        expect(screen.getByText('Back to projects')).toBeInTheDocument();
        expect(screen.queryByText('Data Studio')).not.toBeInTheDocument();
    });
});
