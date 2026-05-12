import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import CommandPalette from './CommandPalette';
import { openCommandPalette } from './commandPaletteBridge';

function renderPalette(
    projectId: number | null = 7,
    beginnerMode = false,
    initialPath = '/project/7/training-config',
) {
    return render(
        <MemoryRouter initialEntries={[initialPath]}>
            <Routes>
                <Route
                    path="*"
                    element={
                        <>
                            <CommandPalette
                                projectId={projectId}
                                beginnerMode={beginnerMode}
                            />
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
        expect(screen.getByText('Training')).toBeInTheDocument();
        // A few well-known actions.
        expect(screen.getByText('Pipeline Runs')).toBeInTheDocument();
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

    it('hides beginner-only actions when beginnerMode is true', async () => {
        renderPalette(7, true);
        const user = userEvent.setup();
        await user.keyboard('{Meta>}k{/Meta}');
        await screen.findByRole('dialog');
        // Always visible.
        expect(screen.getByText('Pipeline Runs')).toBeInTheDocument();
        // Hidden when beginner.
        expect(screen.queryByText('Adapter Studio')).not.toBeInTheDocument();
        expect(screen.queryByText('Extension Studio')).not.toBeInTheDocument();
        expect(screen.queryByText('Workflow Builder')).not.toBeInTheDocument();
        expect(screen.queryByText('Domain Packs')).not.toBeInTheDocument();
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
        expect(screen.queryByText('Pipeline Runs')).not.toBeInTheDocument();
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
        expect(screen.queryByText('Pipeline Runs')).not.toBeInTheDocument();
    });
});
