import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import CommandSnippet from './CommandSnippet';

describe('CommandSnippet', () => {
    it('starts collapsed and expands on click', async () => {
        render(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id foo"
                api={{ method: 'POST', path: '/extensions/scaffold', body: { kind: 'data_adapter', plugin_id: 'foo' } }}
            />,
        );

        // Collapsed: only the toggle is visible.
        expect(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        ).toBeInTheDocument();
        expect(screen.queryByRole('tab', { name: 'CLI' })).not.toBeInTheDocument();

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );

        // Expanded: tabs + code block.
        expect(screen.getByRole('tab', { name: 'CLI' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'API' })).toBeInTheDocument();
        expect(
            screen.getByText('brewslm scaffold adapter --plugin-id foo'),
        ).toBeInTheDocument();
    });

    it('switches between CLI and API tabs', async () => {
        render(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id foo"
                api={{ method: 'POST', path: '/extensions/scaffold', body: { kind: 'data_adapter' } }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );

        // Default tab is CLI.
        const panel = screen.getByRole('tabpanel');
        expect(panel.textContent).toContain('brewslm scaffold adapter');
        expect(panel.textContent).not.toContain('curl -X POST');

        // Switch to API.
        await user.click(screen.getByRole('tab', { name: 'API' }));

        const apiPanel = screen.getByRole('tabpanel');
        expect(apiPanel.textContent).toContain('curl -X POST');
        expect(apiPanel.textContent).not.toContain('brewslm scaffold adapter');
    });

    it('renders the API path with the default host', async () => {
        render(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id foo"
                api={{ method: 'POST', path: '/extensions/scaffold', body: { kind: 'data_adapter' } }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );
        await user.click(screen.getByRole('tab', { name: 'API' }));

        // Default API base hint.
        const panel = screen.getByRole('tabpanel');
        expect(panel.textContent).toContain(
            'http://localhost:8000/api/extensions/scaffold',
        );
    });

    it('omits the body block on GET', async () => {
        render(
            <CommandSnippet
                cli="brewslm extensions list"
                api={{ method: 'GET', path: '/extensions' }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );
        await user.click(screen.getByRole('tab', { name: 'API' }));

        // curl -X GET http://localhost:8000/api/extensions
        // No `-d` flag should appear.
        const code = screen.getByRole('tabpanel');
        expect(code.textContent).toContain('curl -X GET');
        expect(code.textContent).not.toContain('-d ');
    });

    it('copy button shows a status hint after click', async () => {
        // jsdom doesn't ship a working Clipboard implementation and the
        // overrides interact poorly with the readonly getter on the
        // Navigator prototype across vitest versions. We don't try to
        // assert writeText was called — instead, verify the copy click
        // surfaces a transient status node (the visible feedback users
        // actually see).
        render(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id foo"
                api={{ method: 'POST', path: '/extensions/scaffold', body: { kind: 'data_adapter' } }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );
        await user.click(screen.getByRole('button', { name: /Copy CLI snippet/i }));
        const status = await screen.findByRole('status');
        expect(status.textContent).toMatch(/Copied|failed/);
    });

    it('close button collapses the panel back to the toggle', async () => {
        render(
            <CommandSnippet
                cli="brewslm extensions reload"
                api={{ method: 'POST', path: '/extensions/reload', body: {} }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );
        expect(screen.getByRole('tab', { name: 'CLI' })).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Hide snippet/i }));
        expect(screen.queryByRole('tab', { name: 'CLI' })).not.toBeInTheDocument();
        expect(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        ).toBeInTheDocument();
    });

    it('updates the CLI text when the prop changes', async () => {
        const { rerender } = render(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id one"
                api={{ method: 'POST', path: '/extensions/scaffold', body: {} }}
            />,
        );

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Show this action as CLI or API/i }),
        );
        expect(
            screen.getByText('brewslm scaffold adapter --plugin-id one'),
        ).toBeInTheDocument();

        rerender(
            <CommandSnippet
                cli="brewslm scaffold adapter --plugin-id two"
                api={{ method: 'POST', path: '/extensions/scaffold', body: {} }}
            />,
        );

        expect(
            screen.getByText('brewslm scaffold adapter --plugin-id two'),
        ).toBeInTheDocument();
    });

    it('respects a custom label on the toggle', () => {
        render(
            <CommandSnippet
                cli="x"
                api={{ method: 'GET', path: '/x' }}
                label="See it as a script"
            />,
        );
        expect(screen.getByText('See it as a script')).toBeInTheDocument();
    });
});
