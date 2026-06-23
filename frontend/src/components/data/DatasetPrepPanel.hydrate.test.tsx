import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

// Per-URL api mock: most endpoints return benign empty data; the
// prepared-manifest endpoint returns the active prepared version's config,
// which the panel hydrates the split form from on mount.
const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../utils/workflowGraphPrefill', () => ({
    loadWorkflowStagePrefill: vi.fn().mockResolvedValue(null),
}));

import DatasetPrepPanel from './DatasetPrepPanel';

const ACTIVE_MANIFEST = {
    stratify_by: 'label',
    disjoint_by: null,
    resolved_split_config: {
        train_ratio: 0.7,
        val_ratio: 0.15,
        test_ratio: 0.15,
        seed: 7,
        chat_template: 'llama3',
    },
};

function mockApi(manifest: Record<string, unknown>) {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/prepared-manifest')) return Promise.resolve({ data: manifest });
        if (url.includes('/dataset/adapter-preference')) return Promise.resolve({ data: {} });
        if (url.includes('/dataset/adapter-catalog')) return Promise.resolve({ data: { adapters: [], default_adapter: 'default-canonical' } });
        if (url.includes('/data-health')) return Promise.resolve({ data: { groups: [], overall: 'ok', severity_summary: { ok: 0, warn: 0, block: 0 }, total_signals: 0 } });
        return Promise.resolve({ data: {} });
    });
    apiMock.post.mockResolvedValue({ data: {} });
}

async function renderAndOpenSplit() {
    const user = userEvent.setup();
    render(
        <MemoryRouter>
            <DatasetPrepPanel projectId={42} />
        </MemoryRouter>,
    );
    // The split form (and the hydration hint) only render under the Split view.
    await user.click(await screen.findByRole('button', { name: 'Split' }));
    return user;
}

describe('DatasetPrepPanel — split-form hydration from the active prepared version', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('hydrates stratify/ratios/seed from the active manifest and shows the reuse hint', async () => {
        mockApi(ACTIVE_MANIFEST);
        await renderAndOpenSplit();

        // The reuse hint names the inherited fields.
        const hint = await screen.findByTestId('dp-active-config-hint');
        expect(hint).toHaveTextContent(/Reusing the active prepared version's split config/i);
        expect(hint).toHaveTextContent('stratify_by');
        expect(hint).toHaveTextContent('seed');
        expect(hint).toHaveTextContent('train_ratio');

        // The form fields reflect the active config, not empty defaults.
        const stratifyInput = screen.getByPlaceholderText(/blank = uniform random/i) as HTMLInputElement;
        expect(stratifyInput.value).toBe('label');
        // Profile-defaults is switched off so a manual Run Split sends the
        // hydrated ratios/seed rather than falling back to profile defaults.
        const profileToggle = screen.getByRole('checkbox') as HTMLInputElement;
        expect(profileToggle.checked).toBe(false);
    });

    it('shows no reuse hint on a fresh project with no active prepared version', async () => {
        mockApi({}); // empty manifest
        await renderAndOpenSplit();
        // Split form is visible…
        expect(await screen.findByText(/Train \/ Val \/ Test Split/i)).toBeInTheDocument();
        // …but no hydration happened, so no hint.
        expect(screen.queryByTestId('dp-active-config-hint')).not.toBeInTheDocument();
    });
});
