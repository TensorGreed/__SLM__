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

// The REAL on-disk prepared/manifest.json shape: top-level seed/chat_template
// + ratios:{train,val,test}. (split_dataset does NOT persist
// resolved_split_config — that key only exists on the split API response, so
// the hydration must read this shape or ratios/seed silently fail to load.)
const ACTIVE_MANIFEST = {
    stratify_by: 'label',
    disjoint_by: null,
    seed: 7,
    chat_template: 'llama3',
    ratios: { train: 0.7, val: 0.15, test: 0.15 },
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
            <DatasetPrepPanel projectId={42} onNextStep={() => undefined} />
        </MemoryRouter>,
    );
    // The split form is the page itself (no view tabs any more).
    await screen.findByText(/Train \/ Val \/ Test Split/i);
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

    it('sends keep_test when "Keep the current test examples" is ticked', async () => {
        mockApi({ ...ACTIVE_MANIFEST, splits: { train: 341, val: 42, test: 44 } });
        apiMock.post.mockResolvedValue({ data: { splits: { train: 400, val: 50, test: 44 } } });
        const user = await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        const keep = await screen.findByTestId('split-keep-test');
        await user.click(keep);
        await user.click(screen.getByRole('button', { name: /Run Split/ }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/42/dataset/split', expect.objectContaining({ keep_test: true }),
        ));
    });

    it('offers "Replace my own test file" only when the active split is predefined, and sends discard_predefined_test', async () => {
        mockApi({ ...ACTIVE_MANIFEST, split_source: 'predefined', splits: { train: 341, val: 42, test: 44 } });
        apiMock.post.mockResolvedValue({ data: { splits: { train: 400, val: 50, test: 44 } } });
        const user = await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        expect(screen.getByTestId('split-replace-own-test-note')).toHaveTextContent(
            "Re-splitting re-draws validation from train + val; your own val file's membership is not kept.",
        );
        await user.click(await screen.findByTestId('split-discard-predefined-test'));
        await user.click(screen.getByRole('button', { name: /Run Split/ }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/42/dataset/split', expect.objectContaining({ discard_predefined_test: true }),
        ));
    });

    it('keeps offering "Replace my own test file" after a keep-test re-split (predefined_test_active, no split_source)', async () => {
        mockApi({ ...ACTIVE_MANIFEST, predefined_test_active: true, splits: { train: 341, val: 42, test: 44 } });
        await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        expect(screen.getByTestId('split-discard-predefined-test')).toBeInTheDocument();
        expect(screen.getByTestId('split-keep-test')).toBeInTheDocument();
    });

    it('drops the replace option once the replace re-split returns a manifest without predefined_test_active', async () => {
        mockApi({ ...ACTIVE_MANIFEST, predefined_test_active: true, splits: { train: 341, val: 42, test: 44 } });
        apiMock.post.mockResolvedValue({ data: { splits: { train: 400, val: 50, test: 50 } } });
        const user = await renderAndOpenSplit();
        await user.click(await screen.findByTestId('split-discard-predefined-test'));
        await user.click(screen.getByRole('button', { name: /Run Split/ }));
        await waitFor(() => expect(screen.queryByTestId('split-discard-predefined-test')).not.toBeInTheDocument());
    });

    it('shows the replace option when the split fails with the would-be-replaced 409', async () => {
        mockApi({ ...ACTIVE_MANIFEST, splits: { train: 341, val: 42, test: 44 } });
        apiMock.post.mockRejectedValue({
            response: { data: { error_code: 'PREDEFINED_TEST_SPLIT_WOULD_BE_REPLACED', detail: 'keep or replace' } },
        });
        const user = await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        expect(screen.queryByTestId('split-discard-predefined-test')).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: /Run Split/ }));
        expect(await screen.findByTestId('split-discard-predefined-test')).toBeInTheDocument();
    });

    it('does not offer the replace-test option for a random split', async () => {
        mockApi({ ...ACTIVE_MANIFEST, splits: { train: 341, val: 42, test: 44 } });
        await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        expect(screen.queryByTestId('split-discard-predefined-test')).not.toBeInTheDocument();
    });

    it('shows no reuse hint on a fresh project with no active prepared version', async () => {
        mockApi({}); // empty manifest
        await renderAndOpenSplit();
        // Split form is visible…
        expect(await screen.findByText(/Train \/ Val \/ Test Split/i)).toBeInTheDocument();
        // …but no hydration happened, so no hint.
        expect(screen.queryByTestId('dp-active-config-hint')).not.toBeInTheDocument();
    });

    it('shows how many rows a random split left out because their input was their answer, by reason', async () => {
        mockApi({ ...ACTIVE_MANIFEST, splits: { train: 341, val: 42, test: 44 } });
        apiMock.post.mockResolvedValue({ data: {
            splits: { train: 30, val: 4, test: 4 },
            include_types_resolution: { input_rejected: { classification_input_is_label: 5, structured_input_contains_gold: 2 } },
        } });
        const user = await renderAndOpenSplit();
        await screen.findByTestId('dp-active-config-hint');
        await user.click(screen.getByRole('button', { name: /Run Split/ }));
        const note = await screen.findByTestId('dp-input-rejected-note');
        expect(note).toHaveTextContent('Left out 7 row(s)');
        expect(note).toHaveTextContent('classification input is label (5)');
        expect(note).toHaveTextContent('structured input contains gold (2)');
    });
});
