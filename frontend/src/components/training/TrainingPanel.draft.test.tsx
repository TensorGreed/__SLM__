/**
 * Unsaved Train form edits survive leaving Train (UI simplification phase 7b, gap N5): a draft per project in
 * sessionStorage, restored once after the prefill with a notice and Discard; the create body is the one the
 * edits would have sent; a form nobody edited keeps no draft; a created run clears it.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./PreRunConfirmModal', () => ({
    default: ({ onConfirm }: { onConfirm: () => void }) => (
        <div data-testid="pre-run-modal">
            <button type="button" onClick={onConfirm}>Launch</button>
        </div>
    ),
}));
vi.mock('./CloudBurstPlanningSection', () => ({ default: () => <div data-testid="cloud-burst" /> }));
vi.mock('./WhyThisPlanPanel', () => ({ default: () => <div data-testid="why-this-plan" /> }));
vi.mock('../shared/ReadinessPanel', () => ({ ReadinessPanel: () => <div data-testid="readiness-panel" /> }));

import TrainingPanel from './TrainingPanel';

type User = ReturnType<typeof userEvent.setup>;
const BASE = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const KEY = 'brewslm.trainDraft.1';

function installApi() {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.endsWith('/training/experiments')) return { data: [] };
        if (url.includes('/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string, body?: { name?: string; config?: Record<string, unknown> }) => {
        if (url.endsWith('/training/experiments/preflight')) {
            return { data: { preflight: { ok: true, errors: [], warnings: [] }, resolved_training_config: {}, profile_defaults_applied: [] } };
        }
        if (url.endsWith('/training/experiments')) {
            return { data: { id: 77, name: body?.name, status: 'pending', base_model: body?.config?.base_model ?? 'x', config: body?.config ?? {} } };
        }
        return { data: {} };
    });
}

async function openModelSection(user: User) {
    if (!screen.queryByTestId('train-section-model')) await user.click(screen.getByTestId('train-form-more'));
    const toggle = screen.getByTestId('train-section-model');
    if (toggle.getAttribute('aria-expanded') !== 'true') await user.click(toggle);
}

async function ready() {
    await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
}

async function trainConfig(user: User): Promise<{ name: string; config: Record<string, unknown> }> {
    apiMock.post.mockClear();
    await user.click(screen.getByTestId('train-form-train'));
    await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
    const call = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
    return call![1] as { name: string; config: Record<string, unknown> };
}

describe('Train form unsaved edits (phase 7b, N5)', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        try { window.localStorage.clear(); window.sessionStorage.clear(); } catch { /* storage unavailable */ }
        installApi();
    });

    it('edits survive an unmount (leaving Train) and send the same body; Discard goes back', async () => {
        const user = userEvent.setup();
        const first = render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await ready();
        await openModelSection(user);
        // The optimizer select is labelled (phase 7b, N9).
        expect(screen.getByLabelText('Optimizer')).toBeInstanceOf(HTMLSelectElement);
        const epochs = screen.getAllByText('Epochs', { selector: 'label' })[0].parentElement!.querySelector('input')!;
        await user.clear(epochs);
        await user.type(epochs, '5');
        await user.type(screen.getByTestId('train-form-run-name'), 'my-run');
        await waitFor(() => expect(window.sessionStorage.getItem(KEY)).toContain('"my-run"'));
        const edited = await trainConfig(user);
        // Leave without launching: the run was created, so the draft is gone.
        expect(window.sessionStorage.getItem(KEY)).toBeNull();
        first.unmount();

        // Edit again, leave, come back.
        installApi();
        const second = render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await ready();
        expect(screen.queryByTestId('train-form-draft-restored')).toBeNull();
        await openModelSection(user);
        const epochs2 = screen.getAllByText('Epochs', { selector: 'label' })[0].parentElement!.querySelector('input')!;
        await user.clear(epochs2);
        await user.type(epochs2, '5');
        await user.type(screen.getByTestId('train-form-run-name'), 'my-run');
        await waitFor(() => expect(window.sessionStorage.getItem(KEY)).toContain('"my-run"'));
        second.unmount();

        installApi();
        render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        expect(await screen.findByTestId('train-form-draft-restored')).toHaveTextContent('Kept your unsaved edits');
        expect(screen.getByTestId('train-form-run-name')).toHaveValue('my-run');
        await ready();
        const restored = await trainConfig(user);
        expect(restored).toEqual(edited);
    });

    it('Discard puts the form back as it was prefilled and forgets the draft', async () => {
        const user = userEvent.setup();
        const first = render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await ready();
        await user.type(screen.getByTestId('train-form-run-name'), 'scratch');
        await waitFor(() => expect(window.sessionStorage.getItem(KEY)).toContain('"scratch"'));
        first.unmount();

        installApi();
        render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await screen.findByTestId('train-form-draft-restored');
        await user.click(screen.getByTestId('train-form-draft-discard'));
        expect(screen.queryByTestId('train-form-draft-restored')).toBeNull();
        expect(screen.getByTestId('train-form-run-name')).toHaveValue('');
        await waitFor(() => expect(window.sessionStorage.getItem(KEY)).toBeNull());
    });

    it('a form nobody edited keeps no draft, and another project never sees this one', async () => {
        const first = render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await ready();
        expect(window.sessionStorage.getItem(KEY)).toBeNull();
        first.unmount();
        window.sessionStorage.setItem('brewslm.trainDraft.2', JSON.stringify({ values: { num_epochs: 9 }, name: 'other' }));
        installApi();
        render(<TrainingPanel projectId={1} defaultBaseModel={BASE} />);
        await ready();
        expect(screen.queryByTestId('train-form-draft-restored')).toBeNull();
        expect(screen.getByTestId('train-form-run-name')).toHaveValue('');
    });
});
