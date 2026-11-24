/**
 * Payload equivalence for the Train form (UI simplification pass, phase 4).
 *
 * The snapshot in __snapshots__/TrainingPanel.payload.test.tsx.snap was WRITTEN BY THIS TEST AT
 * 7c973df + phase 3 (before More options was split into Common / Advanced and before scoring settings
 * left Train). Phase 4 moves sections and changes no field, so every create body must still equal it.
 * Never update the snapshot to make a moved control pass: a diff here is a payload change.
 *
 * The helper opens a section wherever it lives (More options, or More options › Advanced), so the
 * same steps run against the old and the new layout.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
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

function installApi(experiments: unknown[] = []) {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.endsWith('/training/experiments')) return { data: experiments };
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

/** Open a More-options section, wherever it lives. */
async function openSection(user: User, id: string) {
    if (!screen.queryByTestId(`train-section-${id}`) && !screen.queryByTestId('advanced-train-advanced-toggle')) {
        await user.click(screen.getByTestId('train-form-more'));
    }
    if (!screen.queryByTestId(`train-section-${id}`)) {
        await user.click(screen.getByTestId('advanced-train-advanced-toggle'));
    }
    const toggle = screen.getByTestId(`train-section-${id}`);
    if (toggle.getAttribute('aria-expanded') !== 'true') await user.click(toggle);
}

/** The input / select next to a label in the open section. */
function field(label: string): HTMLInputElement | HTMLSelectElement {
    const el = screen.getAllByText(label, { selector: 'label' })[0].parentElement!;
    return (el.querySelector('input, select') as HTMLInputElement | HTMLSelectElement);
}

async function trainBody(user: User): Promise<{ name: string; config: Record<string, unknown> }> {
    apiMock.post.mockClear();
    await user.click(screen.getByTestId('train-form-train'));
    await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
    // Phase 7k: the confirmation opens on the form's settings; nothing is created until Launch.
    expect(apiMock.post.mock.calls.some(([url]) => url === '/projects/1/training/experiments')).toBe(false);
    await user.click(screen.getByRole('button', { name: 'Launch' }));
    await waitFor(() => expect(apiMock.post.mock.calls.some(([url]) => url === '/projects/1/training/experiments')).toBe(true));
    const call = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
    const body = call![1] as { name: string; config: Record<string, unknown> };
    return { name: /^run-\d{4}-\d{2}-\d{2}-\d{4}$/.test(body.name) ? '<dated>' : body.name, config: body.config };
}

async function ready() {
    await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
}

describe('Train form create bodies are unchanged by the phase 4 layout', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        try { window.localStorage.clear(); } catch { /* storage unavailable */ }
        installApi();
    });

    it('defaults: Train with nothing touched', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} defaultBaseModel="HuggingFaceTB/SmolLM2-135M-Instruct" />);
        await ready();
        expect(await trainBody(user)).toMatchSnapshot();
    });

    it('page one: three seeds and a run name', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} defaultBaseModel="HuggingFaceTB/SmolLM2-135M-Instruct" />);
        await ready();
        await user.click(screen.getByTestId('train-form-seeds-3'));
        await user.type(screen.getByTestId('train-form-run-name'), 'my-run');
        expect(await trainBody(user)).toMatchSnapshot();
    });

    it('Common: epochs, learning rate, optimizer, LoRA rank, base seed (typed over the clamped inputs)', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} defaultBaseModel="HuggingFaceTB/SmolLM2-135M-Instruct" />);
        await ready();
        await openSection(user, 'model');
        const epochs = field('Epochs') as HTMLInputElement;
        await user.clear(epochs);
        await user.type(epochs, '5');
        const lr = field('Learning Rate') as HTMLInputElement;
        await user.clear(lr);
        await user.type(lr, '0.0001');
        const optimizer = field('Optimizer') as HTMLSelectElement;
        const other = Array.from(optimizer.options).map((o) => o.value).find((v) => v !== optimizer.value)!;
        await user.selectOptions(optimizer, other);
        await openSection(user, 'lora');
        const rank = field('Rank (r)') as HTMLInputElement;
        await user.clear(rank);
        await user.type(rank, '32');
        await openSection(user, 'seeds');
        const seedsSection = document.getElementById('train-section-seeds')!;
        const expand = within(seedsSection).queryByLabelText('Expand multi-seed section');
        if (expand) await user.click(expand);
        const baseSeed = field('Base seed') as HTMLInputElement;
        await user.clear(baseSeed);
        await user.type(baseSeed, '7');
        expect(await trainBody(user)).toMatchSnapshot();
    });

    it('Advanced: domain runtime defaults off sends every field with its default, while the section is closed again', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} defaultBaseModel="HuggingFaceTB/SmolLM2-135M-Instruct" />);
        await ready();
        await openSection(user, 'presets');
        await user.click(screen.getByLabelText('Use domain runtime defaults for untouched fields'));
        // Close it: the value lives in the panel and is still sent.
        await user.click(screen.getByTestId('train-section-presets'));
        expect(await trainBody(user)).toMatchSnapshot();
    });

    it('prefilled from the latest run: Train sends its config', async () => {
        installApi([
            { id: 41, name: 'old-run', status: 'completed', base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct', created_at: '2026-10-01T00:00:00Z',
                config: { num_epochs: 2, learning_rate: 0.0002, lora_r: 8, optimizer: 'adamw_torch', chat_template: 'chatml' } },
        ]);
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await ready();
        await screen.findByText('old-run');
        expect(await trainBody(user)).toMatchSnapshot();
    });
});
