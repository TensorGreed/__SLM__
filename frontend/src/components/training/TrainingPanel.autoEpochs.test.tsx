/**
 * Auto epochs survives the prefill (phase 7i fix pass, A). Coming back to Train after a run exists prefills the
 * form from that run's stored config. The fill turned the profile-defaults switch off and marked every key touched,
 * so the next create sent ``num_epochs: 3, auto_epochs: false`` + the stored gradient accumulation: a 108-row run
 * trained 21 steps instead of the auto-planned 216, with no edit. A run that had auto epochs on keeps it on; a run
 * with explicit epochs keeps them; an epochs edit is always exactly what was typed.
 */

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
import { TRAINING_OVERRIDES_APPLIED_EVENT } from '../../api/trainingConfigGaps';

type User = ReturnType<typeof userEvent.setup>;
const BASE = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const AUTO_HINT = 'Auto: scaled to your dataset size unless you change it.';

/** A run config as the backend stores it (the full schema dump: num_epochs 3 + auto_epochs true). */
function storedRun(config: Record<string, unknown>) {
    return {
        id: 41, name: 'old-run', status: 'completed', base_model: BASE, created_at: '2026-11-20T00:00:00Z',
        config: { learning_rate: 0.0002, batch_size: 4, gradient_accumulation_steps: 4, optimizer: 'adamw_torch', ...config },
    };
}

function installApi(opts: { runs?: unknown[]; overrides?: Record<string, unknown> } = {}) {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.endsWith('/training/experiments')) return { data: opts.runs ?? [] };
        if (url.endsWith('/training-config-gaps/overrides')) return { data: { project_id: 1, overrides: opts.overrides ?? {} } };
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

async function ready() {
    await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
}

async function openModelSection(user: User) {
    if (!screen.queryByTestId('train-section-model')) await user.click(screen.getByTestId('train-form-more'));
    const toggle = screen.getByTestId('train-section-model');
    if (toggle.getAttribute('aria-expanded') !== 'true') await user.click(toggle);
}

function epochsInput(): HTMLInputElement {
    return screen.getAllByText('Epochs', { selector: 'label' })[0].parentElement!.querySelector('input')!;
}

async function createBody(user: User): Promise<Record<string, unknown>> {
    apiMock.post.mockClear();
    await user.click(screen.getByTestId('train-form-train'));
    await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
    const call = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
    return (call![1] as { config: Record<string, unknown> }).config;
}

describe('Train form keeps auto epochs through the prefill (phase 7i fix pass)', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        try { window.localStorage.clear(); window.sessionStorage.clear(); } catch { /* storage unavailable */ }
    });

    it('prefill from a run with auto epochs on: no num_epochs, no auto_epochs false, no gradient accumulation', async () => {
        installApi({ runs: [storedRun({ num_epochs: 3, auto_epochs: true })] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByText('old-run');
        await ready();
        await openModelSection(user);
        expect(screen.getByText(AUTO_HINT)).toBeInTheDocument();
        const config = await createBody(user);
        expect(config).not.toHaveProperty('num_epochs');
        expect(config).not.toHaveProperty('auto_epochs');
        expect(config).not.toHaveProperty('gradient_accumulation_steps');
        // The rest of the run's config still rides.
        expect(config.learning_rate).toBe(0.0002);
        expect(config.optimizer).toBe('adamw_torch');
    });

    it('prefill from a run with explicit epochs (auto_epochs false) keeps that exact value', async () => {
        installApi({ runs: [storedRun({ num_epochs: 5, auto_epochs: false })] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByText('old-run');
        await ready();
        await openModelSection(user);
        expect(screen.queryByText(AUTO_HINT)).toBeNull();
        const config = await createBody(user);
        expect(config.num_epochs).toBe(5);
        expect(config.auto_epochs).toBe(false);
        expect(config.gradient_accumulation_steps).toBe(4);
    });

    it('an epochs edit after an auto prefill is exactly what was typed', async () => {
        installApi({ runs: [storedRun({ num_epochs: 3, auto_epochs: true })] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByText('old-run');
        await ready();
        await openModelSection(user);
        fireEvent.change(epochsInput(), { target: { value: '6' } });
        expect(screen.queryByText(AUTO_HINT)).toBeNull();
        const config = await createBody(user);
        expect(config.num_epochs).toBe(6);
        expect(config.auto_epochs).toBe(false);
    });

    it('a gap-panel override that names no epochs leaves auto epochs on, after a prefill too', async () => {
        installApi({ runs: [storedRun({ num_epochs: 3, auto_epochs: true })] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByText('old-run');
        await ready();
        act(() => {
            window.dispatchEvent(new CustomEvent(TRAINING_OVERRIDES_APPLIED_EVENT, {
                detail: { projectId: 1, overrides: { lora_alpha: 64 } },
            }));
        });
        const config = await createBody(user);
        expect(config.lora_alpha).toBe(64);
        expect(config).not.toHaveProperty('num_epochs');
        expect(config).not.toHaveProperty('auto_epochs');
    });

    it('a config from before auto epochs (num_epochs, no auto_epochs) stays explicit, as train.py reads it', async () => {
        installApi({ runs: [storedRun({ num_epochs: 2 })] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByText('old-run');
        await ready();
        const config = await createBody(user);
        expect(config.num_epochs).toBe(2);
        expect(config.auto_epochs).toBe(false);
    });
});
