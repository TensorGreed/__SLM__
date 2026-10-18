/**
 * The Train form layout (Epic U phase U2): readiness from the preflight,
 * Train = create + launch confirmation, More options reach the panel's own
 * sections, the runs list sits below.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./PreRunConfirmModal', () => ({
    default: ({ onConfirm, baseModel }: { onConfirm: () => void; baseModel: string }) => (
        <div data-testid="pre-run-modal">
            {baseModel}
            <button type="button" onClick={onConfirm}>Launch</button>
        </div>
    ),
}));
vi.mock('./CloudBurstPlanningSection', () => ({ default: () => <div data-testid="cloud-burst" /> }));
// After Launch the panel switches to the run view, whose plan panel expects a real estimate.
vi.mock('./WhyThisPlanPanel', () => ({ default: () => <div data-testid="why-this-plan" /> }));
vi.mock('../shared/ReadinessPanel', () => ({ ReadinessPanel: () => <div data-testid="readiness-panel" /> }));

import TrainingPanel from './TrainingPanel';

const EXPERIMENTS = [
    { id: 41, name: 'old-run', status: 'completed', base_model: 'microsoft/deberta-v3-base', config: {}, created_at: '2026-10-01T00:00:00Z' },
];

function installApi(preflight: { ok: boolean; errors: string[]; warnings: string[] }) {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.endsWith('/training/experiments')) return { data: EXPERIMENTS };
        if (url.includes('/training/experiments/41/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
        if (url.includes('/training/experiments/77/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string, body?: { name?: string; config?: Record<string, unknown> }) => {
        if (url.endsWith('/training/experiments/preflight')) {
            return { data: { preflight, resolved_training_config: {}, profile_defaults_applied: [] } };
        }
        if (url.endsWith('/training/experiments')) {
            return { data: { id: 77, name: body?.name, status: 'created', base_model: body?.config?.base_model ?? 'x', config: body?.config ?? {} } };
        }
        if (url.endsWith('/training/experiments/77/start')) return { data: { task_id: 't1' } };
        return { data: {} };
    });
}

describe('TrainingPanel — form layout', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        installApi({ ok: true, errors: [], warnings: [] });
    });

    it('checks readiness on mount and shows the runs list below the form', async () => {
        render(<TrainingPanel projectId={1} title="Runs" taskShapeLabel="Span extraction (span-extraction)" />);
        expect(screen.getByTestId('train-form')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/1/training/experiments/preflight', expect.anything());
        expect(screen.getByTestId('train-form-task')).toHaveTextContent('Span extraction (span-extraction) · causal_lm');
        expect(await screen.findByText('old-run')).toBeInTheDocument();
        // No setup tabs, no journey strip.
        expect(screen.queryByRole('tab', { name: /Power Tools/i })).toBeNull();
        expect(screen.queryByText('+ New Experiment')).toBeNull();
    });

    it('Train creates the run (dated name when blank, seeds in the config) and opens the launch confirmation', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        await user.click(screen.getByTestId('train-form-seeds-3'));
        await user.click(screen.getByTestId('train-form-train'));
        await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
        const createCall = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
        expect(createCall).toBeTruthy();
        const body = createCall![1] as { name: string; config: Record<string, unknown> };
        expect(body.name).toMatch(/^run-\d{4}-\d{2}-\d{2}-\d{4}$/);
        expect(body.config.num_seeds).toBe(3);
        await user.click(screen.getByRole('button', { name: 'Launch' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/1/training/experiments/77/start'));
    });

    it('a blocked preflight disables nothing but shows the issues, and Fix opens the Checks section', async () => {
        installApi({ ok: false, errors: ['No prepared split'], warnings: [] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('1 blocking issue.'));
        expect(screen.getByTestId('train-form-issues')).toHaveTextContent('No prepared split');
        await user.click(screen.getByTestId('train-form-fix'));
        expect(await screen.findByTestId('readiness-panel')).toBeInTheDocument();
    });

    it('More options exposes LoRA, seeds, cloud burst and the extra sections; ?open= opens one', async () => {
        const user = userEvent.setup();
        render(
            <TrainingPanel
                projectId={1}
               
                openSection="lora"
                moreOptions={[{ id: 'models', title: 'Base model registry', render: () => <div data-testid="registry-body" /> }]}
            />,
        );
        expect(await screen.findByTestId('training-config-curriculum-row')).toBeInTheDocument();
        await user.click(screen.getByTestId('train-section-cloud'));
        expect(screen.getByTestId('cloud-burst')).toBeInTheDocument();
        await user.click(screen.getByTestId('train-section-seeds'));
        expect(screen.getByTestId('training-multi-seed-section')).toBeInTheDocument();
        expect(screen.queryByTestId('registry-body')).toBeNull();
        await user.click(screen.getByTestId('train-form-browse'));
        expect(await screen.findByTestId('registry-body')).toBeInTheDocument();
    });
});
