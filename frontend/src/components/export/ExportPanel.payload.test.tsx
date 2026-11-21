/**
 * Ship › Package: the export bodies (UI simplification pass, phase 5). Written and run against the form
 * BEFORE the short Ship form (all fields on one card, empty "Experiment" select, quantization default
 * 4-bit) and kept unchanged after it: the same choices must send the same `/export/create` and
 * `/export/{id}/run` bodies, including the hidden defaults (the recommended deployment targets, smoke
 * tests on). The helpers only skip a click when the form already shows the wanted value (a preselected
 * run, a prefilled quantization) and open the Advanced disclosure when there is one.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ExportPanel from './ExportPanel';

const RUNS = [
    { id: 140, name: 'run-140', status: 'completed' },
    { id: 141, name: 'run-141', status: 'failed' },
    { id: 142, name: 'lift-baseline', status: 'completed', config: { is_baseline: true } },
    { id: 143, name: 'run-143', status: 'completed' },
];

function target(id: string, smoke = true) {
    return { target_id: id, kind: 'runner', display_name: id, description: `${id} runtime`, compatible: true, smoke_supported: smoke };
}

const CATALOG: Record<string, { default_target_ids: string[]; targets: ReturnType<typeof target>[] }> = {
    gguf: { default_target_ids: ['runner.ollama', 'runner.llamacpp', 'exporter.gguf'],
        targets: [target('runner.ollama'), target('runner.llamacpp'), target('exporter.gguf', false), target('runner.lmstudio')] },
    huggingface: { default_target_ids: ['runner.vllm', 'runner.tgi', 'exporter.hf'],
        targets: [target('runner.vllm'), target('runner.tgi'), target('exporter.hf', false)] },
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string, config?: { params?: { export_format?: string } }) => {
        if (url.includes('/training/experiments')) return { data: RUNS };
        if (url.includes('/export/list')) return { data: [] };
        if (url.includes('/registry/models')) return { data: { models: [] } };
        if (url.includes('/export/deployment-targets')) {
            return { data: CATALOG[config?.params?.export_format || ''] || { default_target_ids: [], targets: [] } };
        }
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string) => {
        if (url.endsWith('/export/create')) return { data: { id: 77 } };
        if (url.endsWith('/export/77/run')) return { data: { id: 77, status: 'completed', output_path: '/x', manifest: {} } };
        return { data: {} };
    });
});

function selectByLabel(label: RegExp): HTMLSelectElement {
    const el = screen.getAllByText(label).find((node) => node.tagName === 'LABEL');
    const select = el?.parentElement?.querySelector('select');
    if (!select) throw new Error(`no select labelled ${label}`);
    return select as HTMLSelectElement;
}

async function setUp() {
    const user = userEvent.setup();
    render(<ExportPanel projectId={31} stage="package" />);
    await screen.findByTestId('export-form-card');
    await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/31/training/experiments'));
    await waitFor(() => expect(selectByLabel(/^(Experiment|Run)$/).querySelector('option[value="143"]')).not.toBeNull());
    return user;
}

async function pick(user: ReturnType<typeof userEvent.setup>, label: RegExp, value: string) {
    const select = selectByLabel(label);
    if (select.value !== value) await user.selectOptions(select, value);
}

async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
    const toggle = screen.queryByTestId('advanced-ship-export-advanced-toggle');
    if (toggle && !screen.queryByText('Deployment Targets')) await user.click(toggle);
}

async function exportBodies(user: ReturnType<typeof userEvent.setup>) {
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run Export' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Run Export' }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
    const [createUrl, createBody] = apiMock.post.mock.calls[0];
    const [runUrl, runBody] = apiMock.post.mock.calls[1];
    return { createUrl, createBody, runUrl, runBody };
}

describe('ExportPanel export bodies (pinned before the short Ship form)', () => {
    it('GGUF, the newest completed run, defaults: the 3 recommended targets and smoke tests are sent', async () => {
        const user = await setUp();
        await pick(user, /^(Experiment|Run)$/, '143');
        const bodies = await exportBodies(user);
        expect(bodies).toEqual({
            createUrl: '/projects/31/export/create',
            createBody: { experiment_id: 143, export_format: 'gguf', quantization: '4-bit' },
            runUrl: '/projects/31/export/77/run',
            runBody: { deployment_targets: ['runner.ollama', 'runner.llamacpp', 'exporter.gguf'], run_smoke_tests: true, benchmark_report: null },
        });
    });

    it('HuggingFace with quantization None: the HF recommended targets', async () => {
        const user = await setUp();
        await pick(user, /^(Experiment|Run)$/, '143');
        await pick(user, /^Format$/, 'huggingface');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/31/export/deployment-targets', { params: { export_format: 'huggingface' } }));
        await pick(user, /^Quantization$/, 'none');
        const bodies = await exportBodies(user);
        expect(bodies.createBody).toEqual({ experiment_id: 143, export_format: 'huggingface', quantization: 'none' });
        expect(bodies.runBody).toEqual({ deployment_targets: ['runner.vllm', 'runner.tgi', 'exporter.hf'], run_smoke_tests: true, benchmark_report: null });
    });

    it('an older run, 8-bit, one target added and smoke tests off (Advanced)', async () => {
        const user = await setUp();
        await pick(user, /^(Experiment|Run)$/, '140');
        await pick(user, /^Quantization$/, '8-bit');
        await openAdvanced(user);
        await user.click(await screen.findByText('runner.lmstudio'));
        await user.click(screen.getByText('Run local smoke tests for selected runner targets'));
        const bodies = await exportBodies(user);
        expect(bodies.createBody).toEqual({ experiment_id: 140, export_format: 'gguf', quantization: '8-bit' });
        expect(bodies.runBody).toEqual({
            deployment_targets: ['runner.ollama', 'runner.llamacpp', 'exporter.gguf', 'runner.lmstudio'],
            run_smoke_tests: false, benchmark_report: null,
        });
    });
});
