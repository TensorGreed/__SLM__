import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import PlaybookPickerPanel from './PlaybookPickerPanel';
import SynthRejectSummary from './SynthRejectSummary';

const TAXONOMY = {
    taxonomy: {
        name: 'toy', source_version: null,
        entries: [
            { id: 'injection', title: 'Injection', parent: null, aliases: [], attrs: {}, crosswalk: {} },
            { id: 'injection.query', title: 'Query injection', parent: 'injection', aliases: [], attrs: { severity: 'high' }, crosswalk: {} },
        ],
    },
    version: 'v3', entry_count: 2, namespaces: [], history: [],
};

function stub() {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/synthetic/playbooks')) {
            return { data: { project_id: 1, recipe_id: 'code-findings', playbooks: [
                { recipe_id: 'code-findings', mode: 'seeded_defect' },
                { recipe_id: 'code-findings', mode: 'hard_negatives' },
            ] } };
        }
        if (url.endsWith('/label-taxonomy')) return { data: TAXONOMY };
        if (url.endsWith('/projects/1/provenance/sources')) {
            return { data: { sources: [
                { id: 3, title: 'samples-repo', source_kind: 'git_repo', locator: '', usage: 'any', document_count: 120 },
                { id: 5, title: 'helm-charts', source_kind: 'git_repo', locator: '', usage: 'any', document_count: 4 },
            ], totals: {} } };
        }
        if (url.endsWith('/projects/1/datasets')) {
            return { data: { datasets: [
                { id: 7, name: 'schemas', dataset_type: 'raw', record_count: 10 },
                { id: 9, name: 'answer key', dataset_type: 'gold_test', record_count: 50 },
            ] } };
        }
        if (url.endsWith('/projects/1/datasets/7/columns')) return { data: { dataset_id: 7, columns: ['table_name', 'field_name'], sampled_rows: 10 } };
        if (url.includes('/synthetic/backends/ollama/models')) return { data: { project_id: 1, default: null, models: [], ollama_available: false } };
        if (url.includes('/synthetic/backends/cloud/models')) return { data: { project_id: 1, providers: [] } };
        if (url.includes('/synthetic/backends')) return { data: { project_id: 1, backends: [] } };
        if (url.includes('/jobs/42')) {
            return { data: { id: 42, status: 'succeeded', result: {
                rejected_count: 3, reject_summary: { unchanged_code: 2, parse_error: 1 },
                rejected_examples: [{ reason: 'unchanged_code', detail: 'identical to the clean code', payload: { code: 'x = 1' } }],
            } } };
        }
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string) => {
        if (url.endsWith('/dry-run')) return { data: { ok: true, accepted_count: 1, refusal_detected: false, raw_llm_snippet: '', backend_used: 'b', elapsed_sec: 0.1, prompt_snippet: '', rows: [] } };
        return { data: { id: 42, kind: 'synth_playbook', status: 'queued' } };
    });
}

describe('code-findings seeded defect', () => {
    beforeEach(() => { apiMock.get.mockReset(); apiMock.post.mockReset(); stub(); });

    it('sends label, prefilled attrs and data context as params, then shows rejects', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><PlaybookPickerPanel projectId={1} /></MemoryRouter>);
        await user.click(await screen.findByTestId('playbook-picker-mode-seeded_defect'));
        const run = await screen.findByTestId('playbook-picker-run');
        expect(run).toBeDisabled();

        await user.selectOptions(await screen.findByTestId('cf-label'), 'injection.query');
        expect(screen.getByDisplayValue('severity')).toBeInTheDocument();
        expect(screen.getByDisplayValue('high')).toBeInTheDocument();
        await user.selectOptions(screen.getByTestId('cf-dataset'), '7');
        expect(screen.queryByRole('option', { name: /answer key/ })).not.toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('option', { name: 'field_name' })).toBeInTheDocument());
        await user.selectOptions(screen.getByTestId('cf-column'), 'field_name');
        expect(screen.getByTestId('cf-context-provenance')).toHaveTextContent('dataset #7 "schemas", column field_name');

        await waitFor(() => expect(run).toBeEnabled());
        await user.click(run);

        await waitFor(() => {
            const call = apiMock.post.mock.calls.find(([u]) => String(u).includes('async_job=true'));
            expect(call?.[1]).toMatchObject({
                mode: 'seeded_defect',
                params: {
                    label_id: 'injection.query', attrs: { severity: 'high' },
                    context_dataset_id: 7, context_column: 'field_name', context_sample_size: 20,
                },
            });
        });
        const summary = await screen.findByTestId('synth-reject-summary');
        expect(summary).toHaveTextContent('3 candidates rejected');
        expect(screen.getByTestId('synth-reject-unchanged_code')).toHaveTextContent('2');
    });
});

describe('seed sources on the picker', () => {
    beforeEach(() => { apiMock.get.mockReset(); apiMock.post.mockReset(); stub(); });

    it('sends the chosen sources as params.source_ids', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><PlaybookPickerPanel projectId={1} /></MemoryRouter>);
        await user.click(await screen.findByTestId('playbook-picker-mode-hard_negatives'));
        await user.click(await screen.findByTestId('seed-source-5'));
        await user.click(await screen.findByTestId('playbook-picker-run'));
        await waitFor(() => {
            const call = apiMock.post.mock.calls.find(([u]) => String(u).includes('async_job=true'));
            expect(call?.[1]).toMatchObject({ mode: 'hard_negatives', params: { source_ids: [5] } });
        });
    });
});

describe('code-findings seeded defect without a taxonomy', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.post.mockReset(); stub();
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/label-taxonomy')) return { data: null };
            return base(url);
        });
    });

    it('accepts a typed label and sends it as label_id', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><PlaybookPickerPanel projectId={1} /></MemoryRouter>);
        await user.click(await screen.findByTestId('playbook-picker-mode-seeded_defect'));
        await user.type(await screen.findByTestId('cf-free-label'), 'secret.hardcoded');
        const run = await screen.findByTestId('playbook-picker-run');
        await waitFor(() => expect(run).toBeEnabled());
        await user.click(run);
        await waitFor(() => {
            const call = apiMock.post.mock.calls.find(([u]) => String(u).includes('async_job=true'));
            expect(call?.[1]).toMatchObject({ mode: 'seeded_defect', params: { label_id: 'secret.hardcoded' } });
        });
    });
});

describe('SynthRejectSummary', () => {
    it('renders nothing without rejects and expands examples', () => {
        const { container, rerender } = render(<SynthRejectSummary result={{ rejected_count: 0, reject_summary: {} }} />);
        expect(container).toBeEmptyDOMElement();
        rerender(<SynthRejectSummary result={{
            rejected_count: 1, reject_summary: { code_too_long: 1 },
            rejected_examples: [{ reason: 'code_too_long', detail: '900 lines', payload: { code: 'big()' } }],
        }} />);
        expect(screen.getByTestId('synth-reject-code_too_long')).toHaveTextContent('Code too long');
        expect(screen.getByText('900 lines')).toBeInTheDocument();
        expect(screen.getByText('big()')).toBeInTheDocument();
    });
});
