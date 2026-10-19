import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

const { jobsState } = vi.hoisted(() => ({ jobsState: { jobs: [] as Array<Record<string, unknown>> } }));
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: Object.assign((selector: (s: typeof jobsState) => unknown) => selector(jobsState), { getState: () => jobsState }),
}));

import SeedDocumentsCard from './SeedDocumentsCard';

describe('SeedDocumentsCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        jobsState.jobs = [];
        apiMock.get.mockResolvedValue({ data: { genres: ['customer support ticket', 'clinical visit note'], count: 60, words: 160 } });
    });

    it('starts the seed_documents Job with the kinds, genres and count, then shows the result', async () => {
        apiMock.post.mockResolvedValue({ data: { id: 91, kind: 'seed_documents', status: 'queued', project_id: 5 } });
        const user = userEvent.setup();
        const onDone = vi.fn();
        const { rerender } = render(<SeedDocumentsCard projectId={5} onDone={onDone} />);
        await user.click(screen.getByTestId('seed-documents-toggle'));
        await waitFor(() => expect(screen.getByTestId('seed-documents-genres')).toHaveValue('customer support ticket\nclinical visit note'));
        await user.type(screen.getByTestId('seed-documents-kinds'), 'card numbers, emails');
        await user.clear(screen.getByTestId('seed-documents-count'));
        await user.type(screen.getByTestId('seed-documents-count'), '12');
        await user.click(screen.getByTestId('seed-documents-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/5/flows/seed-documents', {
            kinds: 'card numbers, emails', genres: ['customer support ticket', 'clinical visit note'], count: 12, words: 160,
        }));
        jobsState.jobs = [{ id: 91, kind: 'seed_documents', status: 'running', project_id: 5, progress_message: 'Writing document 3/12: clinical visit note (ollama)' }];
        rerender(<SeedDocumentsCard projectId={5} onDone={onDone} />);
        expect(screen.getByTestId('seed-documents-progress')).toHaveTextContent('Writing document 3/12');
        jobsState.jobs = [{ id: 91, kind: 'seed_documents', status: 'succeeded', project_id: 5, result: { written: 11, failed: 1, backend: 'ollama:gemma4:12b', filename: 'seed-documents-1.txt', passages: 23, warnings: ['1 of 12 documents were empty or failed and were skipped.'] } }];
        rerender(<SeedDocumentsCard projectId={5} onDone={onDone} />);
        expect(screen.getByTestId('seed-documents-result')).toHaveTextContent('Done. 11 documents written (ollama:gemma4:12b) → seed-documents-1.txt, 23 passages. 1 of 12 documents were empty or failed and were skipped.');
        expect(onDone).toHaveBeenCalled();
    });

    it('asks for the chosen layout (multi-page or scanned)', async () => {
        apiMock.post.mockResolvedValue({ data: { id: 92, kind: 'seed_documents', status: 'queued', project_id: 5 } });
        const user = userEvent.setup();
        render(<SeedDocumentsCard projectId={5} />);
        await user.click(screen.getByTestId('seed-documents-toggle'));
        await waitFor(() => expect(screen.getByTestId('seed-documents-genres')).toHaveValue('customer support ticket\nclinical visit note'));
        await user.selectOptions(screen.getByTestId('seed-documents-layout'), 'scanned');
        await user.click(screen.getByTestId('seed-documents-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/5/flows/seed-documents', expect.objectContaining({ layout: 'scanned' })));
    });

    it('dry-runs the prompt against the benchmarks and shows the offending values', async () => {
        apiMock.post.mockResolvedValue({ data: {
            clean: false, offending_total: 1, values_checked: 1480, benchmarks: ['gretel-pii'], fields: ['genres', 'kinds'],
            offending: [{ value: 'EMP-48213', type: 'employee_id', benchmarks: ['gretel-pii'], fields: ['kinds'] }],
        } });
        const user = userEvent.setup();
        render(<SeedDocumentsCard projectId={5} />);
        await user.click(screen.getByTestId('seed-documents-toggle'));
        await user.type(screen.getByTestId('seed-documents-kinds'), 'employee ids like EMP-48213');
        await user.click(screen.getByTestId('seed-documents-check'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/5/synthetic/prompt-check', {
            fields: { kinds: 'employee ids like EMP-48213', genres: ['customer support ticket', 'clinical visit note'] }, include_conventions: false,
        }));
        expect(screen.getByTestId('seed-documents-check-result')).toHaveTextContent(
            '1 benchmark value in the prompt: EMP-48213 (employee_id, in what the documents must contain). A generation model copies examples word for word',
        );
    });

    it('shows a refusal with the values and resends with the override on "Write anyway"', async () => {
        const refusal = { response: { status: 400, data: {
            error_code: 'PROMPT_CARRIES_BENCHMARK_VALUES', detail: 'The prompt carries 1 value(s) from a benchmark\'s gold rows: EMP-48213 (employee_id, gretel-pii).',
            message: 'The prompt carries 1 value(s) from a benchmark\'s gold rows: EMP-48213 (employee_id, gretel-pii).',
            metadata: { offending: [{ value: 'EMP-48213', type: 'employee_id', benchmarks: ['gretel-pii'], fields: ['kinds'] }], override_field: 'ignore_benchmark_values' },
        } } };
        apiMock.post.mockRejectedValueOnce(refusal).mockResolvedValueOnce({ data: { id: 93, kind: 'seed_documents', status: 'queued', project_id: 5 } });
        const user = userEvent.setup();
        render(<SeedDocumentsCard projectId={5} />);
        await user.click(screen.getByTestId('seed-documents-toggle'));
        await waitFor(() => expect(screen.getByTestId('seed-documents-genres')).toHaveValue('customer support ticket\nclinical visit note'));
        await user.type(screen.getByTestId('seed-documents-kinds'), 'use EMP-48213');
        await user.click(screen.getByTestId('seed-documents-start'));
        expect(await screen.findByTestId('seed-documents-blocked')).toHaveTextContent('EMP-48213 (employee_id, gretel-pii)');
        await user.click(screen.getByTestId('seed-documents-write-anyway'));
        await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith('/projects/5/flows/seed-documents', expect.objectContaining({
            kinds: 'use EMP-48213', ignore_benchmark_values: true,
        })));
        await waitFor(() => expect(screen.queryByTestId('seed-documents-blocked')).toBeNull());
    });
    it('sends the cloud model pin when one is picked', async () => {
        apiMock.get.mockImplementation((url: string) => Promise.resolve({ data: url.includes('/backends/cloud/models')
            ? { project_id: 5, providers: [{ provider: 'anthropic', key_saved: true, models: [{ id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' }] }] }
            : { genres: ['customer support ticket'], count: 60, words: 160 } }));
        apiMock.post.mockResolvedValue({ data: { id: 92, kind: 'seed_documents', status: 'queued', project_id: 5 } });
        const user = userEvent.setup();
        render(<SeedDocumentsCard projectId={5} />);
        await user.click(screen.getByTestId('seed-documents-toggle'));
        await waitFor(() => expect(screen.getByTestId('seed-documents-provider')).toHaveTextContent('anthropic (key saved)'));
        await user.selectOptions(screen.getByTestId('seed-documents-provider'), 'anthropic');
        await user.selectOptions(screen.getByTestId('seed-documents-cloud-model'), 'claude-sonnet-5-5');
        await user.click(screen.getByTestId('seed-documents-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/5/flows/seed-documents',
            expect.objectContaining({ backend: 'cloud:anthropic:claude-sonnet-5-5' })));
    });
});
