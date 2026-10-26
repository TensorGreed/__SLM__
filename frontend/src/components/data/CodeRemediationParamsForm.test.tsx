import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import CodeRemediationParamsForm from './CodeRemediationParamsForm';

describe('CodeRemediationParamsForm', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.get.mockResolvedValue({
            data: {
                datasets: [
                    { id: 1, name: 'findings-import', dataset_type: 'train', record_count: 12 },
                    { id: 2, name: 'dev', dataset_type: 'gold_dev', record_count: 5 },
                ],
            },
        });
    });

    it('offers non-answer-key datasets and emits findings_dataset_id + top_k', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        render(<CodeRemediationParamsForm projectId={4} mode="remediation_from_findings" onChange={onChange} />);
        await screen.findByRole('option', { name: 'findings-import (dataset #1)' });
        expect(screen.queryByRole('option', { name: /dev \(dataset/ })).not.toBeInTheDocument();
        expect(onChange).toHaveBeenLastCalledWith({ top_k: 3 });
        await user.selectOptions(screen.getByTestId('cr-dataset'), '1');
        expect(onChange).toHaveBeenLastCalledWith({ top_k: 3, findings_dataset_id: 1 });
        expect(screen.getByTestId('cr-provenance')).toHaveTextContent('dataset #1 "findings-import" (12 rows)');
    });

    it('emits per_chunk for code_examples without fetching datasets', async () => {
        const onChange = vi.fn();
        render(<CodeRemediationParamsForm projectId={4} mode="code_examples" onChange={onChange} />);
        await waitFor(() => expect(onChange).toHaveBeenCalledWith({ per_chunk: 2 }));
        const input = screen.getByTestId('ce-per-chunk');
        await userEvent.clear(input);
        await userEvent.type(input, '9');
        expect(onChange).toHaveBeenLastCalledWith({ per_chunk: 5 });
        expect(apiMock.get).not.toHaveBeenCalled();
    });
});
