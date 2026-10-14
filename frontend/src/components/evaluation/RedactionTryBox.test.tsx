import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import RedactionTryBox from './RedactionTryBox';

const RESPONSE = {
    project_id: 7,
    experiment_id: 58,
    experiment_name: 'deberta tc',
    mode: 'mask',
    spans: [
        { type: 'phone', start: 5, end: 13, text: '555-0101', score: 0.96, min_score: 0.95 },
        { type: 'person_name', start: 22, end: 34, text: 'Ada Lovelace', score: 0.875, min_score: 0.85 },
    ],
    text: 'call [PHONE] and ask [PERSON_NAME]',
    offsets: [],
    settings: { min_score: null, recall_bias: 0.2, per_type: {}, thresholds_file: true, max_length: 512 },
};

beforeEach(() => {
    apiMock.post.mockReset();
});

describe('RedactionTryBox', () => {
    it('sends the text, mode and recall bias and shows the masked text with scored spans', async () => {
        apiMock.post.mockResolvedValue({ data: RESPONSE });
        render(<RedactionTryBox projectId={7} experimentId={58} />);
        const button = screen.getByRole('button', { name: 'Redact' });
        expect(button).toBeDisabled();
        await userEvent.type(screen.getByLabelText('Text to redact'), 'call 555-0101 and ask Ada Lovelace');
        fireEvent.change(screen.getByLabelText('Recall bias'), { target: { value: '0.2' } });
        await userEvent.click(button);
        await waitFor(() => expect(screen.getByTestId('redaction-try-output')).toHaveTextContent('call [PHONE] and ask [PERSON_NAME]'));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/7/redact', {
            text: 'call 555-0101 and ask Ada Lovelace',
            experiment_id: 58,
            mode: 'mask',
            recall_bias: 0.2,
        });
        expect(screen.getByText('Ada Lovelace')).toBeInTheDocument();
        expect(screen.getByText('0.96')).toBeInTheDocument();
        expect(screen.getByText(/Run #58 \(deberta tc\)/)).toHaveTextContent('thresholds.json applied');
    });

    it('shows the server message when there is no span run', async () => {
        apiMock.post.mockRejectedValue({ response: { data: { detail: 'This project has no completed span-tagging (token classification) run yet.' } } });
        render(<RedactionTryBox projectId={7} experimentId={null} />);
        await userEvent.type(screen.getByLabelText('Text to redact'), 'hello');
        await userEvent.click(screen.getByRole('button', { name: 'Redact' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('no completed span-tagging');
        expect(apiMock.post.mock.calls[0][1].recall_bias).toBeNull();
    });
});
