import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import AnswerKeyReviewPanel, { segmentText } from './AnswerKeyReviewPanel';
import { agreementText, reviewProgressText } from '../../api/goldReview';

const T1 = 'Call Ada Lovelace at ada@example.test.';
const T2 = 'Patient Bob Stone, MRN 112233.';

function row(index: number, text: string, entities: Array<{ type: string; start: number; end: number }>, predicted: typeof entities | null, agreement: number | null, review: null | { verdict: 'accepted' | 'corrected' | 'rejected'; reviewed_at: string; reviewed_by: null; note: string }) {
    return { index, key: `k${index}`, text, entities, fields: {}, source: 'answer_key:cloud:anthropic:claude-sonnet-5-5', review, predicted, agreement };
}

const QUEUE = {
    project_id: 31, dataset_type: 'gold_test', model_experiment_id: 117, last_reviewed_at: null,
    total: 2, reviewed: 0, accepted: 0, corrected: 0, rejected: 0, unreviewed: 2,
    rows: [
        // Disagreement first: the model misses the MRN on row 1.
        row(1, T2, [{ type: 'person_name', start: 8, end: 17 }, { type: 'medical_record_number', start: 23, end: 29 }], [{ type: 'person_name', start: 8, end: 17 }], 0.6667, null),
        row(0, T1, [{ type: 'person_name', start: 5, end: 17 }, { type: 'email', start: 21, end: 37 }], [{ type: 'person_name', start: 5, end: 17 }, { type: 'email', start: 21, end: 37 }], 1, null),
    ],
};

describe('segmentText', () => {
    it('cuts at key and model boundaries and keeps the whole text', () => {
        const segs = segmentText('abcdefgh', [{ type: 'x', start: 1, end: 4 }], [{ type: 'y', start: 3, end: 6 }]);
        expect(segs.map((s) => [s.start, s.end, s.key?.type ?? null, s.model?.type ?? null])).toEqual([
            [0, 1, null, null], [1, 3, 'x', null], [3, 4, 'x', 'y'], [4, 6, null, 'y'], [6, 8, null, null],
        ]);
    });
});

describe('text helpers', () => {
    it('reads progress and agreement', () => {
        expect(reviewProgressText({ total: 43, reviewed: 12, accepted: 8, corrected: 3, rejected: 1, unreviewed: 31 })).toBe('12 of 43 checked · 3 corrected · 1 rejected');
        expect(agreementText(QUEUE.rows[0] as never)).toBe('model disagrees (agreement 0.67)');
        expect(agreementText(QUEUE.rows[1] as never)).toBe('model agrees');
    });
});

describe('AnswerKeyReviewPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        apiMock.get.mockResolvedValue({ data: QUEUE });
    });

    it('lists rows disagreement-first with the key spans marked', async () => {
        render(<AnswerKeyReviewPanel projectId={31} />);
        expect(await screen.findByTestId('akr-progress')).toHaveTextContent('0 of 2 checked');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/31/gold/review', { params: { dataset_type: 'gold_test' } });
        const rows = screen.getAllByTestId(/^akr-row-/);
        expect(rows[0]).toHaveAttribute('data-testid', 'akr-row-1');
        expect(screen.getByTestId('akr-status-1')).toHaveTextContent('model disagrees (agreement 0.67)');
        // The MRN the model missed is marked as such; the name both agree on is plain green.
        const text = screen.getByTestId('akr-text-1');
        const missed = text.querySelector('.akr__span--missed');
        expect(missed).toHaveTextContent('112233');
        expect(text.querySelectorAll('.akr__span').length).toBe(2);
        expect(screen.getByText(/run #117 disagrees with the key come first/)).toBeInTheDocument();
    });

    it('flags a row whose earlier verdict a relabel could not carry', async () => {
        apiMock.get.mockResolvedValue({ data: { ...QUEUE, rows: [{ ...QUEUE.rows[1], previous_review: { verdict: 'accepted', reviewed_at: 'x', note: '', reason: 'the labels changed in a relabel' } }] } });
        render(<AnswerKeyReviewPanel projectId={31} />);
        expect(await screen.findByTestId('akr-previous-0')).toHaveTextContent('Previously accepted with different labels (the labels changed in a relabel) — check it again.');
    });

    it('accept sends the verdict and moves the row out of the pending list', async () => {
        apiMock.put.mockResolvedValue({
            data: {
                row: { ...QUEUE.rows[1], review: { verdict: 'accepted', reviewed_at: '2026-10-16T00:00:00Z', reviewed_by: null, note: 'fine' } },
                total: 2, reviewed: 1, accepted: 1, corrected: 0, rejected: 0, unreviewed: 1, last_reviewed_at: '2026-10-16T00:00:00Z',
            },
        });
        const onChanged = vi.fn();
        render(<AnswerKeyReviewPanel projectId={31} onChanged={onChanged} />);
        await screen.findByTestId('akr-row-0');
        fireEvent.change(screen.getAllByLabelText('Review note')[1], { target: { value: 'fine' } });
        fireEvent.click(screen.getByTestId('akr-accept-0'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31/gold/review', {
            dataset_type: 'gold_test', index: 0, key: 'k0', verdict: 'accepted', corrected: undefined, note: 'fine',
        }));
        await waitFor(() => expect(screen.getByTestId('akr-progress')).toHaveTextContent('1 of 2 checked'));
        expect(screen.queryByTestId('akr-row-0')).not.toBeInTheDocument();
        expect(onChanged).toHaveBeenCalled();
        fireEvent.click(screen.getByTestId('akr-toggle-reviewed'));
        expect(screen.getByTestId('akr-status-0')).toHaveTextContent('reviewed: accepted');
        expect(screen.getByTestId('akr-clear-0')).toBeInTheDocument();
    });

    it('fix edits the spans inline and saves a correction', async () => {
        apiMock.put.mockResolvedValue({
            data: {
                row: { ...QUEUE.rows[0], entities: [{ type: 'person_name', start: 8, end: 17 }], review: { verdict: 'corrected', reviewed_at: 'x', reviewed_by: null, note: '' } },
                total: 2, reviewed: 1, accepted: 0, corrected: 1, rejected: 0, unreviewed: 1, last_reviewed_at: 'x',
            },
        });
        render(<AnswerKeyReviewPanel projectId={31} />);
        await screen.findByTestId('akr-row-1');
        fireEvent.click(screen.getByTestId('akr-fix-1'));
        const edit = screen.getByTestId('akr-edit-1');
        expect(edit.querySelectorAll('.akr__edit-row').length).toBe(2);
        // Drop the MRN (the second span), then save.
        fireEvent.click(screen.getAllByLabelText('Remove span')[1]);
        expect(edit.querySelectorAll('.akr__edit-row').length).toBe(1);
        fireEvent.click(screen.getByTestId('akr-save-1'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31/gold/review', {
            dataset_type: 'gold_test', index: 1, key: 'k1', verdict: 'corrected',
            corrected: { entities: [{ type: 'person_name', start: 8, end: 17 }] }, note: '',
        }));
        await waitFor(() => expect(screen.getByTestId('akr-progress')).toHaveTextContent('1 corrected'));
    });

    it('"use the model\'s spans" copies the prediction into the draft', async () => {
        render(<AnswerKeyReviewPanel projectId={31} />);
        await screen.findByTestId('akr-row-1');
        fireEvent.click(screen.getByTestId('akr-fix-1'));
        fireEvent.click(screen.getByTestId('akr-use-model-1'));
        expect(screen.getByTestId('akr-edit-1').querySelectorAll('.akr__edit-row').length).toBe(1);
    });

    it('shows the server error on a refused verdict', async () => {
        apiMock.put.mockRejectedValue({ response: { status: 400, data: { detail: 'the answer key changed since you loaded it — reload the queue' } } });
        render(<AnswerKeyReviewPanel projectId={31} />);
        await screen.findByTestId('akr-row-1');
        fireEvent.click(screen.getByTestId('akr-reject-1'));
        expect(await screen.findByTestId('akr-error')).toHaveTextContent('reload the queue');
    });

    it('renders nothing for an empty key', async () => {
        apiMock.get.mockResolvedValue({ data: { ...QUEUE, total: 0, unreviewed: 0, rows: [] } });
        const { container } = render(<AnswerKeyReviewPanel projectId={31} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        await waitFor(() => expect(container.querySelector('[data-testid="answer-key-review"]')).toBeNull());
    });
});
