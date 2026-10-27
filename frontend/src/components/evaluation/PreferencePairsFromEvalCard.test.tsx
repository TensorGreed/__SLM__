import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import PreferencePairsFromEvalCard from './PreferencePairsFromEvalCard';

const PROV = 'Run #21 · gold_dev · result 1001';

describe('PreferencePairsFromEvalCard', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('labels the source and posts the eval result id with both verdicts', async () => {
        apiMock.post.mockResolvedValue({
            data: {
                dataset_id: null, dataset_name: 'pairs-from-eval-1001', dataset_path: '/data/p4/prepared/alignment/eval_1001_pairs.jsonl',
                source_split: 'gold_dev', warning: null, created: 3,
                skipped: { correct: 5, empty_prediction: 1 }, examples: [],
            },
        });
        const user = userEvent.setup();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} provenance={PROV} />);
        expect(screen.getByTestId('pairs-from-eval-provenance-1001')).toHaveTextContent(PROV);
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/4/alignment/preference-pairs/from-eval',
            { eval_result_id: 1001, include: ['wrong', 'partial'] },
        );
        expect(await screen.findByTestId('pairs-created-1001')).toHaveTextContent('3');
        expect(screen.getByTestId('pairs-dataset-1001')).toHaveTextContent('pairs-from-eval-1001');
        expect(screen.getByTestId('pairs-path-1001')).toHaveTextContent('eval_1001_pairs.jsonl');
        expect(screen.getByText(/set it as the alignment dataset path in Train/)).toBeInTheDocument();
        expect(screen.queryByTestId('pairs-warning-1001')).toBeNull();
        expect(screen.getByTestId('pairs-skipped-1001-correct')).toHaveTextContent('Judged correct: 5');
        expect(screen.getByTestId('pairs-skipped-1001-empty_prediction')).toHaveTextContent('Empty prediction: 1');
    });

    it('sends wrong only when partial is unticked', async () => {
        apiMock.post.mockResolvedValue({ data: { dataset_id: null, dataset_name: 'd', created: 1, skipped: {}, examples: [] } });
        const user = userEvent.setup();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} provenance={PROV} />);
        await user.click(screen.getByTestId('pairs-include-partial-1001'));
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(apiMock.post.mock.calls[0][1].include).toEqual(['wrong']);
    });

    it('labels the new skip reasons', async () => {
        apiMock.post.mockResolvedValue({
            data: { dataset_id: null, dataset_name: 'd', created: 1, skipped: { not_included: 2, not_stored: 3 }, examples: [] },
        });
        const user = userEvent.setup();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} provenance={PROV} />);
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(await screen.findByTestId('pairs-skipped-1001-not_included')).toHaveTextContent('Verdict not included: 2');
        expect(screen.getByTestId('pairs-skipped-1001-not_stored')).toHaveTextContent('3');
    });

    it('shows the test-split checkbox only for a test split and sends the override with its warning', async () => {
        apiMock.post.mockResolvedValue({
            data: {
                dataset_id: null, dataset_name: 'd', source_split: 'test', created: 1, skipped: {}, examples: [],
                warning: 'These pairs come from the held-out test split.',
            },
        });
        const user = userEvent.setup();
        const { unmount } = render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} split="validation" provenance={PROV} />);
        expect(screen.queryByTestId('pairs-allow-test-1001')).toBeNull();
        unmount();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} split="test" provenance={PROV} />);
        expect(screen.getByText('Allow test split (measures memorisation)')).toBeInTheDocument();
        await user.click(screen.getByTestId('pairs-allow-test-1001'));
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(apiMock.post.mock.calls[0][1]).toEqual({
            eval_result_id: 1001, include: ['wrong', 'partial'], allow_test_split: true,
        });
        expect(await screen.findByTestId('pairs-warning-1001')).toHaveTextContent('held-out test split');
    });

    it('does not send the override for a test split unless ticked', async () => {
        apiMock.post.mockRejectedValue({ response: { status: 400, data: { error_code: 'test_split_refused', detail: 'ran on the test split: memorisation', metadata: { skipped: {} } } } });
        const user = userEvent.setup();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} split="test" provenance={PROV} />);
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(apiMock.post.mock.calls[0][1]).not.toHaveProperty('allow_test_split');
        expect(await screen.findByTestId('pairs-from-eval-error-1001')).toHaveTextContent('memorisation');
    });

    it('shows the skip breakdown when the server answers 400', async () => {
        apiMock.post.mockRejectedValue({
            response: {
                status: 400,
                data: {
                    detail: 'No preference pairs could be built from eval result 1001.',
                    error_code: 'NO_PAIRS',
                    metadata: { skipped: { not_judged: 7, no_prediction_stored: 2 } },
                },
            },
        });
        const user = userEvent.setup();
        render(<PreferencePairsFromEvalCard projectId={4} evalResultId={1001} provenance={PROV} />);
        await user.click(screen.getByTestId('pairs-from-eval-run-1001'));
        expect(await screen.findByTestId('pairs-from-eval-error-1001')).toHaveTextContent('No preference pairs could be built');
        expect(screen.getByTestId('pairs-error-skipped-1001-not_judged')).toHaveTextContent('Not judged: 7');
        expect(screen.getByTestId('pairs-error-skipped-1001-no_prediction_stored')).toHaveTextContent('No prediction stored: 2');
        expect(screen.queryByTestId('pairs-from-eval-result-1001')).toBeNull();
    });
});
