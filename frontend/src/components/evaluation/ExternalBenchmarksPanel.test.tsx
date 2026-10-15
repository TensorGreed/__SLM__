import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ExternalBenchmarksPanel from './ExternalBenchmarksPanel';

const yelp = {
    id: 5, name: 'Yelp stars', slug: 'yelp-stars', task_profile: 'classification',
    source: { locator: 'hf:Yelp/yelp_review_full:test', source_id: 'hf', dataset_id: 'Yelp/yelp_review_full', split: 'test', revision: null },
    licence: 'other', local_only: false, rows: 240, mapping: { '1 star': 'negative' }, scored_labels: ['negative', 'positive'],
    leakage: { checked_against: 800, exact: 0, near: 2 }, created_at: '2026-10-15',
    result: { experiment_id: 80, evaluated_at: 'x', rows: 240, headline: 'accuracy', score: 0.912 }, latest_result: null,
};

describe('ExternalBenchmarksPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('lists benchmarks with provenance and this run\'s score, and scores a run as a job', async () => {
        apiMock.get.mockResolvedValue({ data: { benchmarks: [yelp] } });
        apiMock.post.mockResolvedValue({ data: { id: 301 } });
        const user = userEvent.setup();
        render(<ExternalBenchmarksPanel projectId={33} experimentId={80} />);
        const item = await screen.findByTestId('ext-bench-yelp-stars');
        expect(item).toHaveTextContent('Yelp/yelp_review_full:test · licence other · 240 rows · scores negative, positive · 2 rows dropped (in your training data)');
        expect(screen.getByTestId('ext-bench-score-yelp-stars')).toHaveTextContent('Run #80: accuracy 0.91 on 240 rows');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/33/benchmarks', { params: { experiment_id: 80 } });
        await user.click(screen.getByTestId('ext-bench-evaluate-yelp-stars'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/33/benchmarks/5/evaluate', { experiment_id: 80 }));
        expect(screen.getByTestId('ext-bench-score-yelp-stars')).toHaveTextContent('scoring (job #301)');
    });

    it('previews a source, maps labels many-to-one with ignore, and creates the benchmark', async () => {
        apiMock.get.mockResolvedValue({ data: { benchmarks: [] } });
        apiMock.post.mockImplementation((url: string) => {
            if (url.endsWith('/preview')) {
                return Promise.resolve({ data: {
                    source: { locator: 'hf:Yelp/yelp_review_full:test', source_id: 'hf', dataset_id: 'Yelp/yelp_review_full', split: 'test', revision: null },
                    licence: 'other', task_profile: 'classification', format: 'text_label', fields: {},
                    project: { task_profile: 'classification', labels: ['negative', 'positive'] }, shape_matches_project: true,
                    source_labels: { '1 star': 60, '3 stars': 60, '5 stars': 60 },
                    suggested_mapping: { '1 star': 'ignore', '3 stars': 'ignore', '5 stars': 'ignore' }, sample: [], rows_scanned: 1000,
                } });
            }
            return Promise.resolve({ data: { ...yelp, result: null } });
        });
        const user = userEvent.setup();
        render(<ExternalBenchmarksPanel projectId={33} experimentId={null} />);
        await user.click(await screen.findByTestId('ext-bench-add-toggle'));
        await user.type(screen.getByTestId('ext-bench-locator'), 'hf:Yelp/yelp_review_full:test');
        await user.click(screen.getByTestId('ext-bench-preview'));
        expect(await screen.findByTestId('ext-bench-preview-result')).toHaveTextContent('Detected classification');
        expect(screen.getByTestId('ext-bench-create')).toHaveTextContent('Map at least one label');
        await user.selectOptions(screen.getByTestId('ext-bench-map-1 star'), 'negative');
        await user.selectOptions(screen.getByTestId('ext-bench-map-5 stars'), 'positive');
        await user.click(screen.getByTestId('ext-bench-local-only'));
        await user.click(screen.getByTestId('ext-bench-create'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/33/benchmarks', {
            locator: 'hf:Yelp/yelp_review_full:test', name: 'yelp_review_full', limit: 300,
            mapping: { '1 star': 'negative', '3 stars': 'ignore', '5 stars': 'positive' },
            local_only: true, licence: 'other', join_adjacent: false,
        }));
    });
});
