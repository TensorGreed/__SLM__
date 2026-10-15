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

    const spanPreview = {
        source: { locator: 'file:/x/i2b2.jsonl', source_id: 'file', dataset_id: '/x/i2b2.jsonl', split: null, revision: null },
        licence: 'DUA', task_profile: 'structured_extraction', format: 'char_spans', fields: {},
        project: { task_profile: 'structured_extraction', labels: ['date_of_birth', 'person_name', 'visit_date'] }, shape_matches_project: true,
        labels_to_map: true, source_labels: { DATE: 40, DOB: 5, NAME: 30 },
        suggested_mapping: { DATE: 'ignore', DOB: 'ignore', NAME: 'ignore' }, sample: [], rows_scanned: 200,
    };

    it('maps one benchmark label onto a group of project labels and flags a label in two groups', async () => {
        apiMock.get.mockResolvedValue({ data: { benchmarks: [] } });
        apiMock.post.mockImplementation((url: string) => Promise.resolve({ data: url.endsWith('/preview') ? spanPreview : { ...yelp, result: null } }));
        const user = userEvent.setup();
        render(<ExternalBenchmarksPanel projectId={33} experimentId={null} />);
        await user.click(await screen.findByTestId('ext-bench-add-toggle'));
        await user.type(screen.getByTestId('ext-bench-locator'), 'file:/x/i2b2.jsonl');
        await user.click(screen.getByTestId('ext-bench-preview'));
        await screen.findByTestId('ext-bench-preview-result');
        await user.selectOptions(screen.getByTestId('ext-bench-map-DATE'), ['visit_date', 'date_of_birth']);
        expect(screen.getByTestId('ext-bench-map-result-DATE')).toHaveTextContent('→ group: date_of_birth + visit_date');
        await user.selectOptions(screen.getByTestId('ext-bench-map-DOB'), ['date_of_birth']);
        expect(screen.getByTestId('ext-bench-map-conflict')).toHaveTextContent('“date_of_birth” is in two groups');
        expect(screen.getByTestId('ext-bench-create')).toBeDisabled();
        await user.deselectOptions(screen.getByTestId('ext-bench-map-DOB'), ['date_of_birth']);
        await user.selectOptions(screen.getByTestId('ext-bench-map-NAME'), ['person_name']);
        expect(screen.queryByTestId('ext-bench-map-conflict')).toBeNull();
        await user.click(screen.getByTestId('ext-bench-create'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/33/benchmarks', expect.objectContaining({
            mapping: { DATE: ['date_of_birth', 'visit_date'], DOB: 'ignore', NAME: 'person_name' },
        })));
    });

    it('says a Q&A benchmark has no labels to map and lets it be added', async () => {
        apiMock.get.mockResolvedValue({ data: { benchmarks: [] } });
        apiMock.post.mockImplementation((url: string) => Promise.resolve({ data: url.endsWith('/preview') ? {
            ...spanPreview, task_profile: 'qa', format: 'qa_context', fields: { question: 'question', answer: 'answers', context: 'context' },
            project: { task_profile: 'qa', labels: [] }, labels_to_map: false, source_labels: {}, suggested_mapping: {},
            source: { ...spanPreview.source, dataset_id: 'rajpurkar/squad' },
        } : { ...yelp, result: null } }));
        const user = userEvent.setup();
        render(<ExternalBenchmarksPanel projectId={33} experimentId={null} />);
        await user.click(await screen.findByTestId('ext-bench-add-toggle'));
        await user.type(screen.getByTestId('ext-bench-locator'), 'hf:rajpurkar/squad:validation');
        await user.click(screen.getByTestId('ext-bench-preview'));
        expect(await screen.findByTestId('ext-bench-preview-result')).toHaveTextContent('Detected Q&A');
        expect(screen.getByTestId('ext-bench-no-labels')).toHaveTextContent('No labels to map');
        expect(screen.queryByTestId('ext-bench-map-DATE')).toBeNull();
        expect(screen.getByTestId('ext-bench-create')).toHaveTextContent('Add benchmark');
        await user.click(screen.getByTestId('ext-bench-create'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/33/benchmarks', expect.objectContaining({ mapping: {}, name: 'squad' })));
    });

    it('shows each benchmark\'s score history as a trend and a judged Q&A score', async () => {
        const qa = {
            ...yelp, id: 7, name: 'SQuAD', slug: 'squad', task_profile: 'qa', scored_labels: [], mapping: {}, local_only: true, leakage: undefined,
            result: { experiment_id: 82, evaluated_at: 'x', rows: 50, headline: 'f1', score: 0.48,
                judge: { skipped: 'local_only_cloud_judge', judge: 'anthropic:claude-x' } },
        };
        apiMock.get.mockImplementation((url: string) => Promise.resolve({ data: url.endsWith('/history') ? { history: [
            { experiment_id: 80, experiment_name: 'r80', evaluated_at: '2026-10-01T00:00:00', headline: 'f1', score: 0.31, rows: 50 },
            { experiment_id: 81, experiment_name: 'r81', evaluated_at: '2026-10-05T00:00:00', headline: 'f1', score: 0.40, rows: 50 },
            { experiment_id: 82, experiment_name: 'r82', evaluated_at: '2026-10-09T00:00:00', headline: 'f1', score: 0.48, rows: 50 },
        ] } : { benchmarks: [qa] } }));
        render(<ExternalBenchmarksPanel projectId={33} experimentId={82} />);
        const trend = await screen.findByTestId('ext-bench-trend-squad');
        expect(trend).toHaveTextContent('F1 0.31 (run #80) → 0.48 (run #82) · 3 runs');
        expect(trend.querySelectorAll('circle')).toHaveLength(3);
        expect(trend.querySelector('title')?.textContent).toBe('Run #80 (r80) · 2026-10-01 · F1 0.31');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/33/benchmarks/7/history');
        expect(screen.getByTestId('ext-bench-squad')).toHaveTextContent('Q&A — scores the answers');
        expect(screen.getByTestId('ext-bench-score-squad')).toHaveTextContent(
            'Run #82: F1 0.48 on 50 rows · not judged: local-only benchmark, cloud judge (anthropic:claude-x)');
    });
});
