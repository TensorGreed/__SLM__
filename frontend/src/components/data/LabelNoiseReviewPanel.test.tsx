import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import LabelNoiseReviewPanel from './LabelNoiseReviewPanel';

const SCAN_PAYLOAD = {
    id: 9,
    project_id: 1,
    base_experiment_id: 42,
    status: 'succeeded' as const,
    label_count_at_scan: 387,
    suspected_count: 4,
    confidence_threshold: 0.85,
    given_label_floor: 0.15,
    result_payload: {
        scored_at: '2026-06-09T12:00:00Z',
        base_experiment_id: 42,
        label_count_total: 387,
        label_count_scored: 387,
        suspected_count: 4,
        confidence_threshold: 0.85,
        given_label_floor: 0.15,
        skipped_reason: null,
        top_k: [
            { label_row_id: 100, label_job_id: 7, given_label: 'A', predicted_label: 'B', predicted_prob: 0.92, given_label_prob: 0.06, mislabel_score: 0.86, text_preview: 'A→B row one' },
            { label_row_id: 101, label_job_id: 7, given_label: 'A', predicted_label: 'B', predicted_prob: 0.90, given_label_prob: 0.08, mislabel_score: 0.82, text_preview: 'A→B row two' },
            { label_row_id: 200, label_job_id: 7, given_label: 'B', predicted_label: 'A', predicted_prob: 0.88, given_label_prob: 0.10, mislabel_score: 0.78, text_preview: 'B→A row' },
            { label_row_id: 300, label_job_id: 7, given_label: 'C', predicted_label: 'A', predicted_prob: 0.86, given_label_prob: 0.12, mislabel_score: 0.74, text_preview: null },
        ],
    },
    error: null,
    job_id: 11,
    created_at: '2026-06-09T11:00:00Z',
    completed_at: '2026-06-09T12:00:00Z',
};

describe('LabelNoiseReviewPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders the empty state when no scan has run yet', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 1, scan: null, no_scan_reason: 'no_succeeded_scan_yet' },
        });

        render(<LabelNoiseReviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText(/No label-noise scan has run yet/i)).toBeInTheDocument();
        });
    });

    it('renders the clean state when scan succeeded with zero suspects', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                scan: {
                    ...SCAN_PAYLOAD,
                    suspected_count: 0,
                    result_payload: { ...SCAN_PAYLOAD.result_payload, suspected_count: 0, top_k: [] },
                },
                no_scan_reason: null,
            },
        });

        render(<LabelNoiseReviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText(/your labels look clean/i)).toBeInTheDocument();
        });
    });

    it('groups suspects by (given → predicted) transition with bulk affordance', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 1, scan: SCAN_PAYLOAD, no_scan_reason: null },
        });

        render(<LabelNoiseReviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('label-noise-review')).toBeInTheDocument();
        });
        // Three distinct groups: A→B (2 rows), B→A (1), C→A (1).
        expect(screen.getByText(/\(2 rows\)/)).toBeInTheDocument();
        expect(screen.getAllByText(/\(1 row\)/).length).toBe(2);
        // Provenance meta line.
        expect(screen.getByText(/scan #9/)).toBeInTheDocument();
        expect(screen.getByText(/exp #42/)).toBeInTheDocument();
        // Per-row preview text.
        expect(screen.getByText('A→B row one')).toBeInTheDocument();
        expect(screen.getByText('A→B row two')).toBeInTheDocument();
        // Missing-text row falls back to "(no text)".
        expect(screen.getByText(/no text/i)).toBeInTheDocument();
    });

    it('selects all rows in a group with the "Relabel all" affordance and applies', async () => {
        const user = userEvent.setup();
        apiMock.get
            .mockResolvedValueOnce({  // initial load
                data: { project_id: 1, scan: SCAN_PAYLOAD, no_scan_reason: null },
            })
            .mockResolvedValueOnce({  // reload after apply
                data: { project_id: 1, scan: SCAN_PAYLOAD, no_scan_reason: null },
            });
        apiMock.post.mockResolvedValueOnce({
            data: {
                scan_id: 9,
                project_id: 1,
                applied: 2,
                relabeled: 2,
                kept: 0,
                dropped: 0,
                skipped: [],
                applied_actions: {
                    '100': { label_row_id: 100, action: 'relabel', applied_label: 'B', applied_at: '2026-06-09T13:00:00Z' },
                    '101': { label_row_id: 101, action: 'relabel', applied_label: 'B', applied_at: '2026-06-09T13:00:00Z' },
                },
            },
        });

        render(<LabelNoiseReviewPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('label-noise-review')).toBeInTheDocument();
        });

        // The A→B group has the most rows (2), so it's rendered first.
        // Click "Relabel all" on that group — pending count jumps to 2.
        const relabelAllButtons = screen.getAllByRole('button', { name: /Relabel all 2 rows as B/i });
        await user.click(relabelAllButtons[0]);
        expect(screen.getByText(/2 decisions pending/)).toBeInTheDocument();

        // Apply.
        const applyButton = screen.getByRole('button', { name: /Apply decisions/i });
        await user.click(applyButton);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/label-noise/scans/9/apply',
                {
                    actions: [
                        { label_row_id: 100, action: 'relabel' },
                        { label_row_id: 101, action: 'relabel' },
                    ],
                },
            );
        });
        // Apply-result line shows the summary. The counts are wrapped
        // in <strong> elements, so getByText with a regex spanning
        // the static label suffix is the cleanest match.
        await waitFor(() => {
            expect(screen.getByText(/Applied:/)).toBeInTheDocument();
        });
        expect(screen.getByText(/relabeled/)).toBeInTheDocument();
    });

    it('fetches a specific scan when scanId is provided', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SCAN_PAYLOAD });

        render(<LabelNoiseReviewPanel projectId={1} scanId={9} />);

        await waitFor(() => {
            expect(screen.getByTestId('label-noise-review')).toBeInTheDocument();
        });
        // Explicit scan endpoint (not /latest) — different URL pattern.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/label-noise/scans/9');
    });

    it('renders per-row buttons that toggle decisions', async () => {
        const user = userEvent.setup();
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 1, scan: SCAN_PAYLOAD, no_scan_reason: null },
        });

        render(<LabelNoiseReviewPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('label-noise-review')).toBeInTheDocument();
        });

        // Find row #100's Drop button. Each row has Relabel / Keep / Drop.
        // We scope the search to the row's <li> by walking from the row id.
        const rowIdCell = screen.getByText('#100');
        const rowLi = rowIdCell.closest('li')!;
        const dropBtn = within(rowLi).getByRole('button', { name: /Drop/i });
        await user.click(dropBtn);

        // Pending count surfaces.
        expect(screen.getByText(/1 decision pending/)).toBeInTheDocument();
        expect(dropBtn).toHaveAttribute('aria-pressed', 'true');

        // Click the same button again to toggle off.
        await user.click(dropBtn);
        expect(
            screen.getByText(/Select decisions per row or per group/),
        ).toBeInTheDocument();
    });
});
