import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import GoldSetDiagnosticsPanel from './GoldSetDiagnosticsPanel';

const SUFFICIENT = {
    project_id: 7,
    total_rows: 30,
    classification_eligible: true,
    class_balance: {
        labels: [
            { label: 'spam', count: 18, share: 0.6 },
            { label: 'ham', count: 9, share: 0.3 },
            { label: 'promo', count: 3, share: 0.1 },  // below 15% floor
        ],
        total: 30,
        entropy_nats: 0.89,
    },
    similarity: {
        labels: ['spam', 'ham', 'promo'],
        matrix: [
            [0.62, 0.14, 0.10],
            [0.14, 0.45, 0.12],
            [0.10, 0.12, null],  // promo has too few rows for intra-class
        ],
        sample_per_class: 12,
        insufficient_labels: ['promo'],
    },
};

describe('GoldSetDiagnosticsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('fetches diagnostics on mount and renders class-balance + matrix when eligible', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SUFFICIENT });
        render(<GoldSetDiagnosticsPanel projectId={7} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/projects/7/gold/diagnostics');
        });
        // Header reads "30 rows · 3 classes · entropy 0.89".
        await waitFor(() => {
            const node = screen.getByTestId('gold-diag');
            expect(node.textContent).toMatch(/30 rows/);
            expect(node.textContent).toMatch(/3 classes/);
            expect(node.textContent).toMatch(/0\.89/);
        });
        // All three class-balance rows render.
        expect(screen.getByTestId('gold-diag-balance-spam')).toBeInTheDocument();
        expect(screen.getByTestId('gold-diag-balance-ham')).toBeInTheDocument();
        expect(screen.getByTestId('gold-diag-balance-promo')).toBeInTheDocument();
        // Below-floor class gets the is-below-floor class so the bar
        // colours red.
        expect(
            screen.getByTestId('gold-diag-balance-promo').className,
        ).toMatch(/is-below-floor/);
        // Above-floor classes don't get the modifier.
        expect(
            screen.getByTestId('gold-diag-balance-spam').className,
        ).not.toMatch(/is-below-floor/);
    });

    it('renders the similarity matrix cells with values and n/a where insufficient', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SUFFICIENT });
        render(<GoldSetDiagnosticsPanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('gold-diag-matrix')).toBeInTheDocument();
        });
        // Diagonal cells render their values.
        expect(screen.getByTestId('gold-diag-cell-spam-spam').textContent).toMatch(/0\.62/);
        expect(screen.getByTestId('gold-diag-cell-ham-ham').textContent).toMatch(/0\.45/);
        // Promo's intra-class cell is null → 'n/a'.
        expect(screen.getByTestId('gold-diag-cell-promo-promo').textContent).toMatch(/n\/a/);
        // Insufficient classes are named at the bottom.
        expect(screen.getByTestId('gold-diag-insufficient').textContent).toMatch(/promo/);
    });

    it('renders the empty-state hint when there are no classification labels', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                total_rows: 12,
                classification_eligible: false,
                class_balance: { labels: [], total: 12, entropy_nats: 0.0 },
                similarity: { labels: [], matrix: [], sample_per_class: 12, insufficient_labels: [] },
            },
        });
        render(<GoldSetDiagnosticsPanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('gold-diag-empty').textContent).toMatch(/classification/);
        });
        // No matrix/balance scaffolding rendered.
        expect(screen.queryByTestId('gold-diag-matrix')).not.toBeInTheDocument();
        expect(screen.queryByTestId('gold-diag-balance')).not.toBeInTheDocument();
    });

    it('renders the empty-state hint for total_rows=0 (no gold yet)', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                total_rows: 0,
                classification_eligible: false,
                class_balance: { labels: [], total: 0, entropy_nats: 0.0 },
                similarity: { labels: [], matrix: [], sample_per_class: 12, insufficient_labels: [] },
            },
        });
        render(<GoldSetDiagnosticsPanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('gold-diag-empty').textContent).toMatch(/No gold rows yet/);
        });
    });

    it('omits the matrix block when there is only one class', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                total_rows: 8,
                classification_eligible: true,
                class_balance: {
                    labels: [{ label: 'only', count: 8, share: 1.0 }],
                    total: 8,
                    entropy_nats: 0.0,
                },
                similarity: {
                    labels: [], matrix: [], sample_per_class: 12, insufficient_labels: [],
                },
            },
        });
        render(<GoldSetDiagnosticsPanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('gold-diag-balance-only')).toBeInTheDocument();
        });
        // No matrix — need >=2 classes for a similarity heatmap.
        expect(screen.queryByTestId('gold-diag-matrix')).not.toBeInTheDocument();
    });
});
