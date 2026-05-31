import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import TokenLengthDistributionPanel from './TokenLengthDistributionPanel';

const histogram = (counts: number[]) => [
    { bucket: '0-256', count: counts[0] || 0 },
    { bucket: '256-512', count: counts[1] || 0 },
    { bucket: '512-1024', count: counts[2] || 0 },
    { bucket: '1024-2048', count: counts[3] || 0 },
    { bucket: '2048+', count: counts[4] || 0 },
];

function splitStats(overrides: Partial<Record<string, number | unknown>> = {}) {
    return {
        total_samples: 100,
        p50_tokens: 240,
        p95_tokens: 480,
        p99_tokens: 700,
        max_tokens: 850,
        exceeding_max: 0,
        max_seq_length: 2048,
        histogram: histogram([60, 30, 10, 0, 0]),
        ...overrides,
    };
}

describe('TokenLengthDistributionPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('does not call the API until the user clicks Analyze splits', () => {
        render(<TokenLengthDistributionPanel projectId={5} />);
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('renders three rows in the stats table when all splits are present', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                model_name: 'foo/bar',
                max_seq_length: 2048,
                splits: {
                    train: splitStats({ total_samples: 800 }),
                    validation: splitStats({ total_samples: 100 }),
                    test: splitStats({ total_samples: 100 }),
                },
                missing_splits: [],
                errors: {},
            },
        });

        const user = userEvent.setup();
        render(<TokenLengthDistributionPanel projectId={5} />);
        await user.click(screen.getByTestId('token-dist-analyze'));

        await waitFor(() => {
            expect(screen.getByTestId('token-dist-row-train')).toBeInTheDocument();
            expect(screen.getByTestId('token-dist-row-validation')).toBeInTheDocument();
            expect(screen.getByTestId('token-dist-row-test')).toBeInTheDocument();
        });

        // POST shape — single call to the new endpoint with the
        // model_name + max_seq_length the launcher had configured.
        const call = apiMock.post.mock.calls[0];
        expect(call[0]).toBe('/projects/5/tokenization/analyze-splits');
        expect(call[1]).toEqual(expect.objectContaining({
            model_name: expect.any(String),
            max_seq_length: 2048,
        }));
    });

    it('surfaces a "splits not yet prepared" chip when test is missing', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                model_name: 'foo/bar',
                max_seq_length: 2048,
                splits: {
                    train: splitStats({ total_samples: 800 }),
                    validation: splitStats({ total_samples: 100 }),
                },
                missing_splits: ['test'],
                errors: {},
            },
        });
        const user = userEvent.setup();
        render(<TokenLengthDistributionPanel projectId={5} />);
        await user.click(screen.getByTestId('token-dist-analyze'));

        await waitFor(() => {
            expect(screen.getByTestId('token-dist-missing').textContent).toMatch(/test/);
        });
        // No row rendered for the missing split.
        expect(screen.queryByTestId('token-dist-row-test')).not.toBeInTheDocument();
    });

    it('renders the distribution-shift note when test p95 is much larger than train p95', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                model_name: 'foo/bar',
                max_seq_length: 2048,
                splits: {
                    train: splitStats({ p95_tokens: 300, total_samples: 800 }),
                    test: splitStats({ p95_tokens: 600, total_samples: 100 }),
                },
                missing_splits: ['validation'],
                errors: {},
            },
        });
        const user = userEvent.setup();
        render(<TokenLengthDistributionPanel projectId={5} />);
        await user.click(screen.getByTestId('token-dist-analyze'));

        await waitFor(() => {
            const note = screen.getByTestId('token-dist-shift');
            expect(note.textContent).toMatch(/test p95 = 600/);
            expect(note.textContent).toMatch(/train p95 = 300/);
            expect(note.textContent).toMatch(/silently truncate/);
        });
    });

    it('stays silent on the shift note when train/test p95s are within the noise threshold', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                model_name: 'foo/bar',
                max_seq_length: 2048,
                splits: {
                    // 1.20× = below the 1.30 threshold → no warning.
                    train: splitStats({ p95_tokens: 500, total_samples: 800 }),
                    test: splitStats({ p95_tokens: 600, total_samples: 100 }),
                },
                missing_splits: ['validation'],
                errors: {},
            },
        });
        const user = userEvent.setup();
        render(<TokenLengthDistributionPanel projectId={5} />);
        await user.click(screen.getByTestId('token-dist-analyze'));

        await waitFor(() => {
            expect(screen.getByTestId('token-dist-row-train')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('token-dist-shift')).not.toBeInTheDocument();
    });

    it('renders an error banner when the API call fails', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'No prepared splits found.' } },
        });
        const user = userEvent.setup();
        render(<TokenLengthDistributionPanel projectId={5} />);
        await user.click(screen.getByTestId('token-dist-analyze'));

        await waitFor(() => {
            expect(screen.getByTestId('token-dist-error').textContent).toMatch(/No prepared splits/);
        });
    });
});
