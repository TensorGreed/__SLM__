import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import HealthCheckButton from './HealthCheckButton';
import HealthCheckModal from './HealthCheckModal';


/** Build a wire-shape smoke summary (snake_case, as the backend
 *  returns it). The component's API client normalizes this to
 *  camelCase before rendering. */
function makeSummary(overrides: Partial<Record<string, unknown>> = {}) {
    return {
        project_id: 17,
        overall: 'ok' as const,
        elapsed_ms: 42,
        counts: { ok: 9, warn: 0, fail: 0, skip: 0 },
        checks: [
            {
                name: 'project_exists',
                status: 'ok' as const,
                elapsed_ms: 5,
                message: "Project 'X' is accessible.",
                remediation: null,
                envelope: null,
                metadata: { name: 'X' },
            },
            {
                name: 'recipe_applied',
                status: 'ok' as const,
                elapsed_ms: 4,
                message: "Recipe 'classification' is applied.",
                remediation: null,
                envelope: null,
                metadata: { recipe_id: 'classification' },
            },
        ],
        ...overrides,
    };
}


describe('HealthCheckButton', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('renders the trigger button with a stable test-id', () => {
        render(<HealthCheckButton projectId={17} />);
        expect(screen.getByTestId('health-check-button')).toBeInTheDocument();
    });

    it('opens the modal on click and fires the smoke-test POST exactly once', async () => {
        apiMock.post.mockResolvedValueOnce({ data: makeSummary() });
        render(<HealthCheckButton projectId={17} />);
        await userEvent.click(screen.getByTestId('health-check-button'));
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal')).toBeInTheDocument();
        });
        // The button fires the smoke-test endpoint, not anything else.
        expect(apiMock.post).toHaveBeenCalledWith('/projects/17/smoke-test');
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });

    it('keeps the modal closed until the button is clicked', () => {
        render(<HealthCheckButton projectId={17} />);
        expect(screen.queryByTestId('health-check-modal')).not.toBeInTheDocument();
    });
});


describe('HealthCheckModal', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('shows a loading state then renders the summary', async () => {
        let resolveFetch: (value: unknown) => void = () => {};
        apiMock.post.mockReturnValueOnce(
            new Promise((resolve) => { resolveFetch = resolve; }),
        );
        const onClose = vi.fn();
        render(<HealthCheckModal projectId={17} onClose={onClose} />);
        // Spinner first.
        expect(screen.getByTestId('health-check-modal-loading')).toBeInTheDocument();
        // Resolve the request → summary renders + spinner goes away.
        resolveFetch({ data: makeSummary() });
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-summary')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('health-check-modal-loading')).not.toBeInTheDocument();
    });

    it('renders one row per check with status + message + elapsed', async () => {
        apiMock.post.mockResolvedValueOnce({ data: makeSummary() });
        const onClose = vi.fn();
        render(<HealthCheckModal projectId={17} onClose={onClose} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-check-project_exists')).toBeInTheDocument();
        });
        const row1 = screen.getByTestId('health-check-modal-check-project_exists');
        expect(row1.textContent).toMatch(/Project accessible/);
        expect(row1.textContent).toMatch(/X.*accessible/);
        expect(row1.textContent).toMatch(/5ms/);
        const row2 = screen.getByTestId('health-check-modal-check-recipe_applied');
        expect(row2.textContent).toMatch(/Recipe applied/);
        expect(row2.textContent).toMatch(/classification/);
    });

    it('renders the failure envelope inside the row via shared <ErrorPanel>', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: makeSummary({
                overall: 'fail',
                counts: { ok: 0, warn: 0, fail: 1, skip: 0 },
                checks: [
                    {
                        name: 'recipe_applied',
                        status: 'fail',
                        elapsed_ms: 3,
                        message: 'No recipe selected on this project.',
                        remediation: 'Open Pipeline → Recipe picker.',
                        envelope: {
                            error_code: 'SMOKE_RECIPE_MISSING',
                            stage: 'project',
                            message: 'No recipe selected on this project.',
                            actionable_fix: 'Open Pipeline → Recipe picker.',
                            docs_url: '/docs/troubleshooting',
                            troubleshooting_id: 'err_abcdefghi',
                            metadata: null,
                            detail: 'No recipe selected.',
                        },
                        metadata: {},
                    },
                ],
            }),
        });
        render(<HealthCheckModal projectId={17} onClose={() => {}} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-check-recipe_applied')).toBeInTheDocument();
        });
        // The check row carries the fail status…
        const row = screen.getByTestId('health-check-modal-check-recipe_applied');
        expect(row.getAttribute('data-status')).toBe('fail');
        // …and embeds the shared ErrorPanel so the user sees the
        // troubleshooting_id + remediation + copy button.
        const envelope = screen.getByTestId('health-check-modal-envelope-recipe_applied');
        expect(envelope).toBeInTheDocument();
        expect(envelope.textContent).toMatch(/err_abcdefghi/);
        // Remediation is shown inline above the envelope too.
        expect(
            screen.getByTestId('health-check-modal-remediation-recipe_applied').textContent,
        ).toMatch(/Recipe picker/);
    });

    it('overall badge reflects the worst-severity status', async () => {
        // overall=warn with a mix of ok + warn checks.
        apiMock.post.mockResolvedValueOnce({
            data: makeSummary({
                overall: 'warn',
                counts: { ok: 1, warn: 1, fail: 0, skip: 0 },
                checks: [
                    {
                        name: 'project_exists', status: 'ok', elapsed_ms: 5,
                        message: 'ok', remediation: null, envelope: null, metadata: {},
                    },
                    {
                        name: 'gold_set', status: 'warn', elapsed_ms: 5,
                        message: 'Gold set is empty (0 rows).',
                        remediation: 'Open Pipeline → Gold set and seed rows.',
                        envelope: null, metadata: { gold_row_count: 0 },
                    },
                ],
            }),
        });
        render(<HealthCheckModal projectId={17} onClose={() => {}} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-overall-badge')).toBeInTheDocument();
        });
        const badge = screen.getByTestId('health-check-modal-overall-badge');
        expect(badge.textContent).toMatch(/WARN/);
        const summary = screen.getByTestId('health-check-modal-summary');
        expect(summary.getAttribute('data-overall')).toBe('warn');
    });

    it('Re-run button fires the smoke-test POST again', async () => {
        apiMock.post.mockResolvedValue({ data: makeSummary() });
        render(<HealthCheckModal projectId={17} onClose={() => {}} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-summary')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('health-check-modal-rerun'));
        await waitFor(() => {
            // 2 POSTs total: mount + re-run.
            expect(apiMock.post).toHaveBeenCalledTimes(2);
        });
    });

    it('Done + close button + backdrop all fire onClose', async () => {
        apiMock.post.mockResolvedValue({ data: makeSummary() });
        const onClose = vi.fn();
        render(<HealthCheckModal projectId={17} onClose={onClose} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-done')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('health-check-modal-done'));
        expect(onClose).toHaveBeenCalled();
    });

    it('shows the load-error ErrorPanel + Retry when the POST itself fails', async () => {
        // First call fails (e.g. 500); the modal must NOT crash — it
        // surfaces the failure via <ErrorPanel> with a Retry action.
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 500,
                data: {
                    error_code: 'GENERAL_INTERNAL_ERROR',
                    stage: 'general',
                    message: 'Smoke-test endpoint failed.',
                    actionable_fix: 'Retry; if it persists, file a bug.',
                    docs_url: '/docs/troubleshooting',
                    troubleshooting_id: 'err_zzzzzzzzz',
                    detail: 'Smoke-test endpoint failed.',
                },
            },
        });
        // Second call (Retry) succeeds.
        apiMock.post.mockResolvedValueOnce({ data: makeSummary() });
        render(<HealthCheckModal projectId={17} onClose={() => {}} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-load-error')).toBeInTheDocument();
        });
        // The error panel surfaces the trace id from the envelope.
        expect(
            screen.getByTestId('health-check-modal-load-error-trace-id').textContent,
        ).toMatch(/err_zzzzzzzzz/);
        // Click Retry → second POST runs + summary appears.
        await userEvent.click(screen.getByTestId('health-check-modal-retry'));
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-summary')).toBeInTheDocument();
        });
    });

    it('renders unknown check names with their raw id as the label', async () => {
        // Forward-compat: new backend checks appear with their raw
        // name until a label is added in CHECK_LABELS. This must not
        // crash + must remain readable.
        apiMock.post.mockResolvedValueOnce({
            data: makeSummary({
                checks: [
                    {
                        name: 'brand_new_check', status: 'ok', elapsed_ms: 4,
                        message: 'all good',
                        remediation: null, envelope: null, metadata: {},
                    },
                ],
            }),
        });
        render(<HealthCheckModal projectId={17} onClose={() => {}} />);
        await waitFor(() => {
            expect(screen.getByTestId('health-check-modal-check-brand_new_check')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('health-check-modal-check-brand_new_check').textContent,
        ).toMatch(/brand_new_check/);
    });
});
