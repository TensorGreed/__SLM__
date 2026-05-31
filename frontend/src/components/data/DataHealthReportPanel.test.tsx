import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

const { apiMock, navigateMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
    navigateMock: vi.fn(),
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return { ...actual, useNavigate: () => navigateMock };
});

import DataHealthReportPanel from './DataHealthReportPanel';

function renderPanel() {
    return render(
        <MemoryRouter>
            <DataHealthReportPanel projectId={42} />
        </MemoryRouter>,
    );
}

const BLOCK_REPORT = {
    project_id: 42,
    computed_at: '2026-05-31T22:00:00Z',
    overall: 'block' as const,
    severity_summary: { ok: 0, warn: 0, block: 2 },
    total_signals: 2,
    groups: [
        {
            id: 'ingestion',
            title: 'Ingestion',
            subtitle: 'Documents uploaded + parsed',
            signals: [
                {
                    id: 'ingestion.no_documents',
                    severity: 'block' as const,
                    headline: 'No documents uploaded yet.',
                    plain_english: "You haven't uploaded any source documents yet.",
                    why_it_matters: 'Training needs source text to learn from — the platform has nothing to work with.',
                    suggested_action: { kind: 'navigate', label: 'Open Ingest tab', target: 'data' },
                    context: { document_count: 0 },
                },
            ],
        },
        {
            id: 'shape',
            title: 'Data shape vs recipe',
            subtitle: 'Does the data fit the recipe?',
            signals: [
                {
                    id: 'shape.no_recipe_selected',
                    severity: 'block' as const,
                    headline: 'No recipe selected for this project.',
                    plain_english: "You haven't picked a recipe yet (classification, span-extraction, summarization, qa-sft, etc.).",
                    why_it_matters: "Without it, the platform can't tell you whether your data will work.",
                    suggested_action: { kind: 'navigate', label: 'Open recipe picker', target: 'recipe-picker' },
                    context: {},
                },
            ],
        },
    ],
};

const MIXED_REPORT = {
    project_id: 42,
    computed_at: '2026-05-31T22:00:00Z',
    overall: 'warn' as const,
    severity_summary: { ok: 1, warn: 1, block: 0 },
    total_signals: 2,
    groups: [
        {
            id: 'ingestion',
            title: 'Ingestion',
            subtitle: 'Documents uploaded + parsed',
            signals: [
                {
                    id: 'ingestion.parse_failure_rate',
                    severity: 'ok' as const,
                    headline: 'All 20 documents parsed cleanly.',
                    plain_english: '',
                    why_it_matters: '',
                    suggested_action: null,
                    context: { document_count: 20, errored: 0 },
                },
            ],
        },
        {
            id: 'cleaning',
            title: 'Cleaning',
            subtitle: 'PII redaction + quality + dedup',
            signals: [
                {
                    id: 'cleaning.pii_unredacted',
                    severity: 'warn' as const,
                    headline: '18 PII findings detected across 3 cleaned doc(s) — redaction was not applied.',
                    plain_english: 'The cleaning step found personal information but did not redact it.',
                    why_it_matters: 'An SLM trained on unredacted PII can memorise and emit it at inference time.',
                    suggested_action: { kind: 'navigate', label: 'Re-clean with redact-PII on', target: 'cleaning' },
                    context: { pii_findings: 18 },
                },
            ],
        },
    ],
};

describe('DataHealthReportPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        navigateMock.mockReset();
    });

    it('fetches the report on mount and renders all populated groups', async () => {
        apiMock.get.mockResolvedValueOnce({ data: BLOCK_REPORT });
        renderPanel();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/projects/42/data-health');
        });
        await waitFor(() => {
            expect(screen.getByTestId('data-health-group-ingestion')).toBeInTheDocument();
            expect(screen.getByTestId('data-health-group-shape')).toBeInTheDocument();
        });
        // The empty cleaning + balance groups are omitted from the render.
        expect(screen.queryByTestId('data-health-group-cleaning')).not.toBeInTheDocument();
        expect(screen.queryByTestId('data-health-group-balance')).not.toBeInTheDocument();
    });

    it('renders the top-line overall badge matching the worst severity', async () => {
        apiMock.get.mockResolvedValueOnce({ data: BLOCK_REPORT });
        renderPanel();
        await waitFor(() => {
            const badge = screen.getByTestId('data-health-overall-badge');
            expect(badge.textContent).toMatch(/Blocker/i);
            expect(badge.className).toMatch(/--block/);
        });
        // The headline copy is the block message.
        expect(screen.getByTestId('data-health').textContent).toMatch(/Blockers/);
    });

    it('shows the plain-English summary before the technical headline on each signal', async () => {
        apiMock.get.mockResolvedValueOnce({ data: BLOCK_REPORT });
        renderPanel();
        await waitFor(() => {
            const node = screen.getByTestId('data-health-signal-ingestion.no_documents');
            const text = node.textContent || '';
            // Plain-English copy appears in the rendered DOM.
            expect(text).toMatch(/haven't uploaded any source documents/);
            // Technical headline also appears.
            expect(text).toMatch(/No documents uploaded yet/);
        });
    });

    it('toggles the "Why this matters" expander on click', async () => {
        apiMock.get.mockResolvedValueOnce({ data: BLOCK_REPORT });
        renderPanel();
        const user = userEvent.setup();
        await waitFor(() => {
            expect(screen.getByTestId('data-health-why-ingestion.no_documents')).toBeInTheDocument();
        });
        // Initially collapsed.
        expect(
            screen.queryByTestId('data-health-why-text-ingestion.no_documents'),
        ).not.toBeInTheDocument();
        await user.click(screen.getByTestId('data-health-why-ingestion.no_documents'));
        // Expanded — body visible.
        await waitFor(() => {
            const body = screen.getByTestId('data-health-why-text-ingestion.no_documents');
            expect(body.textContent).toMatch(/Training needs source text/);
        });
        // Click again — collapses.
        await user.click(screen.getByTestId('data-health-why-ingestion.no_documents'));
        await waitFor(() => {
            expect(
                screen.queryByTestId('data-health-why-text-ingestion.no_documents'),
            ).not.toBeInTheDocument();
        });
    });

    it('navigates to the action target when the action button is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: BLOCK_REPORT });
        renderPanel();
        const user = userEvent.setup();
        await waitFor(() => {
            expect(screen.getByTestId('data-health-action-shape.no_recipe_selected')).toBeInTheDocument();
        });
        await user.click(screen.getByTestId('data-health-action-shape.no_recipe_selected'));
        expect(navigateMock).toHaveBeenCalledWith('/project/42/recipe-picker');
    });

    it('renders the ok severity badge inline for ok signals', async () => {
        apiMock.get.mockResolvedValueOnce({ data: MIXED_REPORT });
        renderPanel();
        await waitFor(() => {
            const okSignal = screen.getByTestId('data-health-signal-ingestion.parse_failure_rate');
            expect(okSignal.getAttribute('data-severity')).toBe('ok');
        });
        // Mixed report has 1 warn + 1 ok → overall=warn (overall is set
        // by the backend; the panel just displays it).
        expect(
            screen.getByTestId('data-health-overall-badge').className,
        ).toMatch(/--warn/);
    });

    it('shows an autofix button on signals with autofix_kind and applies it on confirm', async () => {
        const reportWithAutofix = {
            ...BLOCK_REPORT,
            groups: [
                {
                    id: 'cleaning',
                    title: 'Cleaning',
                    subtitle: 'PII redaction + quality + dedup',
                    signals: [
                        {
                            id: 'cleaning.duplicate_chunks',
                            severity: 'warn' as const,
                            headline: '5 duplicate document(s) detected.',
                            plain_english: 'A significant share of your cleaned text chunks are duplicates.',
                            why_it_matters: 'The model will overfit on the duplicated patterns.',
                            suggested_action: { kind: 'navigate', label: 'Review duplicates', target: 'cleaning' },
                            context: { duplicate_count: 5 },
                            autofix_kind: 'dedupe_duplicate_docs',
                        },
                    ],
                },
            ],
            overall: 'warn' as const,
            severity_summary: { ok: 0, warn: 1, block: 0 },
            total_signals: 1,
        };
        // After the autofix runs, the report shows the dup signal as ok.
        const reportAfterFix = {
            ...reportWithAutofix,
            groups: [
                {
                    ...reportWithAutofix.groups[0],
                    signals: [
                        {
                            ...reportWithAutofix.groups[0].signals[0],
                            severity: 'ok' as const,
                            headline: 'No duplicate documents detected.',
                            autofix_kind: null,
                        },
                    ],
                },
            ],
            overall: 'ok' as const,
            severity_summary: { ok: 1, warn: 0, block: 0 },
        };
        apiMock.get
            .mockResolvedValueOnce({ data: reportWithAutofix })
            .mockResolvedValueOnce({ data: reportAfterFix });
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'dedupe_duplicate_docs',
                applied_count: 5,
                summary: 'Dropped 5 duplicate documents across 2 dedup groups.',
                details: { group_count: 2 },
            },
        });
        // Stub the confirm dialog to auto-confirm.
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

        renderPanel();
        const user = userEvent.setup();

        // Autofix button is visible.
        await waitFor(() => {
            expect(
                screen.getByTestId('data-health-autofix-cleaning.duplicate_chunks'),
            ).toBeInTheDocument();
        });

        await user.click(screen.getByTestId('data-health-autofix-cleaning.duplicate_chunks'));

        // Confirm prompt fired.
        expect(confirmSpy).toHaveBeenCalled();

        // POST went to the right endpoint with the right kind.
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/data-health/autofix',
                { fix_kind: 'dedupe_duplicate_docs' },
            );
        });

        // Toast surfaces the fix summary.
        await waitFor(() => {
            const toast = screen.getByTestId('data-health-fix-toast');
            expect(toast.textContent).toMatch(/Dropped 5 duplicate documents/);
        });

        // Report refetched (second GET call).
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledTimes(2);
        });

        confirmSpy.mockRestore();
    });

    it('cancelling the autofix confirm does not POST', async () => {
        const reportWithAutofix = {
            ...BLOCK_REPORT,
            groups: [
                {
                    id: 'cleaning',
                    title: 'Cleaning',
                    subtitle: 'PII redaction + quality + dedup',
                    signals: [
                        {
                            id: 'cleaning.pii_unredacted',
                            severity: 'warn' as const,
                            headline: '18 PII findings detected.',
                            plain_english: 'PII detected, not redacted.',
                            why_it_matters: 'Memorisation risk.',
                            suggested_action: null,
                            context: {},
                            autofix_kind: 'redact_pii',
                        },
                    ],
                },
            ],
        };
        apiMock.get.mockResolvedValueOnce({ data: reportWithAutofix });
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

        renderPanel();
        const user = userEvent.setup();
        await waitFor(() => {
            expect(
                screen.getByTestId('data-health-autofix-cleaning.pii_unredacted'),
            ).toBeInTheDocument();
        });
        await user.click(screen.getByTestId('data-health-autofix-cleaning.pii_unredacted'));

        // POST never called when the user cancels the confirm.
        expect(apiMock.post).not.toHaveBeenCalled();
        confirmSpy.mockRestore();
    });

    it('renders an error fallback when the API call fails', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'Project not found' } },
        });
        renderPanel();
        await waitFor(() => {
            const node = screen.getByTestId('data-health');
            expect(node.className).toMatch(/--error/);
            expect(node.textContent).toMatch(/Project not found/);
        });
    });
});
