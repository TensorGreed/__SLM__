/**
 * Document-sample accordion contract.
 *
 * Pins:
 * - On mount, fetches the sample endpoint once.
 * - Renders rows as JSON pretty-print + the scanned-from-total label.
 * - "Refresh sample" re-fetches on demand.
 * - Surfaces the server's "note" (e.g. unsupported file type) when
 *   rows are empty.
 * - Surfaces errors inline rather than crashing the table.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import DocumentSampleAccordion from './DocumentSampleAccordion';

function renderInsideTable(children: React.ReactNode) {
    // The component renders a <tr>; mount it inside a real <table> so
    // colSpan etc. parse correctly under jsdom.
    return render(<table><tbody>{children}</tbody></table>);
}

describe('DocumentSampleAccordion', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('fetches the sample on mount and renders rows as JSON', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                document_id: 42,
                filename: 'train.jsonl',
                rows: [
                    { text: 'hello world', label: 'pos' },
                    { text: 'goodbye', label: 'neg' },
                ],
                total_rows_scanned: 1234,
                source: 'raw',
                file_type: 'jsonl',
                note: '',
            },
        });

        renderInsideTable(
            <DocumentSampleAccordion projectId={77} documentId={42} colSpan={8} />,
        );

        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/77/ingestion/documents/42/sample',
            ),
        );
        expect(await screen.findByText(/hello world/)).toBeInTheDocument();
        expect(screen.getByText(/from 1,234 total/i)).toBeInTheDocument();
    });

    it('shows the server "note" when rows are empty (unsupported file)', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                document_id: 3,
                filename: 'doc.pdf',
                rows: [],
                total_rows_scanned: 0,
                source: 'raw',
                file_type: 'pdf',
                note: "Preview not available for '.pdf' files.",
            },
        });

        renderInsideTable(
            <DocumentSampleAccordion projectId={77} documentId={3} colSpan={8} />,
        );

        const note = await screen.findByTestId('sample-note-3');
        expect(note.textContent).toMatch(/Preview not available/);
    });

    it('refreshes the sample on demand', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                document_id: 5,
                filename: 'rows.jsonl',
                rows: [{ x: 1 }],
                total_rows_scanned: 10,
                source: 'raw',
                file_type: 'jsonl',
                note: '',
            },
        });

        const user = userEvent.setup();
        renderInsideTable(
            <DocumentSampleAccordion projectId={77} documentId={5} colSpan={8} />,
        );
        await screen.findByText(/"x"/);

        await user.click(screen.getByTestId('refresh-sample-5'));
        await waitFor(() => {
            const calls = apiMock.get.mock.calls.filter(
                ([url]) => url === '/projects/77/ingestion/documents/5/sample',
            );
            expect(calls.length).toBeGreaterThanOrEqual(2);
        });
    });

    it('surfaces fetch errors inline', async () => {
        apiMock.get.mockRejectedValue({
            response: { data: { detail: 'Source file not on disk.' } },
        });

        renderInsideTable(
            <DocumentSampleAccordion projectId={77} documentId={9} colSpan={8} />,
        );

        const err = await screen.findByTestId('sample-error-9');
        expect(err.textContent).toContain('Source file not on disk');
    });
});
