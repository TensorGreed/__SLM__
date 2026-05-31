import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import AutofixPreviewModal from './AutofixPreviewModal';

const onClose = vi.fn();
const onApplied = vi.fn();

function renderModal(fixKind: string, fixLabel: string) {
    return render(
        <AutofixPreviewModal
            projectId={42}
            fixKind={fixKind}
            fixLabel={fixLabel}
            onClose={onClose}
            onApplied={onApplied}
        />,
    );
}

describe('AutofixPreviewModal', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        onClose.mockReset();
        onApplied.mockReset();
    });

    it('renders the drop_failed_docs filename list from the preview', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'drop_failed_docs',
                would_apply_count: 2,
                summary: 'Would drop 2 failed documents.',
                details: {},
                items: [
                    { kind: 'document', id: 1, filename: 'bad-1.pdf', error: 'PDF parse error' },
                    { kind: 'document', id: 2, filename: 'bad-2.pdf', error: '' },
                ],
                safe_to_apply: true,
            },
        });
        renderModal('drop_failed_docs', 'Drop failed docs');
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-list')).toBeInTheDocument();
        });
        const body = screen.getByTestId('autofix-modal').textContent ?? '';
        expect(body).toMatch(/bad-1\.pdf/);
        expect(body).toMatch(/bad-2\.pdf/);
        expect(screen.getByTestId('autofix-modal-apply').textContent).toMatch(/Apply \(2\)/);
    });

    it('renders the canonicalise_labels merge map with each variant', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'canonicalise_labels',
                would_apply_count: 4,
                summary: 'Would merge 1 group; 4 rows touched.',
                details: { merge_group_count: 1 },
                items: [
                    {
                        kind: 'label_merge',
                        canonical: 'positive',
                        canonical_count: 10,
                        merge_in: [
                            { label: 'Positive', count: 3 },
                            { label: 'POSITIVE', count: 1 },
                        ],
                        rows_touched: 4,
                    },
                ],
                safe_to_apply: true,
            },
        });
        renderModal('canonicalise_labels', 'Merge label variants');
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-list')).toBeInTheDocument();
        });
        const body = screen.getByTestId('autofix-modal').textContent ?? '';
        // Both source variants are shown.
        expect(body).toMatch(/Positive/);
        expect(body).toMatch(/POSITIVE/);
        // Canonical target is shown.
        expect(body).toMatch(/positive/);
        // Count info per variant is rendered.
        expect(body).toMatch(/3 row/);
        expect(body).toMatch(/canonical/);
    });

    it('Apply triggers POST /autofix then onApplied + onClose', async () => {
        apiMock.post
            .mockResolvedValueOnce({
                data: {
                    fix_kind: 'drop_failed_docs',
                    would_apply_count: 1,
                    summary: 'Would drop 1 failed document.',
                    details: {},
                    items: [{ kind: 'document', id: 1, filename: 'bad.pdf', error: '' }],
                    safe_to_apply: true,
                },
            })
            .mockResolvedValueOnce({
                data: {
                    fix_kind: 'drop_failed_docs',
                    applied_count: 1,
                    summary: 'Dropped 1 failed document.',
                    details: { dropped_filenames: ['bad.pdf'] },
                },
            });
        renderModal('drop_failed_docs', 'Drop failed docs');
        const user = userEvent.setup();
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-apply')).toBeInTheDocument();
        });
        await user.click(screen.getByTestId('autofix-modal-apply'));
        await waitFor(() => {
            expect(onApplied).toHaveBeenCalledWith(expect.objectContaining({
                fix_kind: 'drop_failed_docs',
                applied_count: 1,
            }));
        });
        expect(onClose).toHaveBeenCalled();
        // Two POSTs: preview, then apply.
        expect(apiMock.post).toHaveBeenNthCalledWith(
            1, '/projects/42/data-health/autofix/preview', { fix_kind: 'drop_failed_docs' },
        );
        expect(apiMock.post).toHaveBeenNthCalledWith(
            2, '/projects/42/data-health/autofix', { fix_kind: 'drop_failed_docs' },
        );
    });

    it('Apply is disabled when safe_to_apply is false', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'redact_pii',
                would_apply_count: 0,
                summary: 'PII redaction is unsafe for span-extraction recipes.',
                details: { blocked_reason: 'span_extraction_needs_pii' },
                items: [],
                safe_to_apply: false,
            },
        });
        renderModal('redact_pii', 'Redact PII');
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-blocked')).toBeInTheDocument();
        });
        const btn = screen.getByTestId('autofix-modal-apply') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
    });

    it('Apply is disabled when would_apply_count is 0', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'drop_failed_docs',
                would_apply_count: 0,
                summary: 'No failed documents to drop.',
                details: {},
                items: [],
                safe_to_apply: true,
            },
        });
        renderModal('drop_failed_docs', 'Drop failed docs');
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-items-empty')).toBeInTheDocument();
        });
        const btn = screen.getByTestId('autofix-modal-apply') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
    });

    it('Cancel closes the modal without firing the apply POST', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                fix_kind: 'drop_failed_docs',
                would_apply_count: 1,
                summary: 'Would drop 1.',
                details: {},
                items: [{ kind: 'document', id: 1, filename: 'bad.pdf', error: '' }],
                safe_to_apply: true,
            },
        });
        renderModal('drop_failed_docs', 'Drop failed docs');
        const user = userEvent.setup();
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-cancel')).toBeInTheDocument();
        });
        await user.click(screen.getByTestId('autofix-modal-cancel'));
        expect(onClose).toHaveBeenCalled();
        expect(onApplied).not.toHaveBeenCalled();
        expect(apiMock.post).toHaveBeenCalledTimes(1); // only preview
    });

    it('shows an error if the preview fetch fails', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'Project not found' } },
        });
        renderModal('drop_failed_docs', 'Drop failed docs');
        await waitFor(() => {
            expect(screen.getByTestId('autofix-modal-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('autofix-modal-error').textContent).toMatch(/Project not found/);
        // Apply stays disabled when preview is unavailable.
        const btn = screen.getByTestId('autofix-modal-apply') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
    });
});
