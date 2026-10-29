import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SynthReviewQueue from './SynthReviewQueue';

const planRow = { id: 1, synth_confidence: 0.9, preview: '{}', payload: { row_kind: 'structured_plan', plan_json: { steps: [] } } };
const qaRow = { id: 2, synth_confidence: 0.9, preview: '{}', payload: { question: 'q', answer: 'a' } };

function queue(rows: unknown[]) {
    return {
        project_id: 1, dataset_id: 1, total_rows: rows.length, total_pending: rows.length, total_accepted: 0,
        groups: [{ synth_source: 'playbook:x', count: rows.length, truncated: false, rows }],
        accepted_groups: [],
    };
}

const REPORT = {
    dry_run: true, rules: 1, checked: 3, flagged: 2, warned_only: 0, rejected: 0,
    by_rule: { 'widget-sized': { severity: 'reject', description: 'small', count: 2, sample_row_ids: [1, 3] } },
};

async function openAndDryRun(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByTestId('synth-review-queue-rules-toggle'));
    await user.click(screen.getByTestId('synth-review-queue-rules-text'));
    await user.paste('{"rules": []}');
    await user.click(screen.getByTestId('synth-review-queue-rules-dry-run'));
}

describe('SynthReviewQueue plan rules', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('is offered only when pending rows are structured plans', async () => {
        apiMock.get.mockResolvedValue({ data: queue([qaRow]) });
        render(<SynthReviewQueue projectId={1} />);
        await screen.findByTestId('synth-review-queue');
        expect(screen.queryByTestId('synth-review-queue-rules-card')).toBeNull();
    });

    it('dry-runs, shows counts by rule, then applies after confirmation', async () => {
        apiMock.get.mockResolvedValue({ data: queue([planRow]) });
        apiMock.post.mockResolvedValueOnce({ data: REPORT }).mockResolvedValueOnce({ data: { ...REPORT, dry_run: false, rejected: 2 } });
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={1} />);
        await openAndDryRun(user);
        const row = await screen.findByTestId('synth-review-queue-rules-row-widget-sized');
        expect(row.textContent).toMatch(/reject.*2.*#1, #3/);
        expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/check-plan-rules', { rules: '{"rules": []}', dry_run: true });
        await user.click(screen.getByTestId('synth-review-queue-rules-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith('/projects/1/synthetic/check-plan-rules', { rules: '{"rules": []}', dry_run: false }));
    });

    it('does not apply when the confirmation is declined', async () => {
        apiMock.get.mockResolvedValue({ data: queue([planRow]) });
        apiMock.post.mockResolvedValueOnce({ data: REPORT });
        vi.spyOn(window, 'confirm').mockReturnValue(false);
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={1} />);
        await openAndDryRun(user);
        await user.click(await screen.findByTestId('synth-review-queue-rules-apply'));
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });

    it('restores a rejected-reason group after confirmation', async () => {
        const rejected = { id: 9, synth_confidence: 0.9, preview: '{}', payload: { row_kind: 'structured_plan', reject_reason: 'rule_violation:widget-sized' } };
        apiMock.get.mockResolvedValue({ data: { ...queue([planRow]), total_rejected: 1, rejected_groups: [{ synth_source: 'playbook:x', count: 1, truncated: false, rows: [rejected] }] } });
        apiMock.post.mockResolvedValue({ data: { dry_run: false, matched: 1, restored: 1, by_reason: {} } });
        const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={1} />);
        const button = await screen.findByTestId('synth-review-queue-restore-reason-rule_violation:widget-sized');
        await user.click(button);
        expect(apiMock.post).not.toHaveBeenCalled();
        await user.click(button);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/synthetic/review-queue/restore', { reject_reason: 'rule_violation:widget-sized', dry_run: false },
        ));
        expect(confirm).toHaveBeenCalledTimes(2);
    });
});
