import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import SynthReviewQueue from './SynthReviewQueue';

const A = 'cloud:anthropic:m';
const B = 'ollama:gpt-oss';
const judge = (j: string, verdict: string, note = 'n', reason_code = verdict === 'pass' ? 'ok' : 'incorrect') => ({
    judge: j, verdict, reason_code, note, job_id: 1,
});
const row = (id: number, consensus: string, judges: ReturnType<typeof judge>[]) => ({
    id, synth_confidence: 0.8, preview: `{"r": ${id}}`, payload: {},
    review_assist: { ...judges[judges.length - 1], backend: judges[judges.length - 1].judge },
    review_assist_judges: judges, assist_consensus: consensus,
});

const groups = [
    { synth_source: 'agree_pass', label: `agree pass · ${A} + ${B}`, count: 2, rows: [
        row(1, 'agree_pass', [judge(A, 'pass'), judge(B, 'pass')]),
        row(2, 'agree_pass', [judge(A, 'pass'), judge(B, 'pass')]),
    ] },
    { synth_source: 'disagree', label: `disagree · ${A} + ${B}`, count: 1, rows: [
        row(3, 'disagree', [judge(A, 'pass', 'looks fine'), judge(B, 'fail', 'wrong unit')]),
    ] },
    { synth_source: 'agree_fail', label: `agree fail · ${A} + ${B}`, count: 1, rows: [
        row(4, 'agree_fail', [judge(A, 'fail'), judge(B, 'fail')]),
    ] },
    { synth_source: 'single:pass', label: `single · pass · ${A}`, count: 1, rows: [
        { ...row(5, 'single', [judge(A, 'pass')]) },
    ] },
    { synth_source: 'none', label: 'none', count: 1, rows: [
        { id: 6, synth_confidence: 0.8, preview: '{"r": 6}', payload: {}, review_assist: null, assist_consensus: 'none' },
    ] },
];

const payload = {
    project_id: 5, dataset_id: 1, total_rows: 6, total_pending: 6, total_accepted: 0,
    facets: {
        jobs: [], backends: [], verdicts: [{ verdict: 'pass', count: 4 }], reason_codes: [{ reason_code: 'ok', count: 4 }],
        consensus: [{ consensus: 'agree_pass', count: 2 }, { consensus: 'disagree', count: 1 }],
        judges: [{ judge: A, count: 5 }, { judge: B, count: 3 }],
    },
    groups, accepted_groups: [],
};

function mockGets(data: unknown = payload) {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/synthetic/backends/cloud/models')) return Promise.resolve({ data: { providers: [] } });
        if (url.endsWith('/synthetic/backends')) return Promise.resolve({ data: { backends: [] } });
        return Promise.resolve({ data });
    });
}

describe('SynthReviewQueue with several judges', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        Object.values(toastMock).forEach((fn) => fn.mockReset());
        mockGets();
    });

    it('shows every judge verdict and note on a multi-judge row and the lone badge on a single one', async () => {
        render(<SynthReviewQueue projectId={5} />);
        const judges = await screen.findByTestId('synth-review-queue-judges-3');
        expect(judges.textContent).toContain('disagree');
        expect(screen.getByTestId(`synth-review-queue-judge-3-${A}`).textContent).toContain('pass — looks fine');
        expect(screen.getByTestId(`synth-review-queue-judge-3-${B}`).textContent).toContain('fail · incorrect — wrong unit');
        expect(screen.queryByTestId('synth-review-queue-judges-5')).toBeNull();
        expect(screen.getByTestId('synth-review-queue-assist-5')).toBeTruthy();
    });

    it('groups by consensus and offers Accept only on agree_pass and a lone judge\'s pass', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_consensus'));
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue', { params: { group_by: 'assist_consensus' } },
        ));
        expect(await screen.findByTestId('synth-review-queue-group-accept-agree_pass')).toBeTruthy();
        expect(screen.getByTestId('synth-review-queue-group-accept-single:pass')).toBeTruthy();
        for (const key of ['disagree', 'agree_fail', 'none']) {
            expect(screen.queryByTestId(`synth-review-queue-group-accept-${key}`)).toBeNull();
        }
        // a failing consensus can be rejected as a group, a disagreement or an un-screened group cannot
        expect(screen.getByTestId('synth-review-queue-group-reject-agree_fail')).toBeTruthy();
        expect(screen.queryByTestId('synth-review-queue-group-reject-disagree')).toBeNull();
        expect(screen.queryByTestId('synth-review-queue-group-reject-none')).toBeNull();
        expect(screen.getAllByTestId('synth-review-queue-group-label')[0].textContent).toBe(`agree pass · ${A} + ${B}`);
    });

    it('re-fetches before the bulk accept and sends only rows still in agree_pass', async () => {
        apiMock.post.mockResolvedValue({ data: { accepted: 1, rejected: 0, not_found: 0, not_pending: 0, total_remaining_pending: 5 } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_consensus'));
        const button = await screen.findByTestId('synth-review-queue-group-accept-agree_pass');
        // Meanwhile a second run flipped row 2 to disagree, so it is no longer an agreement.
        const changed = {
            ...payload,
            groups: [
                { ...groups[0], count: 2, rows: [groups[0].rows[0], { ...groups[0].rows[1], assist_consensus: 'disagree' }] },
                ...groups.slice(1),
            ],
        };
        mockGets(changed);
        await user.click(button);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [1], action: 'accept', reject_reason: null, decided_via: 'group:assist_consensus:agree_pass' },
        ));
    });

    it('never accepts a verdict-group row that another judge disagrees on', async () => {
        apiMock.post.mockResolvedValue({ data: { accepted: 1, rejected: 0, not_found: 0, not_pending: 0, total_remaining_pending: 0 } });
        const passGroup = { synth_source: 'pass', count: 2, rows: [groups[0].rows[0], groups[1].rows[0]] };
        mockGets({ ...payload, groups: [passGroup] });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-verdict'));
        await user.click(await screen.findByTestId('synth-review-queue-group-accept-pass'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [1], action: 'accept', reject_reason: null, decided_via: 'group:verdict:pass' },
        ));
    });

    it('passes the consensus and judge filters as query params', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.selectOptions(await screen.findByTestId('synth-review-queue-filter-consensus'), 'disagree');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue', { params: { assist_consensus: 'disagree' } },
        ));
        await user.selectOptions(screen.getByTestId('synth-review-queue-filter-judge'), B);
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/5/synthetic/review-queue', { params: { assist_consensus: 'disagree', judge: B } },
        ));
    });

    it('starts a second-opinion run for rows another model failed, resuming where it left off', async () => {
        apiMock.post.mockResolvedValue({ data: { id: 9 } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-assist-toggle'));
        await user.click(screen.getByTestId('synth-review-queue-assist-only-missing'));
        await user.selectOptions(screen.getByTestId('synth-review-queue-assist-target'), 'prior_doubt');
        await user.click(screen.getByTestId('synth-review-queue-assist-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/5/synthetic/review-assist');
        expect(body).toMatchObject({ only_missing_judge: true, prior_verdicts: ['fail', 'unsure'] });
        expect(body.consensus).toBeUndefined();
    });
    it('shows each judge pass rate and flags a rubber-stamp judge in plain text', async () => {
        mockGets({
            ...payload,
            facets: {
                ...payload.facets,
                judge_stats: [
                    { judge: A, judged: 40, pass: 30, fail: 8, unsure: 2, pass_rate: 0.75, rubber_stamp: false },
                    { judge: B, judged: 247, pass: 245, fail: 1, unsure: 1, pass_rate: 0.9919, rubber_stamp: true },
                ],
            },
        });
        render(<SynthReviewQueue projectId={5} />);
        const strip = await screen.findByTestId('synth-review-queue-judge-stats');
        expect(screen.getByTestId(`synth-review-queue-judge-stat-${A}`).textContent).toBe(`${A} passes 75% of 40 rows`);
        expect(screen.getByTestId(`synth-review-queue-judge-stat-${B}`).textContent)
            .toBe(`${B} passes 99% of 247 rows; may not be checking`);
        expect(strip.textContent).not.toContain(`${A} passes 75% of 40 rows; may not be checking`);
        // next to a judge's verdict on a row too
        expect(screen.getByTestId(`synth-review-queue-judge-rate-3-${B}`).textContent)
            .toBe(' (passes 99% of 247 rows; may not be checking)');
        expect(screen.getByTestId(`synth-review-queue-judge-rate-3-${A}`).textContent).toBe(' (passes 75% of 40 rows)');
    });

    it('renders no judge strip when the queue carries no judge stats', async () => {
        render(<SynthReviewQueue projectId={5} />);
        await screen.findByTestId('synth-review-queue-judges-3');
        expect(screen.queryByTestId('synth-review-queue-judge-stats')).toBeNull();
        expect(screen.queryByTestId(`synth-review-queue-judge-rate-3-${A}`)).toBeNull();
    });

    it('offers "Calibrate judges" only once some row has a judge verdict', async () => {
        const { unmount } = render(<SynthReviewQueue projectId={5} />);
        await screen.findByTestId('synth-review-queue-judges-3');
        expect(screen.queryByTestId('judge-calibration-card')).toBeNull();
        unmount();
        mockGets({
            ...payload,
            facets: { ...payload.facets, judge_stats: [{ judge: A, judged: 4, pass: 3, fail: 1, unsure: 0, pass_rate: 0.75, rubber_stamp: false }] },
        });
        render(<SynthReviewQueue projectId={5} />);
        expect(await screen.findByTestId('judge-calibration-toggle')).toHaveTextContent('Calibrate judges');
    });

    describe('trust signals (judge calibration V3)', () => {
        const trust = (status: string, extra: Record<string, unknown> = {}) => ({
            status, tier: 'human', fail_recall: null, ci: null, n: 0, label_source: 'human', floor: 0.7, min_labels: 20, ...extra,
        });
        const withTrust = (a: ReturnType<typeof trust>, b: ReturnType<typeof trust>, bStamp = false) => ({
            ...payload,
            facets: {
                ...payload.facets,
                judge_stats: [
                    { judge: A, judged: 40, pass: 30, fail: 8, unsure: 2, pass_rate: 0.75, rubber_stamp: false, trust: a },
                    { judge: B, judged: 247, pass: bStamp ? 247 : 200, fail: bStamp ? 0 : 47, unsure: 0,
                        pass_rate: bStamp ? 1 : 0.8097, rubber_stamp: bStamp, trust: b },
                ],
            },
            groups: groups.map((g) => (g.synth_source === 'agree_pass'
                ? { ...g, judge_trust: [{ judge: A, status: a.status }, { judge: B, status: b.status }], trust: b.status }
                : g.synth_source === 'single:pass' ? { ...g, judge_trust: [{ judge: A, status: a.status }], trust: a.status } : g)),
        });
        const okA = trust('calibrated_ok', { fail_recall: 0.9, ci: [0.75, 0.97], n: 30 });
        const accepted = { data: { accepted: 2, rejected: 0, not_found: 0, not_pending: 0, total_remaining_pending: 4 } };
        const sameRequest = ['/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [1, 2], action: 'accept', reject_reason: null, decided_via: 'group:assist_consensus:agree_pass' }];

        async function openConsensus() {
            const user = userEvent.setup();
            render(<SynthReviewQueue projectId={5} />);
            await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_consensus'));
            return user;
        }

        it('shows a trust badge next to each judge pass rate', async () => {
            mockGets(withTrust(okA, trust('rubber_stamp'), true));
            render(<SynthReviewQueue projectId={5} />);
            await screen.findByTestId('synth-review-queue-judge-stats');
            expect(screen.getByTestId(`synth-review-queue-judge-trust-${A}`).textContent).toBe('calibrated');
            expect(screen.getByTestId(`synth-review-queue-judge-trust-${B}`).textContent).toBe('rubber stamp?');
            expect(screen.getByTestId(`synth-review-queue-judge-stat-${B}`).textContent)
                .toBe(`${B} passes 100% of 247 rows; may not be checkingrubber stamp?`);
        });

        it.each([
            ['below_floor', 'below floor'],
            ['uncalibrated', 'not calibrated'],
            ['insufficient', 'too few labels'],
        ])('badge for %s', async (status, text) => {
            mockGets(withTrust(okA, trust(status)));
            render(<SynthReviewQueue projectId={5} />);
            expect((await screen.findByTestId(`synth-review-queue-judge-trust-${B}`)).textContent).toBe(text);
        });

        it('lists the judges behind an accept group with their trust status', async () => {
            mockGets(withTrust(okA, trust('uncalibrated')));
            await openConsensus();
            const head = await screen.findByTestId('synth-review-queue-group-trust-agree_pass');
            expect(head.textContent).toBe(`${A} calibrated${B} not calibrated`);
            expect(screen.getByTestId('synth-review-queue-group-trust-single:pass').textContent).toBe(`${A} calibrated`);
            expect(screen.queryByTestId('synth-review-queue-group-trust-agree_fail')).toBeNull();
        });

        it.each([
            ['uncalibrated', trust('uncalibrated'), false, `${B}: not calibrated yet; label 20 rows in Calibrate judges`],
            ['below floor', trust('below_floor', { fail_recall: 0.3, ci: [0.12, 0.55], n: 24 }), false,
                `${B} catches at most 55% of bad rows (vs human, n 24)`],
            ['rubber stamp', trust('rubber_stamp'), true, `${B} passes 100% of 247 rows; may not be checking`],
        ])('warns before accepting a group behind a %s judge, and Accept anyway sends the same request', async (_name, bTrust, stamp, text) => {
            apiMock.post.mockResolvedValue(accepted);
            mockGets(withTrust(okA, bTrust, stamp));
            const user = await openConsensus();
            await user.click(await screen.findByTestId('synth-review-queue-group-accept-agree_pass'));
            const confirm = await screen.findByTestId('synth-review-queue-trust-confirm-agree_pass');
            expect(confirm.textContent).toContain(text);
            expect(confirm.textContent).not.toContain(`${A} `); // the calibrated judge is not listed
            expect(apiMock.post).not.toHaveBeenCalled();
            await user.click(screen.getByTestId('synth-review-queue-trust-accept-agree_pass'));
            await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(...sameRequest));
        });

        it('Cancel accepts nothing', async () => {
            mockGets(withTrust(okA, trust('uncalibrated')));
            const user = await openConsensus();
            await user.click(await screen.findByTestId('synth-review-queue-group-accept-agree_pass'));
            await user.click(await screen.findByTestId('synth-review-queue-trust-cancel-agree_pass'));
            expect(screen.queryByTestId('synth-review-queue-trust-confirm-agree_pass')).toBeNull();
            expect(apiMock.post).not.toHaveBeenCalled();
        });

        it('accepts with one click when every judge behind the group is calibrated', async () => {
            apiMock.post.mockResolvedValue(accepted);
            mockGets(withTrust(okA, trust('calibrated_ok', { fail_recall: 0.8, ci: [0.71, 0.9], n: 40 })));
            const user = await openConsensus();
            await user.click(await screen.findByTestId('synth-review-queue-group-accept-agree_pass'));
            await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(...sameRequest));
            expect(screen.queryByTestId('synth-review-queue-trust-confirm-agree_pass')).toBeNull();
        });
    });
});
