/**
 * Quality-Lift phase 7 slice 3 — MultiSeedConfigSection tests.
 *
 * Pins:
 *   * Section renders in the Power Tools tab (it's a power-user
 *     authoring affordance, not basics).
 *   * Section is collapsed by default — every existing project ships
 *     with num_seeds=1 and we don't want to clutter the form.
 *   * Toggle expands the body: base seed + count + explicit + parallel.
 *   * Variance preview banner appears when num_seeds > 1 (cites
 *     seed_group_id + mean − std framing — the no-vanity-metrics rule).
 *   * URL ?expand_multi_seed=1 forces the section open on mount
 *     (coach nudge deep-link).
 *   * URL ?suggested_num_seeds=N (when between 2 and 8) pre-fills the
 *     count and shows the variance preview without further clicks.
 *   * CustomEvent 'brewslm:expand-multi-seed' opens the section + sets
 *     the count (same-page coach activation when the user is already
 *     on training-config).
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));


import TrainingPanel from './TrainingPanel';


function defaultApiMocks() {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/preferences')) {
            return { data: { project_id: 1, preferred_plan_profile: 'balanced' } };
        }
        if (url.includes('/training/runtimes')) {
            return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        }
        if (url.includes('/training/recipes')) {
            return { data: { project_id: 1, recipes: [] } };
        }
        if (url.includes('/training/cloud-burst/catalog')) {
            return { data: { project_id: 1, providers: [], gpu_skus: [] } };
        }
        if (url.includes('/training/cloud-burst/jobs')) {
            return { data: { project_id: 1, count: 0, runs: [] } };
        }
        if (url.includes('/training/experiments')) {
            return { data: [] };
        }
        return { data: {} };
    });
    apiMock.post.mockImplementation(async () => ({ data: {} }));
    apiMock.put.mockImplementation(async () => ({ data: {} }));
    apiMock.delete.mockImplementation(async () => ({ data: {} }));
}


function setUrlSearch(search: string) {
    // Vitest's jsdom respects window.history.replaceState; useEffect
    // reads window.location.search on first render, which is what
    // we're exercising here.
    window.history.replaceState({}, '', `/${search ? `?${search}` : ''}`);
}


describe('TrainingPanel — multi-seed variance section (phase 7 slice 3)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
        defaultApiMocks();
        setUrlSearch(''); // clean URL between tests
    });

    it('section is present in Power Tools tab and starts collapsed', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));

        const section = await screen.findByTestId('training-multi-seed-section');
        expect(section).toBeInTheDocument();
        // Section header is rendered but the body is NOT — collapsed
        // by default keeps the form short for the 99% num_seeds=1 case.
        expect(screen.queryByTestId('training-multi-seed-body')).toBeNull();
        // No variance banner before the user opts in.
        expect(screen.queryByTestId('training-multi-seed-variance-preview')).toBeNull();
    });

    it('toggle expands the body and reveals all four fields', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
        await user.click(await screen.findByTestId('training-multi-seed-toggle'));

        expect(screen.getByTestId('training-multi-seed-body')).toBeInTheDocument();
        expect(screen.getByTestId('training-multi-seed-base')).toBeInTheDocument();
        expect(screen.getByTestId('training-multi-seed-count')).toBeInTheDocument();
        expect(screen.getByTestId('training-multi-seed-explicit')).toBeInTheDocument();
        expect(screen.getByTestId('training-multi-seed-parallel')).toBeInTheDocument();
        // Count starts at 1 (default — single-seed behavior preserved).
        expect(screen.getByTestId('training-multi-seed-count')).toHaveValue(1);
        // No variance banner yet because count is still 1.
        expect(screen.queryByTestId('training-multi-seed-variance-preview')).toBeNull();
    });

    it('bumping count above 1 surfaces the variance preview + active badge', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
        await user.click(await screen.findByTestId('training-multi-seed-toggle'));

        const count = screen.getByTestId('training-multi-seed-count');
        // ``fireEvent.change`` because the controlled+clamped input
        // (Math.min(8, Math.max(1, ...))) makes ``clear()`` then
        // ``type('3')`` produce '13' → clamped to 8. A single atomic
        // change is the test-shaped equivalent of a paste.
        fireEvent.change(count, { target: { value: '3' } });

        const preview = screen.getByTestId('training-multi-seed-variance-preview');
        expect(preview).toBeInTheDocument();
        // Variance preview cites the no-vanity-metrics framing (mean − std).
        expect(preview.textContent).toMatch(/3 independent trainings/i);
        expect(preview.textContent?.toLowerCase()).toContain('mean');
        // Active badge on the header surfaces the count.
        expect(screen.getByTestId('training-multi-seed-active-badge').textContent).toContain('3');
    });

    it('multiple explicit seeds also trigger the variance preview', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
        await user.click(await screen.findByTestId('training-multi-seed-toggle'));

        const explicit = screen.getByTestId('training-multi-seed-explicit');
        await user.type(explicit, '42, 1337, 7');

        const preview = await screen.findByTestId('training-multi-seed-variance-preview');
        // The preview should reflect 3 seeds even though the count
        // input is still at its default — explicit wins.
        expect(preview.textContent).toMatch(/3 independent trainings/i);
    });

    it('URL ?expand_multi_seed=1 auto-expands the section on mount', async () => {
        const user = userEvent.setup();
        setUrlSearch('expand_multi_seed=1');
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));

        // Body present without a manual toggle click — the coach
        // deep-link path.
        await waitFor(() => {
            expect(screen.getByTestId('training-multi-seed-body')).toBeInTheDocument();
        });
        // No suggested count → count stays at 1 and no preview yet.
        expect(screen.getByTestId('training-multi-seed-count')).toHaveValue(1);
        expect(screen.queryByTestId('training-multi-seed-variance-preview')).toBeNull();
    });

    it('URL ?suggested_num_seeds=3 pre-fills count + variance preview', async () => {
        const user = userEvent.setup();
        setUrlSearch('expand_multi_seed=1&suggested_num_seeds=3');
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));

        await waitFor(() => {
            expect(screen.getByTestId('training-multi-seed-body')).toBeInTheDocument();
        });
        // Count pre-filled to the suggested value from the coach nudge.
        expect(screen.getByTestId('training-multi-seed-count')).toHaveValue(3);
        expect(screen.getByTestId('training-multi-seed-variance-preview')).toBeInTheDocument();
        expect(screen.getByTestId('training-multi-seed-active-badge').textContent).toContain('3');
    });

    it('out-of-range suggested_num_seeds is clamped and does not crash', async () => {
        const user = userEvent.setup();
        setUrlSearch('expand_multi_seed=1&suggested_num_seeds=99');
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));

        await waitFor(() => {
            expect(screen.getByTestId('training-multi-seed-body')).toBeInTheDocument();
        });
        // 99 is outside the [2, 8] band the URL reader accepts, so
        // count stays at the default of 1.
        expect(screen.getByTestId('training-multi-seed-count')).toHaveValue(1);
    });

    it('brewslm:expand-multi-seed CustomEvent expands + sets count', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));

        // Same-page coach activation: section is initially collapsed,
        // CustomEvent fires, body appears + count = suggested.
        expect(screen.queryByTestId('training-multi-seed-body')).toBeNull();
        window.dispatchEvent(new CustomEvent('brewslm:expand-multi-seed', {
            detail: { suggestedNumSeeds: 5 },
        }));
        await waitFor(() => {
            expect(screen.getByTestId('training-multi-seed-body')).toBeInTheDocument();
        });
        expect(screen.getByTestId('training-multi-seed-count')).toHaveValue(5);
        expect(screen.getByTestId('training-multi-seed-variance-preview')).toBeInTheDocument();
    });
});
