/**
 * Quality-Lift phase 7 slice 2 — BehavioralTestsSection tests.
 *
 * Pins:
 *   * Loads from /api/projects/{id}/behavioral-tests AND
 *     /api/projects/{id}/slice-definitions on mount (slice ids
 *     surface in the Gate-this-test modal).
 *   * Empty state on fresh project.
 *   * Add → kind defaults to INV → form shows seed_examples +
 *     perturbations + same_label expectation.
 *   * Switch kind to MFT → form shows examples (not seed_examples);
 *     kind-specific state from INV is discarded.
 *   * Switch kind to DIR → expectation kind picker appears;
 *     ``must_change_to`` shows the target_label input.
 *   * Perturbation kind picker controls the param fields
 *     (typo → intensity; insert_token → token + position; etc.).
 *   * Gate-this-test modal shows the top-level metric_id AND a
 *     per-slice variant per project slice when slices exist.
 *   * Modal omits the per-slice section when no slices configured.
 *   * Save PUTs the cleaned payload (kind-specific fields stripped
 *     for the other kinds).
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        put: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import BehavioralTestsSection from './BehavioralTestsSection';

const EMPTY_BT_RESPONSE = {
    project_id: 1,
    task_profile: 'classification',
    behavioral_tests: [],
};

const EMPTY_SLICES_RESPONSE = {
    project_id: 1,
    slice_definitions: { slices: [] },
};

const TWO_SLICES_RESPONSE = {
    project_id: 1,
    slice_definitions: {
        slices: [
            { slice_id: 'long_input', display_name: 'Long', where: [{ field: 'input_length', op: 'gte', value: 100 }] },
            { slice_id: 'short_input', display_name: 'Short', where: [{ field: 'input_length', op: 'lt', value: 100 }] },
        ],
    },
};


// The component fetches behavioral tests + slice definitions in
// parallel; mock both for every test setup.
function mockGetSequence(btResp: unknown, slicesResp: unknown) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/behavioral-tests')) return { data: btResp };
        if (url.endsWith('/slice-definitions')) return { data: slicesResp };
        return { data: {} };
    });
}


describe('BehavioralTestsSection', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('renders the empty state and fetches both endpoints', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => {
            expect(screen.getByText(/No test.* defined yet/i)).toBeInTheDocument();
        });
        // Both endpoints were hit in parallel on mount.
        const urls = apiMock.get.mock.calls.map((c) => c[0]);
        expect(urls).toContain('/projects/1/behavioral-tests');
        expect(urls).toContain('/projects/1/slice-definitions');
    });

    it('Add → INV defaults render seed_examples + perturbations + same_label', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-add')).toBeInTheDocument());

        await user.click(screen.getByTestId('bt-add'));
        // Kind dropdown defaults to INV.
        expect(screen.getByTestId('bt-item-0-kind')).toHaveValue('INV');
        // Seed_examples subgroup renders (not "examples" — that's MFT).
        expect(screen.getByText('seed_examples')).toBeInTheDocument();
        expect(screen.getByText('perturbations')).toBeInTheDocument();
        // Same-label is the implicit expectation for INV → no DIR
        // expectation picker.
        expect(screen.queryByText('expectation')).toBeNull();
    });

    it('Switching kind to MFT shows examples, drops seed_examples', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-add')).toBeInTheDocument());

        await user.click(screen.getByTestId('bt-add'));
        await user.selectOptions(screen.getByTestId('bt-item-0-kind'), 'MFT');

        expect(screen.getByText('examples')).toBeInTheDocument();
        // No INV/DIR-only subgroups on an MFT.
        expect(screen.queryByText('seed_examples')).toBeNull();
        expect(screen.queryByText('perturbations')).toBeNull();
    });

    it('Switching kind to DIR shows the expectation picker + target_label', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-add')).toBeInTheDocument());

        await user.click(screen.getByTestId('bt-add'));
        await user.selectOptions(screen.getByTestId('bt-item-0-kind'), 'DIR');

        // Expectation kind picker renders for DIR.
        const dirKind = screen.getByTestId('bt-item-0-dir-kind');
        expect(dirKind).toBeInTheDocument();
        // Defaults to must_change → no target_label input rendered.
        expect(screen.queryByTestId('bt-item-0-dir-target')).toBeNull();
        // Switch to must_change_to → target_label input appears.
        await user.selectOptions(dirKind, 'must_change_to');
        expect(screen.getByTestId('bt-item-0-dir-target')).toBeInTheDocument();
    });

    it('Perturbation kind picker controls the param fields', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-add')).toBeInTheDocument());
        await user.click(screen.getByTestId('bt-add'));

        // Default is typo → intensity input present.
        expect(screen.getByTestId('bt-item-0-pert-0-intensity')).toBeInTheDocument();

        // Switch to insert_token → token + position inputs appear.
        const pertKind = screen.getByTestId('bt-item-0-pert-0-kind');
        await user.selectOptions(pertKind, 'insert_token');
        expect(screen.getByTestId('bt-item-0-pert-0-token')).toBeInTheDocument();
        expect(screen.getByTestId('bt-item-0-pert-0-position')).toBeInTheDocument();

        // Switch to case_change → case select replaces them.
        await user.selectOptions(pertKind, 'case_change');
        expect(screen.getByTestId('bt-item-0-pert-0-case')).toBeInTheDocument();
        expect(screen.queryByTestId('bt-item-0-pert-0-token')).toBeNull();
    });

    it('Gate-this-test modal renders top-level + per-slice metric_ids when slices exist', async () => {
        // Project already has one test + two slices configured.
        mockGetSequence(
            {
                project_id: 1,
                task_profile: 'classification',
                behavioral_tests: [{
                    test_id: 'typo_invariance',
                    kind: 'INV',
                    description: 'Typos should not change predictions.',
                    seed_examples: [{ input: 'great', given_label: 'positive' }],
                    perturbations: [{ kind: 'typo', intensity: 0.05 }],
                }],
            },
            TWO_SLICES_RESPONSE,
        );
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('bt-item-0-id')).toBeInTheDocument();
        });

        await user.click(screen.getByTestId('bt-gate-typo_invariance'));
        const dialog = await screen.findByRole('dialog', { name: /Gate this test/i });
        // Top-level metric_id always present.
        expect(within(dialog).getByText('behavioral.typo_invariance.pass_rate')).toBeInTheDocument();
        // Per-slice variants — one per slice from the slice_definitions load.
        expect(within(dialog).getByText('behavioral.typo_invariance.per_slice.long_input.pass_rate')).toBeInTheDocument();
        expect(within(dialog).getByText('behavioral.typo_invariance.per_slice.short_input.pass_rate')).toBeInTheDocument();
    });

    it('Gate-this-test modal omits the per-slice section when no slices configured', async () => {
        mockGetSequence(
            {
                project_id: 1,
                task_profile: 'classification',
                behavioral_tests: [{
                    test_id: 'typo_invariance',
                    kind: 'INV',
                    seed_examples: [{ input: 'great' }],
                    perturbations: [{ kind: 'typo', intensity: 0.05 }],
                }],
            },
            EMPTY_SLICES_RESPONSE,
        );
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-item-0-id')).toBeInTheDocument());

        await user.click(screen.getByTestId('bt-gate-typo_invariance'));
        const dialog = await screen.findByRole('dialog', { name: /Gate this test/i });
        // Top-level still present.
        expect(within(dialog).getByText('behavioral.typo_invariance.pass_rate')).toBeInTheDocument();
        // No per-slice header.
        expect(within(dialog).queryByText(/Per-slice variants/i)).toBeNull();
    });

    it('Save PUTs the cleaned payload with kind-specific fields only', async () => {
        mockGetSequence(EMPTY_BT_RESPONSE, EMPTY_SLICES_RESPONSE);
        apiMock.put.mockResolvedValueOnce({
            data: {
                project_id: 1,
                task_profile: 'classification',
                behavioral_tests: [],
            },
        });
        const user = userEvent.setup();
        render(<BehavioralTestsSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('bt-add')).toBeInTheDocument());

        await user.click(screen.getByTestId('bt-add'));
        await user.type(screen.getByTestId('bt-item-0-id'), 'typo_inv');
        // Type a non-empty seed input so the form is valid.
        const seedInput = screen.getByPlaceholderText('input');
        await user.type(seedInput, 'great product');

        await user.click(screen.getByTestId('bt-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const [url, body] = apiMock.put.mock.calls[0];
        expect(url).toBe('/projects/1/behavioral-tests');
        const test = (body as { behavioral_tests: Array<Record<string, unknown>> }).behavioral_tests[0];
        expect(test.test_id).toBe('typo_inv');
        expect(test.kind).toBe('INV');
        // INV: examples (MFT-only) should be stripped.
        expect(test.examples).toBeUndefined();
        // INV: seed_examples + perturbations present.
        expect((test.seed_examples as unknown[]).length).toBeGreaterThan(0);
    });
});
