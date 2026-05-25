import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, mockLocation } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    mockLocation: { pathname: '/', search: '', hash: '' },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../utils/workflowGraphPrefill', () => ({
    loadWorkflowStagePrefill: vi.fn(async () => null),
}));
// SyntheticPanel uses `useLocation` (Phase 5c) to read the Coach
// landing's ?focus_synth_source param + #synth-review-queue hash.
// Stub the hook so the panel can be rendered without a <Router>.
vi.mock('react-router-dom', () => ({
    useLocation: () => mockLocation,
}));

import SyntheticPanel from './SyntheticPanel';

// Map of GET URLs → response data for the bevy of mount-time fetches
// the panel fires (prepared-manifest, playbooks, review-queue,
// cleaning/chunks). We keep these minimal — none of the children
// under test depend on their content other than "the request didn't
// throw."
function installGetRouter(overrides: Record<string, any> = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('prepared-manifest')) {
            return { data: overrides['prepared-manifest'] ?? {} };
        }
        if (url.includes('/synthetic/playbooks')) {
            return {
                data: overrides['playbooks'] ?? {
                    project_id: 1,
                    recipe_id: null,
                    playbooks: [],
                },
            };
        }
        if (url.includes('/synthetic/review-queue')) {
            return {
                data: overrides['review-queue'] ?? {
                    project_id: 1,
                    dataset_id: null,
                    total_rows: 0,
                    total_pending: 0,
                    total_accepted: 0,
                    groups: [],
                    accepted_groups: [],
                },
            };
        }
        if (url.includes('cleaning/chunks')) {
            return {
                data: overrides['cleaning/chunks'] ?? {
                    chunks: [],
                    total: 0,
                },
            };
        }
        if (url.includes('/synthetic/tasks/')) {
            // Default polling reply: completed immediately with zero
            // rows. Tests that exercise the async path override this.
            return {
                data: overrides['tasks'] ?? {
                    task_id: 'task-1',
                    task_kind: 'qa',
                    status: 'completed',
                    target_rows: 0,
                    rows_so_far: 0,
                    batches_done: 0,
                    batches_total: 0,
                    rows: [],
                    error: null,
                },
            };
        }
        return { data: {} };
    });
}


describe('SyntheticPanel — QA + Conversation parity (USER-SUCCESS Epic 2c)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        installGetRouter();
        // Reset the URL stub each test — Phase 5c plumbing reads it.
        mockLocation.pathname = '/';
        mockLocation.search = '';
        mockLocation.hash = '';
    });

    it('shows the "sample randomly from all cleaned chunks" toggle in QA mode (not just span)', async () => {
        render(<SyntheticPanel projectId={1} />);
        // QA mode is the default for non-span projects.
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        // The toggle used to be span-only — now it's a shared row and
        // must be visible in QA mode too.
        expect(screen.getByTestId('synth-use-all-chunks')).toBeInTheDocument();
    });

    it('lets the QA "pairs to generate" input accept values above 50', async () => {
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const input = screen.getByTestId('qa-num-pairs') as HTMLInputElement;
        expect(input.getAttribute('max')).toBe('5000');
        // fireEvent.change is more reliable than userEvent.type for
        // number inputs since the latter sometimes prepends to existing
        // digits rather than replacing them.
        fireEvent.change(input, { target: { value: '500' } });
        expect(input.value).toBe('500');
        // Hint reflects the batched path (500 / 50 = 10 batches).
        expect(screen.getByText(/Will run in 10 background batches/i)).toBeInTheDocument();
    });

    it('lets the Conversation "dialogues" input accept values above 20 and shows the batched hint', async () => {
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        // Switch generation mode to conversation.
        const modeSelect = screen.getByTestId('synth-generation-mode') as HTMLSelectElement;
        await userEvent.selectOptions(modeSelect, 'conversation');
        const input = screen.getByTestId('conversation-num-dialogues') as HTMLInputElement;
        expect(input.getAttribute('max')).toBe('5000');
        fireEvent.change(input, { target: { value: '50' } });
        // Per-batch cap for conversations is 5 → 50 / 5 = 10 batches.
        expect(screen.getByText(/Will run in 10 background batches/i)).toBeInTheDocument();
    });

    it('uses the QA async endpoint when num_pairs > 50', async () => {
        installGetRouter({
            tasks: {
                task_id: 'task-qa-1',
                task_kind: 'qa',
                status: 'completed',
                target_rows: 75,
                rows_so_far: 75,
                batches_done: 2,
                batches_total: 2,
                rows: [{ question: 'q', answer: 'a' }],
                error: null,
            },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                task_id: 'task-qa-1',
                status: 'pending',
                target_rows: 75,
                batches_total: 2,
            },
        });

        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const input = screen.getByTestId('qa-num-pairs') as HTMLInputElement;
        fireEvent.change(input, { target: { value: '75' } });
        // Provide source text so the form isn't disabled.
        const sourceTextarea = screen.getByPlaceholderText(/Paste domain text here/i);
        fireEvent.change(sourceTextarea, { target: { value: 'seed text seed text' } });
        await userEvent.click(screen.getByRole('button', { name: /Generate/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/generate-async',
                expect.objectContaining({
                    target_rows: 75,
                    use_all_chunks: false,
                }),
            );
        });
    });

    it('uses the conversation async endpoint when num_dialogues > 5', async () => {
        installGetRouter({
            tasks: {
                task_id: 'task-conv-1',
                task_kind: 'conversation',
                status: 'completed',
                target_rows: 10,
                rows_so_far: 10,
                batches_done: 2,
                batches_total: 2,
                rows: [{ turns: [{ role: 'user', content: 'hi' }] }],
                error: null,
            },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                task_id: 'task-conv-1',
                status: 'pending',
                target_rows: 10,
                batches_total: 2,
            },
        });

        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const modeSelect = screen.getByTestId('synth-generation-mode') as HTMLSelectElement;
        await userEvent.selectOptions(modeSelect, 'conversation');
        const input = screen.getByTestId('conversation-num-dialogues') as HTMLInputElement;
        fireEvent.change(input, { target: { value: '10' } });
        const sourceTextarea = screen.getByPlaceholderText(/Paste domain text here/i);
        fireEvent.change(sourceTextarea, { target: { value: 'seed text seed text' } });
        await userEvent.click(screen.getByRole('button', { name: /Generate/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/generate-conversations-async',
                expect.objectContaining({
                    target_rows: 10,
                    min_turns: 3,
                    use_all_chunks: false,
                }),
            );
        });
    });

    it('forwards ?focus_synth_source from the URL to SynthReviewQueue (Phase 5c)', async () => {
        // Coach Mode lands the user here with a focus source on the
        // URL. The panel must read it + pass it down so the queue
        // renders its focused banner. We override the review-queue
        // fetch to include a matching pending group; SynthReviewQueue's
        // banner only renders when the source matches a group.
        installGetRouter({
            'review-queue': {
                project_id: 1,
                dataset_id: 7,
                total_rows: 3,
                total_pending: 3,
                total_accepted: 0,
                groups: [
                    {
                        synth_source: 'playbook:classification:class_balance_fill:class=technical',
                        count: 3,
                        truncated: false,
                        rows: [
                            { id: 1, synth_confidence: 0.9, preview: 'a', payload: {} },
                            { id: 2, synth_confidence: 0.85, preview: 'b', payload: {} },
                            { id: 3, synth_confidence: 0.8, preview: 'c', payload: {} },
                        ],
                    },
                ],
                accepted_groups: [],
            },
        });
        mockLocation.search =
            '?focus_synth_source=playbook%3Aclassification%3Aclass_balance_fill%3Aclass%3Dtechnical';
        mockLocation.hash = '#synth-review-queue';

        render(<SyntheticPanel projectId={1} />);

        // The focused banner appears with the right copy.
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-focus-banner'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('synth-review-queue-focus-source').textContent,
        ).toBe('playbook:classification:class_balance_fill:class=technical');
        const acceptAll = screen.getByTestId('synth-review-queue-focus-accept-all');
        expect(acceptAll.textContent).toMatch(/Accept all 3 rows/);
    });
});
