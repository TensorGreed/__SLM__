import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import PlaybookPickerPanel from './PlaybookPickerPanel';


describe('PlaybookPickerPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders the available paraphrase mode for a classification recipe', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker')).toBeInTheDocument();
        });
        // Header notes the recipe + the count.
        expect(screen.getByText(/classification/)).toBeInTheDocument();
        expect(screen.getByText(/1 mode available/)).toBeInTheDocument();
        // The single paraphrase mode appears with its label and hint.
        const modeOption = screen.getByTestId('playbook-picker-mode-positives_paraphrase');
        expect(modeOption.textContent).toContain('Paraphrase positives');
        // Auto-selected on first render.
        const radio = modeOption.querySelector('input[type="radio"]') as HTMLInputElement;
        expect(radio.checked).toBe(true);
    });

    it('shows an empty-state message when no playbooks are registered', async () => {
        apiMock.get.mockResolvedValue({
            data: { project_id: 1, recipe_id: null, playbooks: [] },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-empty')).toBeInTheDocument();
        });
        expect(screen.getByText(/Select a recipe/)).toBeInTheDocument();
    });

    it('renders a "pick a recipe first" CTA when the server flags recipe_required', async () => {
        // Legacy project (pre-dating the auto-apply-on-create fix):
        // server returns empty playbooks + recipe_required=true instead
        // of dumping the full cross-task-shape catalog. The panel must
        // surface a directive CTA pointing at the Dataset Import wizard.
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 7,
                recipe_id: null,
                recipe_required: true,
                playbooks: [],
            },
        });
        render(<PlaybookPickerPanel projectId={7} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('playbook-picker-empty-recipe-required'),
            ).toBeInTheDocument();
        });
        const cta = screen.getByTestId('playbook-picker-empty-recipe-required');
        expect(cta.textContent).toMatch(/Pick a recipe first/);
        expect(cta.textContent).toMatch(/Dataset Import wizard/);
    });

    it('runs a playbook via the async-job endpoint and shows a queued confirmation', async () => {
        // Hardening Phase H1 — runs now go through the background-
        // job framework instead of blocking on the LLM call. The
        // panel renders a tiny "job #N queued" confirmation; actual
        // rows surface in the notification bell when the job
        // completes.
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 5,
                recipe_id: 'qa-sft',
                playbooks: [{ recipe_id: 'qa-sft', mode: 'positives_paraphrase' }],
            },
        });
        apiMock.post.mockResolvedValue({
            data: {
                id: 11,
                kind: 'synth_playbook',
                title: 'Synth · positives_paraphrase · 30 rows',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 5,
                user_id: null,
                params: { mode: 'positives_paraphrase', target_count: 30 },
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });

        render(<PlaybookPickerPanel projectId={5} />);

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-result')).toBeInTheDocument();
        });
        // Confirmation preview shows the queued job marker.
        expect(screen.getByText(/job #11 queued/)).toBeInTheDocument();
        // POST hit the async-job variant of the endpoint.
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/run-playbook?async_job=true',
            {
                mode: 'positives_paraphrase',
                target_count: 30,
                target_class: null,
                backend: null,
            },
        );
    });

    it('surfaces a no-backend message when the run hits a 503', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'qa-sft',
                playbooks: [{ recipe_id: 'qa-sft', mode: 'positives_paraphrase' }],
            },
        });
        apiMock.post.mockRejectedValue({
            response: { status: 503, data: { detail: 'No synth backend available.' } },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('playbook-picker-run-error').textContent).toMatch(/No synth backend/i);
    });

    it('clamps the target-count input to the [1, 500] range', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);

        const countInput = await screen.findByTestId('playbook-picker-count') as HTMLInputElement;
        // Default is 30.
        expect(countInput.value).toBe('30');

        // Set to 9999 — should clamp to 500.
        fireEvent.change(countInput, { target: { value: '9999' } });
        expect(countInput.value).toBe('500');

        // Set to 0 — should clamp to 1.
        fireEvent.change(countInput, { target: { value: '0' } });
        expect(countInput.value).toBe('1');
    });

    // ── Backend picker (Epic 5 Phase 5a) ─────────────────────────────

    /** Route-aware GET stub: distinguishes /playbooks from /backends so
     * the panel can fetch both on mount without one overwriting the
     * other. ``backendsResponse`` defaults to a single-backend payload
     * so most tests get the picker-hidden behavior. */
    function installRouter(opts: {
        playbooks: unknown;
        backendsResponse?: unknown;
    }) {
        const backendsResponse = opts.backendsResponse ?? {
            project_id: 1,
            backends: [
                { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
            ],
        };
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/synthetic/backends')) {
                return { data: backendsResponse };
            }
            return { data: opts.playbooks };
        });
    }

    it('hides the backend picker when only one backend is available', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        // Picker UI not in the DOM.
        expect(screen.queryByTestId('playbook-picker-backend')).not.toBeInTheDocument();
    });

    it('shows the backend picker when 2+ backends are available', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct' },
                ],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        // "Auto (recommended)" + one option per available backend.
        expect(select.options.length).toBe(3);
        expect(select.options[0].text).toMatch(/Auto/);
        expect(select.options[1].value).toBe('ollama:llama3.1:8b');
        expect(select.options[2].value).toBe('nemo:meta/llama-3.1-70b-instruct');
    });

    // ── Schema-aware badge (Epic 5 Phase 5c) ─────────────────────────

    it('renders the schema-aware badge when the active backend honors response_schema', async () => {
        // Two backends; vLLM is schema_aware. Default selection is
        // "Auto" → first available backend → ollama (NOT schema-aware),
        // so the badge should NOT render at first.
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'vllm', available: true, describe: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct', schema_aware: true },
                ],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });

        // Auto picks ollama → no badge.
        expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();

        // Switch to vLLM → badge appears.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-backend'),
            'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
        );
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-schema-badge')).toBeInTheDocument();
        });
        // Switch back to Ollama → badge disappears.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-backend'),
            'ollama:llama3.1:8b',
        );
        await waitFor(() => {
            expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();
        });
    });

    it('suffixes schema-aware options in the dropdown text', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct', schema_aware: true },
                    { name: 'vllm', available: true, describe: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct', schema_aware: true },
                ],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        // Auto + 3 backends.
        expect(select.options.length).toBe(4);
        expect(select.options[1].text).toBe('ollama:llama3.1:8b');           // no suffix
        expect(select.options[2].text).toMatch(/nemo:.*· schema-aware$/);
        expect(select.options[3].text).toMatch(/vllm:.*· schema-aware$/);
        // The "what does schema-aware mean?" hint appears when at least
        // one schema-aware option is in the picker.
        expect(screen.getByTestId('playbook-picker-schema-hint')).toBeInTheDocument();
    });

    it('omits the schema hint and badge when no available backend is schema-aware', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'teacher', available: true, describe: 'teacher:llama3', schema_aware: false },
                ],
            },
        });
        render(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('playbook-picker-schema-hint')).not.toBeInTheDocument();
        expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();
    });

    it('passes the selected backend pin through to runPlaybook', async () => {
        installRouter({
            playbooks: {
                project_id: 3,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 3,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct' },
                ],
            },
        });
        // Hardening Phase H1 — async-job POST returns a Job stub
        // (202). The pinned backend still flows through verbatim in
        // the request body.
        apiMock.post.mockResolvedValue({
            data: {
                id: 22,
                kind: 'synth_playbook',
                title: 'Synth · positives_paraphrase · 30 rows',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 3,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });

        render(<PlaybookPickerPanel projectId={3} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });

        // Select the NeMo backend.
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        await userEvent.selectOptions(select, 'nemo:meta/llama-3.1-70b-instruct');
        expect(select.value).toBe('nemo:meta/llama-3.1-70b-instruct');

        // Click generate.
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/3/synthetic/run-playbook?async_job=true',
                expect.objectContaining({
                    mode: 'positives_paraphrase',
                    backend: 'nemo:meta/llama-3.1-70b-instruct',
                }),
            );
        });
    });
});
