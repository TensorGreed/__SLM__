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

    it('runs a playbook and renders the result preview', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 5,
                recipe_id: 'qa-sft',
                playbooks: [{ recipe_id: 'qa-sft', mode: 'positives_paraphrase' }],
            },
        });
        apiMock.post.mockResolvedValue({
            data: {
                rows: [
                    {
                        payload: { question: 'How do I reset?', answer: 'Visit settings.' },
                        synth_confidence: 0.9,
                        synth_source: 'playbook:qa-sft:positives_paraphrase',
                    },
                    {
                        payload: { question: 'Where to reset?', answer: 'Visit settings.' },
                        synth_confidence: 0.85,
                        synth_source: 'playbook:qa-sft:positives_paraphrase',
                    },
                ],
                backend_used: 'ollama:llama3.1:8b',
                elapsed_sec: 1.42,
                prompt_snippet: 'You are generating training data…',
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
        // Result headline + backend.
        expect(screen.getByText(/Generated/)).toBeInTheDocument();
        expect(screen.getByText(/ollama:llama3.1:8b/)).toBeInTheDocument();
        // POST hit the right path with the right body.
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/run-playbook',
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
        apiMock.post.mockResolvedValue({
            data: {
                rows: [{ payload: {}, synth_confidence: 0.9, synth_source: 'playbook:classification:positives_paraphrase' }],
                backend_used: 'nemo:meta/llama-3.1-70b-instruct',
                elapsed_sec: 0.5,
                prompt_snippet: '...',
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
                '/projects/3/synthetic/run-playbook',
                expect.objectContaining({
                    mode: 'positives_paraphrase',
                    backend: 'nemo:meta/llama-3.1-70b-instruct',
                }),
            );
        });
    });
});
