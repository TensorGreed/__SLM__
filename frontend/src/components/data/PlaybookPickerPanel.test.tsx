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
});
