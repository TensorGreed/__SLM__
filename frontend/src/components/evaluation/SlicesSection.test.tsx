/**
 * Quality-Lift phase 7 slice 1 — SlicesSection tests.
 *
 * Pins for the concrete slices editor:
 *   * Loads from /api/projects/{id}/slice-definitions on mount.
 *   * Empty state when the project has no slices yet.
 *   * Add slice → form fields render; entering an invalid slice_id
 *     surfaces the inline regex error; Save stays disabled.
 *   * Valid edit → Save PUTs the cleaned payload.
 *   * Op picker controls the value cell type (numeric vs comma list
 *     vs presence stub).
 *   * "Gate this slice" appears only when slice_id is valid, opens the
 *     modal, and the rendered metric_id matches per_slice.<id>.f1.
 *   * onGateSlice callback (when provided) preempts the modal.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SlicesSection from './SlicesSection';

const EMPTY_RESPONSE = {
    project_id: 1,
    slice_definitions: { slices: [] },
};

const LONG_INPUT_SLICE = {
    slice_id: 'long_input',
    display_name: 'Long inputs',
    where: [{ field: 'input_length', op: 'gte', value: 100 }],
};

describe('SlicesSection', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('renders the empty state when the project has no slices yet', async () => {
        apiMock.get.mockResolvedValueOnce({ data: EMPTY_RESPONSE });
        render(<SlicesSection projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText(/No slice.*defined yet/i)).toBeInTheDocument();
        });
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/slice-definitions');
        // The Add button is rendered even on empty state.
        expect(screen.getByTestId('slices-add')).toBeInTheDocument();
    });

    it('loads existing slices + renders them', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: { slices: [LONG_INPUT_SLICE] },
            },
        });
        render(<SlicesSection projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('slices-item-0-id')).toHaveValue('long_input');
        });
        expect(screen.getByTestId('slices-item-0-display-name')).toHaveValue('Long inputs');
    });

    it('Adding a slice shows the form; Save stays disabled until slice_id is valid', async () => {
        apiMock.get.mockResolvedValueOnce({ data: EMPTY_RESPONSE });
        const user = userEvent.setup();
        render(<SlicesSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('slices-add')).toBeInTheDocument());

        await user.click(screen.getByTestId('slices-add'));
        const save = screen.getByTestId('slices-save');
        // Newly added slice has empty slice_id → invalid → save disabled.
        expect(save).toBeDisabled();

        // Type an invalid slice_id with uppercase — the grammar error
        // surfaces inline.
        const idInput = screen.getByTestId('slices-item-0-id');
        await user.type(idInput, 'BadID');
        expect(
            screen.getByText(/must match.*\^\[a-z\]/i),
        ).toBeInTheDocument();
        expect(save).toBeDisabled();

        // Replace with a valid id — error clears + Save enables.
        await user.clear(idInput);
        await user.type(idInput, 'long_input');
        expect(screen.queryByText(/must match/i)).toBeNull();
        expect(save).not.toBeDisabled();
    });

    it('PUTs the saved payload when the user clicks Save', async () => {
        apiMock.get.mockResolvedValueOnce({ data: EMPTY_RESPONSE });
        apiMock.put.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: { slices: [LONG_INPUT_SLICE] },
            },
        });
        const user = userEvent.setup();
        render(<SlicesSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('slices-add')).toBeInTheDocument());
        await user.click(screen.getByTestId('slices-add'));
        await user.type(screen.getByTestId('slices-item-0-id'), 'long_input');
        await user.type(screen.getByTestId('slices-item-0-display-name'), 'Long inputs');
        await user.click(screen.getByTestId('slices-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const [url, body] = apiMock.put.mock.calls[0];
        expect(url).toBe('/projects/1/slice-definitions');
        expect((body as { slices: unknown[] }).slices).toHaveLength(1);
        expect(((body as { slices: Array<{ slice_id: string }> }).slices[0]).slice_id)
            .toBe('long_input');
    });

    it('op picker changes the value cell type', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: { slices: [LONG_INPUT_SLICE] },
            },
        });
        const user = userEvent.setup();
        render(<SlicesSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('slices-item-0-id')).toBeInTheDocument());

        // gte (numeric) → number input.
        const valueInput = screen.getByTestId('slices-item-0-clause-0-value');
        expect(valueInput).toHaveAttribute('type', 'number');

        // Switch to ``exists`` → presence-check stub renders instead.
        const opSelect = screen.getByTestId('slices-item-0-clause-0-op');
        await user.selectOptions(opSelect, 'exists');
        expect(screen.getByText(/presence check/i)).toBeInTheDocument();

        // Switch to ``in`` → text input that takes comma list.
        await user.selectOptions(opSelect, 'in');
        const inValueInput = screen.getByTestId('slices-item-0-clause-0-value');
        expect(inValueInput).toHaveAttribute('type', 'text');
        expect(inValueInput).toHaveAttribute(
            'placeholder',
            expect.stringContaining('comma'),
        );
    });

    it('Gate this slice opens a modal with the canonical metric_id', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: { slices: [LONG_INPUT_SLICE] },
            },
        });
        const user = userEvent.setup();
        render(<SlicesSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('slices-item-0-id')).toBeInTheDocument());

        const gateBtn = screen.getByTestId('slices-gate-long_input');
        await user.click(gateBtn);

        const dialog = await screen.findByRole('dialog', { name: /Gate this slice/i });
        // Suggested metric_id surfaces in the dialog using the
        // canonical per_slice.<id>.<metric> shape.
        expect(within(dialog).getByText('per_slice.long_input.f1')).toBeInTheDocument();
        // Help text mentions the worst_slice operators so the user
        // learns about that affordance from the same surface.
        expect(within(dialog).getByText(/worst_slice/i)).toBeInTheDocument();
    });

    it('onGateSlice callback preempts the modal when provided', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: { slices: [LONG_INPUT_SLICE] },
            },
        });
        const onGateSlice = vi.fn();
        const user = userEvent.setup();
        render(<SlicesSection projectId={1} onGateSlice={onGateSlice} />);
        await waitFor(() => expect(screen.getByTestId('slices-item-0-id')).toBeInTheDocument());

        await user.click(screen.getByTestId('slices-gate-long_input'));
        expect(onGateSlice).toHaveBeenCalledWith('long_input', 'per_slice.long_input.f1');
        // Modal does NOT appear when the parent handles it.
        expect(screen.queryByRole('dialog')).toBeNull();
    });

    it('Gate this slice button is hidden when slice_id is invalid', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 1,
                slice_definitions: {
                    slices: [{
                        // Trailing dash makes the slice_id invalid per
                        // the regex; the gate button should hide.
                        slice_id: 'BadID',
                        display_name: 'invalid',
                        where: [{ field: 'input_length', op: 'gte', value: 100 }],
                    }],
                },
            },
        });
        render(<SlicesSection projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('slices-item-0-id')).toBeInTheDocument());
        // The grammar error renders.
        expect(screen.getByText(/must match/i)).toBeInTheDocument();
        // No gate button for the invalid slice id.
        expect(screen.queryByTestId(/slices-gate-/)).toBeNull();
    });
});
