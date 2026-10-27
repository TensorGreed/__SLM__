import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import MultiTurnParamsForm from './MultiTurnParamsForm';

describe('MultiTurnParamsForm', () => {
    it('emits defaults and updates on change', async () => {
        const onChange = vi.fn();
        render(<MultiTurnParamsForm onChange={onChange} />);
        expect(onChange).toHaveBeenLastCalledWith({
            turns_min: 2, turns_max: 5, passages_per_dialogue: 2, dialogues_per_passage_group: 1,
        });
        const user = userEvent.setup();
        const passages = screen.getByTestId('mt-passages');
        await user.clear(passages);
        await user.type(passages, '3');
        expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ passages_per_dialogue: 3 }));
    });

    it('never emits fewer than two exchanges (the server rejects turns_min below 2)', async () => {
        const onChange = vi.fn();
        render(<MultiTurnParamsForm onChange={onChange} />);
        const user = userEvent.setup();
        for (const id of ['mt-turns-min', 'mt-turns-max']) {
            const input = screen.getByTestId(id);
            expect(input).toHaveAttribute('min', '2');
            await user.clear(input);  // an emptied field falls back to the minimum
        }
        expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ turns_min: 2, turns_max: 2 }));
    });
});
