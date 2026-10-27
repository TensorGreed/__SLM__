import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import MultiTurnMaskingCard from './MultiTurnMaskingCard';
import type { Experiment } from './trainingPanelTypes';

const BLOCK = {
    rows: 12, assistant_token_share: 0.4321, assistant_turns: 30,
    truncated_rows: 2, dropped_turns: 4, dropped_rows: 1, mask_fallback_rows: 3,
};

function exp(extra: Record<string, unknown>): Experiment {
    return { id: 21, name: 'exp', status: 'completed', training_mode: 'sft', base_model: 'm', ...extra } as Experiment;
}

describe('MultiTurnMaskingCard', () => {
    it('renders the block from config._runtime', () => {
        render(<MultiTurnMaskingCard experiment={exp({ config: { _runtime: { multi_turn: BLOCK } } })} />);
        expect(screen.getByTestId('multi-turn-share')).toHaveTextContent('43.2%');
        expect(screen.getByTestId('multi-turn-turns')).toHaveTextContent('30');
        expect(screen.getByTestId('multi-turn-truncated')).toHaveTextContent('2');
        expect(screen.getByTestId('multi-turn-dropped-turns')).toHaveTextContent('4');
        expect(screen.getByTestId('multi-turn-fallbacks')).toHaveTextContent('3');
        expect(screen.getByTestId('multi-turn-card')).toHaveTextContent('Run #21');
        expect(screen.queryByTestId('multi-turn-approximate')).toBeNull();
    });

    it('tags simulate estimates as approximate', () => {
        render(<MultiTurnMaskingCard experiment={exp({ config: { _runtime: { multi_turn: { ...BLOCK, approximate: true } } } })} />);
        expect(screen.getByTestId('multi-turn-approximate')).toHaveTextContent('approximate');
    });

    it('falls back to training_report.multi_turn', () => {
        render(<MultiTurnMaskingCard experiment={exp({ training_report: { multi_turn: BLOCK } })} />);
        expect(screen.getByTestId('multi-turn-rows')).toHaveTextContent('12');
    });

    it('renders nothing when absent', () => {
        const { container } = render(<MultiTurnMaskingCard experiment={exp({ config: { _runtime: {} } })} />);
        expect(container).toBeEmptyDOMElement();
    });
});
