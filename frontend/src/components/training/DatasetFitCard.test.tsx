import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import DatasetFitCard from './DatasetFitCard';

function wrap(ui: React.ReactNode) {
    return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe('DatasetFitCard', () => {
    it('renders the "ready" headline when coverage clears the gate', () => {
        wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair', 'text'],
                    shape_counts: { qa_pair: 95, text: 95, classification_label: 0 },
                    sampled_rows: 100,
                    compatible_rows: 95,
                    coverage: 0.95,
                    errors: [],
                    warnings: [],
                    hints: [],
                }}
            />,
        );

        expect(
            screen.getByRole('heading', { level: 4, name: /Your dataset looks ready/i }),
        ).toBeInTheDocument();
        // Branch buttons should NOT render when ready.
        expect(
            screen.queryByText(/Three ways to unblock yourself/i),
        ).not.toBeInTheDocument();
    });

    it('renders the blocked headline + required shapes when the contract failed', () => {
        wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair', 'text', 'chat_messages'],
                    shape_counts: {
                        qa_pair: 0,
                        text: 0,
                        chat_messages: 0,
                        classification_label: 387,
                    },
                    sampled_rows: 400,
                    compatible_rows: 0,
                    coverage: 0.0,
                    errors: ['Dataset contract mismatch'],
                    warnings: [],
                    hints: [],
                }}
            />,
        );

        expect(
            screen.getByRole('heading', {
                level: 4,
                name: /Why your dataset isn't ready for SFT/i,
            }),
        ).toBeInTheDocument();
        // The required-shape list (3 entries: qa_pair, text, chat_messages).
        expect(screen.getAllByText(/Question \+ answer/i).length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText(/Plain text/i).length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText(/Chat transcripts/i).length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText(/Coverage:/i)).toBeInTheDocument();
    });

    it('offers the matching alternative task when one shape dominates the sample', () => {
        wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair', 'text', 'chat_messages'],
                    shape_counts: {
                        qa_pair: 0,
                        text: 0,
                        chat_messages: 0,
                        classification_label: 387,
                    },
                    sampled_rows: 400,
                    compatible_rows: 0,
                    coverage: 0.0,
                    errors: ['Dataset contract mismatch'],
                    warnings: [],
                    hints: [],
                }}
            />,
        );

        const switchLink = screen.getByRole('link', {
            name: /Switch task to Classification/i,
        });
        expect(switchLink).toHaveAttribute('href', '/project/42/training-config');
        // It mentions the coverage percentage of the alternative.
        expect(switchLink).toHaveTextContent(/97% match/i);
    });

    it('omits the switch-task branch when no shape dominates', () => {
        wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair', 'text', 'chat_messages'],
                    shape_counts: {
                        qa_pair: 0,
                        text: 0,
                        chat_messages: 0,
                        classification_label: 12,
                        seq2seq_pair: 8,
                    },
                    sampled_rows: 400,
                    compatible_rows: 0,
                    coverage: 0.0,
                    errors: ['Dataset contract mismatch'],
                }}
            />,
        );

        expect(screen.queryByText(/Switch task to/i)).not.toBeInTheDocument();
        // The other two branches still render.
        expect(screen.getByRole('link', { name: /Map your columns/i })).toHaveAttribute(
            'href',
            '/project/42/adapter-studio',
        );
        expect(screen.getByRole('link', { name: /Start from a demo/i })).toHaveAttribute(
            'href',
            '/',
        );
    });

    it('renders nothing when the contract didn’t run (no sample + no errors)', () => {
        const { container } = wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair'],
                    shape_counts: {},
                    sampled_rows: 0,
                    compatible_rows: 0,
                    coverage: 0,
                    errors: [],
                }}
            />,
        );
        expect(container.querySelector('.dataset-fit-card')).toBeNull();
    });

    it('marks required shapes with a check when their count > 0 in sample', () => {
        wrap(
            <DatasetFitCard
                projectId={42}
                contract={{
                    task_type: 'causal_lm',
                    required_shapes: ['qa_pair'],
                    shape_counts: { qa_pair: 87, classification_label: 0 },
                    sampled_rows: 100,
                    compatible_rows: 87,
                    coverage: 0.87,
                    errors: [
                        'Dataset contract mismatch for task_type=causal_lm: compatible coverage 87% is below required 90%.',
                    ],
                }}
            />,
        );

        // Scope to the shape-counts list (the per-shape progress rows).
        const countsList = document.querySelector(
            '.dataset-fit-card__shape-counts',
        ) as HTMLElement | null;
        expect(countsList).not.toBeNull();
        const qaRow = within(countsList as HTMLElement)
            .getByText('Question + answer')
            .closest('li');
        expect(qaRow).not.toBeNull();
        expect(within(qaRow as HTMLElement).getByText(/✓/)).toBeInTheDocument();
    });
});
