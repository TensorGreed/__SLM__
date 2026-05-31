import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ClassificationChartsPanel from './ClassificationChartsPanel';

describe('ClassificationChartsPanel', () => {
    it('returns null when neither per_class nor confusion_matrix is populated', () => {
        const { container } = render(
            <ClassificationChartsPanel perClass={{}} confusionMatrix={{}} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders bars sorted by F1 ascending so the worst class is first', () => {
        const perClass = {
            good: { precision: 0.95, recall: 0.95, f1: 0.95, support: 100 },
            bad: { precision: 0.30, recall: 0.20, f1: 0.24, support: 25 },
            mid: { precision: 0.70, recall: 0.65, f1: 0.67, support: 50 },
        };
        render(
            <ClassificationChartsPanel
                perClass={perClass}
                confusionMatrix={{}}
                macroF1={0.62}
                accuracy={0.68}
            />,
        );

        // Headline metrics surface in the head.
        const head = screen.getByTestId('cls-charts').textContent || '';
        expect(head).toMatch(/accuracy/);
        expect(head).toMatch(/68\.0%/);
        expect(head).toMatch(/macro-F1/);
        expect(head).toMatch(/62\.0%/);

        // All three bar rows render.
        expect(screen.getByTestId('cls-bar-good')).toBeInTheDocument();
        expect(screen.getByTestId('cls-bar-bad')).toBeInTheDocument();
        expect(screen.getByTestId('cls-bar-mid')).toBeInTheDocument();

        // DOM order matches F1-ascending (bad → mid → good). We compare
        // bounding positions of the rows in the SVG since they're rendered
        // top-to-bottom by document order.
        const rows = screen.getAllByTestId(/^cls-bar-/);
        const labels = rows.map((row) => row.getAttribute('data-testid'));
        expect(labels).toEqual(['cls-bar-bad', 'cls-bar-mid', 'cls-bar-good']);
    });

    it('renders the confusion matrix with each gold-class row and its predicted columns', () => {
        const confusionMatrix = {
            spam: { spam: 9, ham: 1, __unparseable__: 0 },
            ham: { spam: 2, ham: 7, __unparseable__: 1 },
        };
        render(
            <ClassificationChartsPanel
                perClass={{}}
                confusionMatrix={confusionMatrix}
                candidates={['spam', 'ham']}
            />,
        );

        expect(screen.getByTestId('cls-charts-matrix')).toBeInTheDocument();
        // Each row exists.
        expect(screen.getByTestId('cls-matrix-row-spam')).toBeInTheDocument();
        expect(screen.getByTestId('cls-matrix-row-ham')).toBeInTheDocument();
        // Diagonal cells (correct predictions) are present.
        expect(screen.getByTestId('cls-matrix-cell-spam-spam')).toBeInTheDocument();
        expect(screen.getByTestId('cls-matrix-cell-ham-ham')).toBeInTheDocument();
        // Off-diagonal mistake.
        expect(screen.getByTestId('cls-matrix-cell-spam-ham')).toBeInTheDocument();
        // Unparseable column is included.
        expect(screen.getByTestId('cls-matrix-cell-ham-__unparseable__')).toBeInTheDocument();
    });

    it('uses the candidate set order for columns and pushes __unparseable__ to the end', () => {
        // candidates in non-alphabetical order should be respected.
        const matrix = {
            beta: { alpha: 2, beta: 8, __unparseable__: 0 },
            alpha: { alpha: 10, beta: 0, __unparseable__: 0 },
        };
        render(
            <ClassificationChartsPanel
                perClass={{}}
                confusionMatrix={matrix}
                candidates={['beta', 'alpha']}
            />,
        );
        // Verify the unparseable cell is the rightmost — its data-testid
        // matches the column we positioned last. The svg renders cells in
        // DOM order so the last cell-with-__unparseable__-col is in DOM
        // *after* the alpha + beta columns for the same row.
        const betaRow = screen.getByTestId('cls-matrix-row-beta');
        const cells = betaRow.querySelectorAll('[data-testid^="cls-matrix-cell-beta-"]');
        const orderedTestIds = Array.from(cells).map((n) => n.getAttribute('data-testid'));
        expect(orderedTestIds).toEqual([
            'cls-matrix-cell-beta-beta',
            'cls-matrix-cell-beta-alpha',
            'cls-matrix-cell-beta-__unparseable__',
        ]);
    });
});
