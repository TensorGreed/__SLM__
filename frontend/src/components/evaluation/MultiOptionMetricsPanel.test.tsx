import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import MultiOptionMetricsPanel, { OptionCard, widthStep } from './MultiOptionMetricsPanel';

describe('MultiOptionMetricsPanel', () => {
    it('says when the verifier could not run and when parts were not evaluated', () => {
        render(
            <MultiOptionMetricsPanel
                provenance="Run #3 · test · result 9"
                metrics={{
                    scoring_mode: 'multi_option',
                    multi_option: { formats: ['format_a'], row_pass: 'any_option', min_options: 1, max_options: 3, verifier: null },
                    multi_option_pass_rate: null,
                    verifier_error_rows: 4,
                    verifier_unavailable: "option verifier 'v' is not allow-listed (CODE_CHECK_ALLOWED)",
                    total: 0,
                    correct: 0,
                }}
            />,
        );
        expect(screen.getByTestId('mo-verifier-unavailable')).toHaveTextContent('not allow-listed');
        expect(screen.getByTestId('mo-verifier-errors')).toHaveTextContent('4 rows could not be verified');
        // every row excluded: the headline is "not scored", never 0%
        expect(screen.getByTestId('mo-pass')).toHaveTextContent('not scored');
        expect(screen.getByTestId('mo-pass')).not.toHaveTextContent('0.0%');
        expect(screen.getByTestId('mo-taxonomy')).toHaveTextContent('no taxonomy');
        expect(screen.getByTestId('mo-agreement')).toHaveTextContent('not reported');
        expect(screen.getByTestId('mo-provenance')).toHaveTextContent('no verifier');
        expect(screen.getByText(/at least one option passes/)).toBeInTheDocument();
    });

    it('snaps bar widths to 10% steps', () => {
        expect(widthStep(0.04)).toBe(0);
        expect(widthStep(0.75)).toBe(80);
        expect(widthStep(1)).toBe(100);
        expect(widthStep(null)).toBe(0);
    });

    it('an option with a missing format shows the empty fix and a failed presence check', () => {
        render(
            <OptionCard
                testId="card"
                option={{
                    rank: 1,
                    strategy: 'restrict',
                    in_taxonomy: true,
                    fixes: { format_a: 'fix', format_b: null },
                    formats: {
                        format_a: { checks: { parses: true }, passes: true, messages: [] },
                        format_b: { checks: { present: false }, passes: false, messages: ['no fix given in this format'] },
                    },
                    passes: false,
                }}
            />,
        );
        expect(screen.getByText('Option 2')).toBeInTheDocument();
        expect(screen.getByTestId('card-check-format_b-present')).toHaveTextContent('fail');
        expect(screen.getByTestId('card-check-format_a-present')).toHaveTextContent('n/a');
        expect(screen.getByTestId('card-messages')).toHaveTextContent('format_b: no fix given in this format');
        expect(screen.getByTestId('card-strategy')).not.toHaveClass('is-muted');
    });
});
