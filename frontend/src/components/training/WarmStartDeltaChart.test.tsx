import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import WarmStartDeltaChart from './WarmStartDeltaChart';

const METRICS = [
    { step: 10, train_loss: 2.0 },
    { step: 20, train_loss: 1.7 },
    { step: 30, train_loss: 1.4 },
];

describe('WarmStartDeltaChart', () => {
    it('renders nothing for a cold-start run', () => {
        const { container } = render(
            <WarmStartDeltaChart metrics={METRICS} warmStart={{ source: 'base_model', reason: 'no_checkpoint_recommended' }} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing when warmStart is absent', () => {
        const { container } = render(<WarmStartDeltaChart metrics={METRICS} warmStart={null} />);
        expect(container.firstChild).toBeNull();
    });

    it('plots both the loss and delta series for a warm-started run', () => {
        render(
            <WarmStartDeltaChart
                metrics={METRICS}
                warmStart={{ source: 'checkpoint', checkpoint_name: 'qa-base-135m', reason: 'warm_start:qa-base-135m' }}
            />,
        );
        expect(screen.getByTestId('warmstart-delta')).toBeInTheDocument();
        expect(screen.getByTestId('warmstart-delta-loss')).toBeInTheDocument();
        expect(screen.getByTestId('warmstart-delta-delta')).toBeInTheDocument();
        // baseline = first loss (2.0); final delta = 1.4 - 2.0 = -0.6 → improvement.
        expect(screen.getByText(/reduced loss by 0\.600 below/i)).toBeInTheDocument();
        expect(screen.getByText(/baseline = 2\.000/i)).toBeInTheDocument();
    });

    it('flags a regression when loss rose above the baseline', () => {
        render(
            <WarmStartDeltaChart
                metrics={[{ step: 10, train_loss: 1.5 }, { step: 20, train_loss: 1.9 }]}
                warmStart={{ source: 'checkpoint', checkpoint_name: 'qa-base-135m' }}
            />,
        );
        expect(screen.getByText(/0\.400 above the warm-start's starting point/i)).toBeInTheDocument();
    });

    it('shows an empty hint when warm-started but metrics have not streamed yet', () => {
        render(
            <WarmStartDeltaChart metrics={[{ step: 10, train_loss: 2.0 }]} warmStart={{ source: 'checkpoint', checkpoint_name: 'qa-base-135m' }} />,
        );
        expect(screen.getByText(/delta-from-baseline curve appears once training metrics stream in/i)).toBeInTheDocument();
    });
});
