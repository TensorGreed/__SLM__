import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import TrainAnywayButton from './TrainAnywayButton';

describe('TrainAnywayButton', () => {
    it('renders the neutral "Train" label when the verdict is likely_pass', () => {
        render(<TrainAnywayButton verdict="likely_pass" confidencePct={78} />);
        const btn = screen.getByTestId('train-anyway-button');
        expect(btn.textContent).toContain('Train');
        // No advisory hint at likely_pass.
        expect(screen.queryByTestId('train-anyway-hint')).not.toBeInTheDocument();
        expect(btn).toHaveAttribute('data-verdict', 'likely_pass');
    });

    it('renders the borderline label with the confidence pct baked in', () => {
        render(<TrainAnywayButton verdict="borderline" confidencePct={58} />);
        const btn = screen.getByTestId('train-anyway-button');
        expect(btn.textContent).toMatch(/Train.*~58%/);
        expect(btn).toHaveAttribute('data-verdict', 'borderline');
        expect(screen.getByTestId('train-anyway-hint').textContent).toMatch(/borderline/i);
    });

    it('renders the "Train anyway" label and warning hint when the verdict is likely_fail', () => {
        render(<TrainAnywayButton verdict="likely_fail" confidencePct={28} />);
        const btn = screen.getByTestId('train-anyway-button');
        expect(btn.textContent).toMatch(/Train anyway/);
        expect(btn).toHaveAttribute('data-verdict', 'likely_fail');
        expect(screen.getByTestId('train-anyway-hint').textContent).toMatch(/forecast suggests/i);
    });

    it('scrolls and focuses the anchor button when clicked', async () => {
        // Mount a fake "Create Experiment" anchor that the button
        // should target. Our default selector is the .training-create-shell
        // primary button.
        const anchor = document.createElement('button');
        anchor.className = 'btn btn-primary';
        const wrapper = document.createElement('div');
        wrapper.className = 'training-create-shell__actions';
        wrapper.appendChild(anchor);
        document.body.appendChild(wrapper);

        // jsdom doesn't implement scrollIntoView — install a stub before spying.
        anchor.scrollIntoView = vi.fn();
        const scrollSpy = anchor.scrollIntoView as ReturnType<typeof vi.fn>;
        const focusSpy = vi.spyOn(anchor, 'focus');

        render(<TrainAnywayButton verdict="likely_fail" confidencePct={28} />);
        const btn = screen.getByTestId('train-anyway-button');
        await userEvent.click(btn);

        expect(scrollSpy).toHaveBeenCalledTimes(1);
        expect(scrollSpy.mock.calls[0][0]).toMatchObject({ behavior: 'smooth' });
        // Focus happens after a setTimeout(300) — flush timers.
        await new Promise((r) => setTimeout(r, 350));
        expect(focusSpy).toHaveBeenCalledTimes(1);

        document.body.removeChild(wrapper);
    });

    it('is a no-op when the anchor element is missing from the DOM', async () => {
        // No anchor mounted — clicking should not throw.
        render(<TrainAnywayButton verdict="likely_pass" confidencePct={80} />);
        const btn = screen.getByTestId('train-anyway-button');
        await userEvent.click(btn);  // no error expected
        expect(btn).toBeInTheDocument();
    });
});
