import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ExperimentClassifierHeadBadge from './ExperimentClassifierHeadBadge';


describe('ExperimentClassifierHeadBadge (δ surface)', () => {
    it('renders the badge when task_type is classification and run completed', () => {
        render(
            <ExperimentClassifierHeadBadge
                taskType="classification"
                status="completed"
            />,
        );
        const badge = screen.getByTestId(
            'experiment-classifier-head-badge',
        );
        expect(badge).toHaveTextContent('classifier head');
        // Tooltip is the load-bearing affordance — without it the
        // user has no path from "huh, what's this badge" to
        // understanding which inference route their eval takes.
        const title = badge.getAttribute('title') || '';
        expect(title).toContain('classifier head');
        expect(title).toContain('AutoModelForSequenceClassification');
        expect(title).toContain('head=sequence_classification');
    });

    it('renders nothing when task_type is not classification', () => {
        // The most important regression guard: a normal SFT /
        // instruction / qa experiment must not show this badge,
        // because those don't ship a classifier head. Mis-labeling
        // them would mislead a user into thinking δ will route
        // through a head that doesn't exist.
        const { container } = render(
            <ExperimentClassifierHeadBadge
                taskType="causal_lm"
                status="completed"
            />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for a still-running classification experiment', () => {
        // The badge means "a classifier head WAS trained" — a
        // pending / running job hasn't produced a saved adapter
        // yet, so showing the badge would be premature. Tested
        // explicitly so a future refactor doesn't accidentally
        // drop the status guard.
        const { container } = render(
            <ExperimentClassifierHeadBadge
                taskType="classification"
                status="running"
            />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for a failed classification experiment', () => {
        const { container } = render(
            <ExperimentClassifierHeadBadge
                taskType="classification"
                status="failed"
            />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('is case-insensitive on the task_type comparison', () => {
        // Config values can drift in casing across migrations
        // (CLASSIFICATION vs classification). Tolerate both.
        render(
            <ExperimentClassifierHeadBadge
                taskType="Classification"
                status="completed"
            />,
        );
        expect(
            screen.getByTestId('experiment-classifier-head-badge'),
        ).toBeInTheDocument();
    });

    it('handles missing taskType + status gracefully', () => {
        const { container } = render(
            <ExperimentClassifierHeadBadge />,
        );
        expect(container.firstChild).toBeNull();
    });
});
