/**
 * AnnotationProgress contract.
 *
 * Pins:
 * - Renders job name + label_type.
 * - Reports "labeled / denom" with target_rows as the denominator
 *   when set, falling back to total otherwise.
 * - Progress bar width tracks labeled / denom and is capped at 100%.
 * - Surfaces in-flight assigned count when > 0.
 */

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import AnnotationProgress from './AnnotationProgress';
import type { JobStats } from '../../api/annotation';

function statsFixture(overrides: Partial<JobStats> = {}): JobStats {
    return {
        job_id: 1,
        name: 'fixture',
        label_type: 'classification',
        status: 'active',
        target_rows: 50,
        total: 30,
        labeled: 12,
        assigned: 1,
        unlabeled: 17,
        ...overrides,
    };
}

describe('AnnotationProgress', () => {
    it('renders the job name + counts against target_rows', () => {
        render(
            <AnnotationProgress jobName="Sentiment v1" stats={statsFixture()} />,
        );
        expect(screen.getByText('Sentiment v1')).toBeInTheDocument();
        expect(screen.getByText('classification')).toBeInTheDocument();
        expect(
            screen.getByTestId('annotation-progress-labeled'),
        ).toHaveTextContent('12');
        expect(
            screen.getByTestId('annotation-progress-denom'),
        ).toHaveTextContent('50');
    });

    it('falls back to total when target_rows is null', () => {
        render(
            <AnnotationProgress
                jobName="No target"
                stats={statsFixture({ target_rows: null })}
            />,
        );
        expect(
            screen.getByTestId('annotation-progress-denom'),
        ).toHaveTextContent('30');
    });

    it('renders the progress bar at labeled / denom (capped at 100%)', () => {
        render(
            <AnnotationProgress
                jobName="Cap me"
                stats={statsFixture({ labeled: 200, target_rows: 50 })}
            />,
        );
        const bar = screen.getByTestId(
            'annotation-progress-bar',
        ) as HTMLDivElement;
        expect(bar.style.width).toBe('100%');
    });

    it('surfaces in-flight count when assigned > 0', () => {
        render(
            <AnnotationProgress
                jobName="With in-flight"
                stats={statsFixture({ assigned: 3 })}
            />,
        );
        expect(
            screen.getByTestId('annotation-progress-assigned'),
        ).toHaveTextContent('3 in flight');
    });

    it('hides in-flight count when assigned is 0', () => {
        render(
            <AnnotationProgress
                jobName="Quiet"
                stats={statsFixture({ assigned: 0 })}
            />,
        );
        expect(
            screen.queryByTestId('annotation-progress-assigned'),
        ).toBeNull();
    });
});
