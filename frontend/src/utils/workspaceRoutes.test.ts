import { describe, expect, it } from 'vitest';

import { legacyRedirectPath, nextTab, resolveStep, stagePath } from './workspaceRoutes';

describe('workspaceRoutes', () => {
    it('builds the canonical stage URL, omitting step for the first step', () => {
        expect(stagePath(7, 'data')).toBe('/project/7/data');
        expect(stagePath(7, 'cleaning')).toBe('/project/7/data?step=cleaning');
        expect(stagePath(7, 'training')).toBe('/project/7/train?step=training');
        expect(stagePath(7, 'eval')).toBe('/project/7/results');
        expect(stagePath(7, 'export')).toBe('/project/7/ship?step=export');
        expect(stagePath(7, 'data', 'import=auto')).toBe('/project/7/data?import=auto');
    });

    it('resolves a step inside its destination only', () => {
        expect(resolveStep('data', 'goldset')).toBe('goldset');
        expect(resolveStep('data', 'training')).toBe('data');
        expect(resolveStep('ship', null)).toBe('compression');
    });

    it('walks the pipeline order across destinations', () => {
        expect(nextTab('dataprep')).toBe('tokenization');
        expect(nextTab('training')).toBe('eval');
        expect(nextTab('export')).toBeNull();
    });

    it('redirects retired URLs and keeps search + hash', () => {
        expect(legacyRedirectPath(7, '/project/7/guide')).toBe('/project/7');
        expect(legacyRedirectPath(7, '/project/7/wizard')).toBe('/project/7');
        expect(legacyRedirectPath(7, '/project/7/pipeline')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/pipeline/data', '?import=auto')).toBe('/project/7/data?import=auto');
        expect(legacyRedirectPath(7, '/project/7/pipeline/synthetic', '?focus_synth_source=x', '#synth-review-queue'))
            .toBe('/project/7/data?focus_synth_source=x&step=synthetic#synth-review-queue');
        expect(legacyRedirectPath(7, '/project/7/pipeline/training')).toBe('/project/7/train?step=training');
        expect(legacyRedirectPath(7, '/project/7/pipeline/nope')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/playground')).toBeNull();
    });
});
