import { describe, expect, it } from 'vitest';

import { legacyRedirectPath, nextTab, resolveStep, stagePath, trainSectionPath } from './workspaceRoutes';

describe('workspaceRoutes', () => {
    it('builds the canonical stage URL, omitting step for the first step', () => {
        expect(stagePath(7, 'data')).toBe('/project/7/data');
        expect(stagePath(7, 'cleaning')).toBe('/project/7/data?step=cleaning');
        expect(stagePath(7, 'training')).toBe('/project/7/train');
        expect(stagePath(7, 'tokenization')).toBe('/project/7/train?open=tokenization');
        expect(trainSectionPath(7, 'plan')).toBe('/project/7/train?open=plan');
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
        expect(nextTab('dataprep')).toBe('training');
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
        expect(legacyRedirectPath(7, '/project/7/pipeline/training')).toBe('/project/7/train');
        expect(legacyRedirectPath(7, '/project/7/pipeline/tokenization')).toBe('/project/7/train?open=tokenization');
        expect(legacyRedirectPath(7, '/project/7/pipeline/nope')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/playground')).toBeNull();
    });

    it('folds the retired training pages into Train sections (U2)', () => {
        expect(legacyRedirectPath(7, '/project/7/training-config')).toBe('/project/7/train?open=plan');
        expect(legacyRedirectPath(7, '/project/7/training-config', '', '#base-model')).toBe('/project/7/train?open=model');
        expect(legacyRedirectPath(7, '/project/7/training-config', '?expand_multi_seed=1&suggested_num_seeds=3'))
            .toBe('/project/7/train?expand_multi_seed=1&suggested_num_seeds=3&open=seeds');
        expect(legacyRedirectPath(7, '/project/7/training-config', '', '#training-config-gaps-panel')).toBe('/project/7/train?open=plan');
        expect(legacyRedirectPath(7, '/project/7/models')).toBe('/project/7/train?open=models');
        expect(legacyRedirectPath(7, '/project/7/recipe-picker', '?return_to=%2Fproject%2F7%2Fdata')).toBe('/project/7/train?open=task');
        expect(legacyRedirectPath(7, '/project/7/recipes')).toBe('/project/7/train?open=pipeline-presets');
        expect(legacyRedirectPath(7, '/project/7/adapter-studio')).toBe('/project/7/train?open=adapters');
    });
});
