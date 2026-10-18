import { describe, expect, it } from 'vitest';

import { legacyRedirectPath, nextTab, resolveStep, resultsPath, stagePath, trainSectionPath } from './workspaceRoutes';

describe('workspaceRoutes', () => {
    it('builds the canonical stage URL, omitting step for the first step', () => {
        expect(stagePath(7, 'data')).toBe('/project/7/data');
        expect(stagePath(7, 'cleaning')).toBe('/project/7/data?step=cleaning');
        expect(stagePath(7, 'training')).toBe('/project/7/train');
        expect(stagePath(7, 'tokenization')).toBe('/project/7/train?open=tokenization');
        expect(trainSectionPath(7, 'plan')).toBe('/project/7/train?open=plan');
        expect(stagePath(7, 'eval')).toBe('/project/7/results');
        expect(stagePath(7, 'export')).toBe('/project/7/ship');
        expect(stagePath(7, 'compression')).toBe('/project/7/ship?open=compress');
        expect(stagePath(7, 'data', 'import=auto')).toBe('/project/7/data?import=auto');
    });

    it('resolves a step inside its destination only', () => {
        expect(resolveStep('data', 'goldset')).toBe('goldset');
        expect(resolveStep('data', 'training')).toBe('data');
        expect(resolveStep('ship', null)).toBe('export');
    });

    it('walks the pipeline order across destinations', () => {
        expect(nextTab('dataprep')).toBe('training');
        expect(nextTab('training')).toBe('eval');
        expect(nextTab('eval')).toBe('export');
        expect(nextTab('export')).toBeNull();
    });

    it('redirects retired URLs and keeps search + hash', () => {
        expect(legacyRedirectPath(7, '/project/7/guide')).toBe('/project/7');
        expect(legacyRedirectPath(7, '/project/7/wizard')).toBe('/project/7');
        expect(legacyRedirectPath(7, '/project/7/pipeline')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/pipeline/data', '?import=auto')).toBe('/project/7/data?import=auto');
        expect(legacyRedirectPath(7, '/project/7/pipeline/synthetic', '?focus_synth_source=x', '#synth-review-queue'))
            .toBe('/project/7/data?focus_synth_source=x&open=review-queue&step=synthetic#synth-review-queue');
        expect(legacyRedirectPath(7, '/project/7/pipeline/cleaning', '', '#label-noise-review'))
            .toBe('/project/7/data?open=label-noise&step=cleaning#label-noise-review');
        expect(legacyRedirectPath(7, '/project/7/pipeline/training')).toBe('/project/7/train');
        expect(legacyRedirectPath(7, '/project/7/pipeline/tokenization')).toBe('/project/7/train?open=tokenization');
        expect(legacyRedirectPath(7, '/project/7/pipeline/nope')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/extensions')).toBeNull();
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

    it('folds eval anchors, eval/compare and the playground into Results (U3)', () => {
        expect(resultsPath(7, 'failures')).toBe('/project/7/results?tab=failures');
        expect(legacyRedirectPath(7, '/project/7/pipeline/eval')).toBe('/project/7/results');
        expect(legacyRedirectPath(7, '/project/7/pipeline/eval', '', '#probe-pack-panel')).toBe('/project/7/results?tab=failures#probe-pack-panel');
        expect(legacyRedirectPath(7, '/project/7/pipeline/eval', '', '#external-benchmarks')).toBe('/project/7/results?tab=compare#external-benchmarks');
        expect(legacyRedirectPath(7, '/project/7/eval/compare', '?a=1&b=2')).toBe('/project/7/results?a=1&b=2&tab=compare');
        expect(legacyRedirectPath(7, '/project/7/playground', '?run=5')).toBe('/project/7/results?run=5&tab=try');
    });

    it('folds Data Studio sections and the annotation list into the Data steps (U4)', () => {
        expect(legacyRedirectPath(7, '/project/7/data-studio')).toBe('/project/7/data');
        expect(legacyRedirectPath(7, '/project/7/data-studio', '', '#dataset-versions')).toBe('/project/7/data?open=versions&step=dataprep');
        expect(legacyRedirectPath(7, '/project/7/data-studio', '', '#review_queue')).toBe('/project/7/data?open=review-queue&step=goldset');
        expect(legacyRedirectPath(7, '/project/7/annotate')).toBe('/project/7/data?open=annotation&step=goldset');
    });

    it('folds deployments, observability and the manifest into Ship (U5)', () => {
        expect(legacyRedirectPath(7, '/project/7/deployments')).toBe('/project/7/ship?stage=deploy');
        expect(legacyRedirectPath(7, '/project/7/observability', '', '#failure-clusters')).toBe('/project/7/ship?stage=monitor#failure-clusters');
        expect(legacyRedirectPath(7, '/project/7/manifest')).toBe('/project/7/ship?open=manifest&stage=package');
        expect(legacyRedirectPath(7, '/project/7/pipeline/compression')).toBe('/project/7/ship?open=compress');
        expect(legacyRedirectPath(7, '/project/7/pipeline/export')).toBe('/project/7/ship');
    });
});
