import { describe, it, expect } from 'vitest';

import { buildClusterFocusHint, routeClusterFix } from './clusterFixRouter';


describe('buildClusterFocusHint', () => {
    it('prefers the classifier_reason when present', () => {
        const hint = buildClusterFocusHint({
            cluster_id: 'c1',
            reason_code: 'hallucination',
            classifier_reason: 'Model is fabricating dates.',
            output_pattern: 'len-medium:lead-prose',
        });
        // reason_code is bracket-prefixed so the LLM sees the
        // structured tag alongside the narrative.
        expect(hint).toBe('[hallucination] Model is fabricating dates.');
    });

    it('falls back to output_pattern when classifier_reason is empty', () => {
        const hint = buildClusterFocusHint({
            cluster_id: 'c2',
            reason_code: 'coverage_gap',
            classifier_reason: '   ',
            output_pattern: 'len-short:lead-refusal',
        });
        expect(hint).toBe('[coverage_gap] len-short:lead-refusal');
    });

    it('falls back to the first exemplar prompt when other fields are absent', () => {
        const hint = buildClusterFocusHint({
            cluster_id: 'c3',
            reason_code: 'safety_failure',
            exemplars: [{ prompt: 'What is my SSN?', reference: 'I cannot share.', prediction: '...' }],
        });
        expect(hint).toBe('[safety_failure] What is my SSN?');
    });

    it('truncates pathologically long hints with an ellipsis', () => {
        const longReason = 'X '.repeat(300);
        const hint = buildClusterFocusHint({
            cluster_id: 'c4',
            classifier_reason: longReason,
        });
        // Hint must end with the ellipsis marker + stay under the cap.
        expect(hint.endsWith('…')).toBe(true);
        expect(hint.length).toBeLessThanOrEqual(240);
    });

    it('returns empty string when the cluster has no usable fields', () => {
        const hint = buildClusterFocusHint({ cluster_id: 'empty' });
        expect(hint).toBe('');
    });
});


describe('routeClusterFix', () => {
    it('returns the gold-set path with focus_cluster_id + focus_hint + trap_count', () => {
        const route = routeClusterFix(7, {
            cluster_id: 'cluster-42',
            reason_code: 'hallucination',
            classifier_reason: 'Model is fabricating dates.',
        });
        expect(route.path).toBe('/project/7/pipeline/goldset');
        const params = new URLSearchParams(route.search);
        expect(params.get('focus_cluster_id')).toBe('cluster-42');
        expect(params.get('focus_hint')).toBe('[hallucination] Model is fabricating dates.');
        expect(params.get('trap_count')).toBe('5');
    });

    it('honors an explicit trapCount override', () => {
        const route = routeClusterFix(7, { cluster_id: 'c' }, { trapCount: 10 });
        expect(new URLSearchParams(route.search).get('trap_count')).toBe('10');
    });

    it('clamps the trap count to [1, 20]', () => {
        const tooMany = routeClusterFix(1, { cluster_id: 'c' }, { trapCount: 999 });
        expect(new URLSearchParams(tooMany.search).get('trap_count')).toBe('20');
        const tooFew = routeClusterFix(1, { cluster_id: 'c' }, { trapCount: -5 });
        expect(new URLSearchParams(tooFew.search).get('trap_count')).toBe('1');
    });

    it('omits focus_hint when the cluster yields no usable hint', () => {
        const route = routeClusterFix(7, { cluster_id: 'empty' });
        const params = new URLSearchParams(route.search);
        expect(params.get('focus_cluster_id')).toBe('empty');
        expect(params.get('focus_hint')).toBeNull();
        // trap_count still applies even without a hint.
        expect(params.get('trap_count')).toBe('5');
    });
});
