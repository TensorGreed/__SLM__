import { describe, it, expect } from 'vitest';

import {
    decodeFragmentGroups,
    decodeRowIds,
    routeForecastAction,
} from './forecastActionRouter';


describe('routeForecastAction', () => {
    describe('synth_augment', () => {
        it('routes to /pipeline/synthetic with positives_paraphrase as the safe default', () => {
            const route = routeForecastAction(1, 'synth_augment', { target_rows: 100 });
            expect(route).not.toBeNull();
            expect(route!.path).toBe('/project/1/pipeline/synthetic');
            const params = new URLSearchParams(route!.search);
            expect(params.get('prefill_mode')).toBe('positives_paraphrase');
            expect(params.get('prefill_count')).toBe('100');
        });

        it('maps diversify="entity_types" hint to cluster_targeted (span-extraction T3 signal)', () => {
            const route = routeForecastAction(1, 'synth_augment', {
                target_rows: 50,
                diversify: 'entity_types',
            });
            expect(new URLSearchParams(route!.search).get('prefill_mode')).toBe('cluster_targeted');
        });

        it('maps diversify="negative_examples" hint to hard_negatives', () => {
            const route = routeForecastAction(1, 'synth_augment', {
                target_rows: 10,
                diversify: 'negative_examples',
            });
            expect(new URLSearchParams(route!.search).get('prefill_mode')).toBe('hard_negatives');
        });

        it('falls back to positives_paraphrase for unknown diversify hints', () => {
            const route = routeForecastAction(1, 'synth_augment', {
                target_rows: 30,
                diversify: 'something_new_in_v2',
            });
            expect(new URLSearchParams(route!.search).get('prefill_mode')).toBe('positives_paraphrase');
        });

        it('falls back to count=50 when target_rows is missing or invalid', () => {
            expect(new URLSearchParams(routeForecastAction(1, 'synth_augment', {})!.search).get('prefill_count')).toBe('50');
            expect(new URLSearchParams(routeForecastAction(1, 'synth_augment', { target_rows: -1 })!.search).get('prefill_count')).toBe('50');
        });
    });

    describe('synth_balance', () => {
        it('routes to class_balance_fill with target_rows = per_class * under_count (≥30)', () => {
            // 3 under-classes × 10 rows each = 30 → matches the floor.
            const route = routeForecastAction(2, 'synth_balance', {
                underrepresented_classes: ['a', 'b', 'c'],
                target_rows_per_class: 10,
            });
            expect(route!.path).toBe('/project/2/pipeline/synthetic');
            const params = new URLSearchParams(route!.search);
            expect(params.get('prefill_mode')).toBe('class_balance_fill');
            expect(params.get('prefill_count')).toBe('30');
        });

        it('scales target_rows up when per_class * under_count exceeds the 30 floor', () => {
            const route = routeForecastAction(2, 'synth_balance', {
                underrepresented_classes: ['a', 'b'],
                target_rows_per_class: 25,
            });
            // 2 × 25 = 50 → above the 30 floor.
            expect(new URLSearchParams(route!.search).get('prefill_count')).toBe('50');
        });

        it('falls back to count=30 when per_class is missing', () => {
            const route = routeForecastAction(2, 'synth_balance', {
                underrepresented_classes: ['a', 'b'],
            });
            expect(new URLSearchParams(route!.search).get('prefill_count')).toBe('30');
        });
    });

    describe('synth_diversify', () => {
        it('routes to positives_paraphrase with the target_rows count', () => {
            const route = routeForecastAction(3, 'synth_diversify', { target_rows: 75 });
            const params = new URLSearchParams(route!.search);
            expect(route!.path).toBe('/project/3/pipeline/synthetic');
            expect(params.get('prefill_mode')).toBe('positives_paraphrase');
            expect(params.get('prefill_count')).toBe('75');
        });
    });

    describe('fix_gold_rows', () => {
        it('encodes invalid_row_ids as a comma-separated fix_rows query param', () => {
            const route = routeForecastAction(7, 'fix_gold_rows', {
                invalid_row_ids: [1, 7, 12],
            });
            expect(route!.path).toBe('/project/7/pipeline/goldset');
            expect(new URLSearchParams(route!.search).get('fix_rows')).toBe('1,7,12');
        });

        it('encodes fragment_groups as pipe-within-group, semicolon-between-groups', () => {
            const route = routeForecastAction(7, 'fix_gold_rows', {
                fragment_groups: [['positive', 'Positive'], ['billing', 'Billing']],
            });
            const decoded = decodeURIComponent(
                new URLSearchParams(route!.search).get('fragment_groups') || '',
            );
            // URL encoding preserves the | and ; separators (they're
            // not reserved in the query portion). The labels are
            // URI-encoded so a "value with spaces" still round-trips.
            expect(decoded).toBe('positive|Positive;billing|Billing');
        });

        it('handles both row ids and fragment groups in the same action', () => {
            const route = routeForecastAction(7, 'fix_gold_rows', {
                invalid_row_ids: [1, 2],
                fragment_groups: [['a', 'A']],
            });
            const params = new URLSearchParams(route!.search);
            expect(params.get('fix_rows')).toBe('1,2');
            expect(params.get('fragment_groups')).toBeTruthy();
        });

        it('drops single-variant "groups" (a group needs >= 2 variants to be a fragment)', () => {
            const route = routeForecastAction(7, 'fix_gold_rows', {
                fragment_groups: [['lonely'], ['a', 'A']],
            });
            const params = new URLSearchParams(route!.search);
            const decoded = decodeURIComponent(params.get('fragment_groups') || '');
            expect(decoded).toBe('a|A');
        });

        it('emits the goldset path even when no params are supplied (signal carries no row ids)', () => {
            const route = routeForecastAction(7, 'fix_gold_rows', {});
            expect(route!.path).toBe('/project/7/pipeline/goldset');
            expect(route!.search).toBe('');
        });
    });

    it('returns null for an unrecognised kind', () => {
        // Keeps the caller from navigating somewhere wrong if a new
        // backend action lands before the frontend learns about it.
        const route = routeForecastAction(1, 'experimental_kind' as any, {});
        expect(route).toBeNull();
    });
});


describe('decodeRowIds', () => {
    it('parses a comma-separated list of row ids', () => {
        expect(decodeRowIds('1,7,12')).toEqual([1, 7, 12]);
    });
    it('drops malformed tokens but keeps the well-formed ones', () => {
        expect(decodeRowIds('1, , bogus, 4')).toEqual([1, 4]);
    });
    it('returns [] on null / empty input', () => {
        expect(decodeRowIds(null)).toEqual([]);
        expect(decodeRowIds('')).toEqual([]);
    });
});


describe('decodeFragmentGroups', () => {
    it('round-trips a multi-group encoding', () => {
        expect(decodeFragmentGroups('positive|Positive;billing|Billing')).toEqual([
            ['positive', 'Positive'],
            ['billing', 'Billing'],
        ]);
    });
    it('URI-decodes the variants', () => {
        const encoded = `${encodeURIComponent('value with space')}|${encodeURIComponent('Value With Space')}`;
        expect(decodeFragmentGroups(encoded)).toEqual([['value with space', 'Value With Space']]);
    });
    it('drops single-variant groups (need >= 2 to be a fragment)', () => {
        expect(decodeFragmentGroups('lonely;a|A')).toEqual([['a', 'A']]);
    });
});
