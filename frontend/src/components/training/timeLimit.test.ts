import { describe, expect, it } from 'vitest';

import { hoursFromSeconds, timeLimitSecondsFromHours, timeLimitWarning } from './trainingPanelUtils';

describe('training time limit helpers', () => {
    it('hours text → seconds, blank → server limit, out of range → invalid', () => {
        expect(timeLimitSecondsFromHours('')).toBeNull();
        expect(timeLimitSecondsFromHours('  ')).toBeNull();
        expect(timeLimitSecondsFromHours('24')).toBe(86400);
        expect(timeLimitSecondsFromHours('1.5')).toBe(5400);
        expect(timeLimitSecondsFromHours('0.01')).toBe('invalid'); // 36 s
        expect(timeLimitSecondsFromHours('169')).toBe('invalid');
        expect(timeLimitSecondsFromHours('abc')).toBe('invalid');
    });

    it('saved seconds → hours text', () => {
        expect(hoursFromSeconds(86400)).toBe('24');
        expect(hoursFromSeconds(5400)).toBe('1.5');
        expect(hoursFromSeconds(undefined)).toBe('');
        expect(hoursFromSeconds(null)).toBe('');
    });

    it('reads the in-run warning from the runtime block', () => {
        expect(timeLimitWarning({ _runtime: { time_limit_warning: { message: 'On pace to need about 20 h' } } })).toBe(
            'On pace to need about 20 h',
        );
        expect(timeLimitWarning({ _runtime: {} })).toBeNull();
        expect(timeLimitWarning(undefined)).toBeNull();
    });
});
