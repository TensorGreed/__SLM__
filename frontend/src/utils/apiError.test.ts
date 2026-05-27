import { describe, it, expect } from 'vitest';

import { extractApiErrorMessage, parseApiErrorDetail } from './apiError';


describe('parseApiErrorDetail', () => {
    it('returns code + message for structured FastAPI detail', () => {
        const err = {
            response: {
                data: {
                    detail: {
                        error_code: 'EMPTY_GOLD_ROW',
                        message: 'Gold row has no recipe-shaped content.',
                    },
                },
            },
        };
        expect(parseApiErrorDetail(err)).toEqual({
            code: 'EMPTY_GOLD_ROW',
            message: 'Gold row has no recipe-shaped content.',
        });
    });

    it('returns UPSTREAM_ERROR + detail for plain-string detail', () => {
        const err = {
            response: { data: { detail: 'Gold dataset is locked. Cannot add new entries.' } },
        };
        expect(parseApiErrorDetail(err)).toEqual({
            code: 'UPSTREAM_ERROR',
            message: 'Gold dataset is locked. Cannot add new entries.',
        });
    });

    it('returns null when no response body (network/timeout)', () => {
        // axios reports message: "Network Error" with NO response field
        // for connection-level failures.
        expect(parseApiErrorDetail({ message: 'Network Error' })).toBeNull();
        expect(parseApiErrorDetail(new Error('boom'))).toBeNull();
    });

    it('returns null for undefined / null input', () => {
        expect(parseApiErrorDetail(undefined)).toBeNull();
        expect(parseApiErrorDetail(null)).toBeNull();
    });

    it('backfills missing error_code with "UNKNOWN"', () => {
        const err = {
            response: { data: { detail: { message: 'something went wrong' } } },
        };
        expect(parseApiErrorDetail(err)).toEqual({
            code: 'UNKNOWN',
            message: 'something went wrong',
        });
    });
});


describe('extractApiErrorMessage', () => {
    it('prefers backend detail over the supplied fallback', () => {
        const err = {
            response: { data: { detail: 'Gold dataset is locked.' } },
        };
        expect(extractApiErrorMessage(err, 'should not see this')).toEqual({
            code: 'UPSTREAM_ERROR',
            message: 'Gold dataset is locked.',
        });
    });

    it('falls back to axios "message" when no response body', () => {
        // For HTTP rejections without a body axios sets message like
        // "Request failed with status code 500"; for network failures
        // it sets "Network Error". Both reach this fallback path.
        const err = { message: 'Network Error' };
        expect(extractApiErrorMessage(err, 'Failed to add row')).toEqual({
            code: 'UNKNOWN',
            message: 'Network Error',
        });
    });

    it('uses the supplied fallback when err has neither response nor message', () => {
        expect(extractApiErrorMessage({}, 'Failed to add row')).toEqual({
            code: 'UNKNOWN',
            message: 'Failed to add row',
        });
        expect(extractApiErrorMessage(null, 'Failed to add row')).toEqual({
            code: 'UNKNOWN',
            message: 'Failed to add row',
        });
    });
});
