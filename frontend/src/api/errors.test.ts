import { describe, expect, it } from 'vitest';

import { parseErrorEnvelope } from './errors';


describe('parseErrorEnvelope', () => {
    it('parses the full envelope shape into camelCase fields', () => {
        const err = {
            response: {
                status: 400,
                data: {
                    error_code: 'SYNTHETIC_LLM_REFUSAL',
                    stage: 'synthetic',
                    message: 'Model refused on guardrails.',
                    actionable_fix: 'Switch to Qwen 2.5.',
                    docs_url: '/docs/troubleshooting',
                    troubleshooting_id: 'err_a3f8d4e2',
                    metadata: { raw_llm_snippet: 'I cannot...' },
                    detail: 'Model refused on guardrails.',
                },
            },
        };
        const env = parseErrorEnvelope(err);
        expect(env.errorCode).toBe('SYNTHETIC_LLM_REFUSAL');
        expect(env.stage).toBe('synthetic');
        expect(env.message).toBe('Model refused on guardrails.');
        expect(env.actionableFix).toBe('Switch to Qwen 2.5.');
        expect(env.troubleshootingId).toBe('err_a3f8d4e2');
        expect(env.statusCode).toBe(400);
        expect(env.metadata).toEqual({ raw_llm_snippet: 'I cannot...' });
        expect(env.isFallback).toBe(false);
    });

    it('falls back gracefully on a legacy {detail: "..."} response', () => {
        const err = {
            response: {
                status: 404,
                data: { detail: 'Project not found' },
            },
        };
        const env = parseErrorEnvelope(err);
        expect(env.message).toBe('Project not found');
        expect(env.statusCode).toBe(404);
        expect(env.errorCode).toBe('NOT_FOUND');
        expect(env.actionableFix).toMatch(/identifier/);
        // Fallback envelopes mint a client-side trace id so the panel
        // still has something to display.
        expect(env.troubleshootingId).toMatch(/^local_/);
        expect(env.isFallback).toBe(true);
    });

    it('handles FastAPI validation arrays (legacy {detail: [{msg}]} shape)', () => {
        const err = {
            response: {
                status: 422,
                data: {
                    detail: [
                        { loc: ['body', 'target_count'], msg: 'must be an integer', type: 'type_error' },
                    ],
                },
            },
        };
        const env = parseErrorEnvelope(err);
        expect(env.statusCode).toBe(422);
        expect(env.message).toBe('must be an integer');
        expect(env.errorCode).toBe('VALIDATION_ERROR');
        // The full validation list is preserved under metadata so a
        // power user can drill in.
        expect(env.metadata).toEqual({
            validation_errors: [
                { loc: ['body', 'target_count'], msg: 'must be an integer', type: 'type_error' },
            ],
        });
    });

    it('handles network errors (no response object)', () => {
        const err = { message: 'Network Error' };
        const env = parseErrorEnvelope(err);
        expect(env.statusCode).toBe(0);
        expect(env.errorCode).toBe('NETWORK_ERROR');
        expect(env.message).toBe('Network Error');
        expect(env.isFallback).toBe(true);
    });

    it('handles strings thrown directly', () => {
        const env = parseErrorEnvelope('something broke');
        expect(env.message).toBe('something broke');
        expect(env.isFallback).toBe(true);
    });

    it('handles undefined / null without throwing', () => {
        expect(parseErrorEnvelope(null).isFallback).toBe(true);
        expect(parseErrorEnvelope(undefined).isFallback).toBe(true);
    });

    it('detects the envelope by troubleshooting_id, not error_code', () => {
        // An older endpoint might set error_code but not be envelope-
        // wrapped (no troubleshooting_id). The parser must treat it
        // as legacy + add a client-side trace id.
        const err = {
            response: {
                status: 500,
                data: { error_code: 'LEGACY_CODE', detail: 'kaboom' },
            },
        };
        const env = parseErrorEnvelope(err);
        expect(env.isFallback).toBe(true);
        expect(env.troubleshootingId).toMatch(/^local_/);
    });

    it('provides status-tailored remediation for common HTTP codes', () => {
        expect(
            parseErrorEnvelope({ response: { status: 401, data: { detail: 'no' } } }).actionableFix,
        ).toMatch(/Sign in/);
        expect(
            parseErrorEnvelope({ response: { status: 403, data: { detail: 'no' } } }).actionableFix,
        ).toMatch(/permission/);
        expect(
            parseErrorEnvelope({ response: { status: 500, data: { detail: 'oops' } } }).actionableFix,
        ).toMatch(/troubleshooting id/);
    });
});
