/**
 * Shared error parsing + types for the platform's structured error
 * envelope (Diagnostics Intervention A).
 *
 * Every API endpoint now returns the same shape on 4xx/5xx:
 *   {
 *     error_code: "SYNTHETIC_NOT_FOUND",
 *     stage: "synthetic",
 *     message: "Project 17 not found.",
 *     actionable_fix: "Verify the project/resource identifier and try again.",
 *     docs_url: "/docs/troubleshooting",
 *     metadata: { ... } | null,
 *     troubleshooting_id: "err_a3f8d4e2",
 *     detail: "Project 17 not found."          // legacy compat
 *   }
 *
 * Plus, for completeness:
 *   * 422 validation errors carry ``metadata.validation_errors``
 *   * Last-resort unhandled exceptions carry ``metadata.exception_type``
 *
 * Frontend code should:
 *   1. Catch axios errors → call ``parseErrorEnvelope(err)`` to get a
 *      normalized object regardless of whether the response was the
 *      new envelope, a legacy ``{detail: ...}``, or a network error.
 *   2. Pass that object to the shared ``<ErrorPanel>`` component for
 *      consistent rendering.
 *
 * Goal of all this: when a user reports a bug, they paste the
 * troubleshooting_id from the panel, and a developer can grep server
 * logs for it to find the full traceback.
 */

export interface ErrorEnvelope {
    /** Stable upper-snake-case identifier, e.g. ``SYNTHETIC_LLM_REFUSAL``.
     *  Frontend dispatchers can branch on this to render
     *  affordances (e.g. show a 'Retry with Qwen' button for refusals). */
    errorCode: string;
    /** Loosely the API surface that produced the error
     *  ('synthetic' / 'training' / 'gold' / etc.). Useful for log
     *  filters + grouping in the upcoming admin dashboard. */
    stage: string;
    /** One-sentence human-readable summary. Goes in the panel headline. */
    message: string;
    /** Specific next-action remediation. Always populated — the
     *  backend's default fallbacks ('Review request inputs and retry')
     *  beat nothing. */
    actionableFix: string;
    /** Docs link surface for the user to drill into. */
    docsUrl: string;
    /** Short opaque token (e.g. ``err_a3f8d4e2``). The user copies
     *  this into a bug report; the developer greps logs for it. */
    troubleshootingId: string;
    /** Per-error context: raw LLM output, validation errors,
     *  exception type, etc. Frontend shows this collapsed by default. */
    metadata: Record<string, unknown> | null;
    /** HTTP status — useful for UI badge color (4xx vs 5xx). */
    statusCode: number;
    /** True when the parser had to fall back to a network-error or
     *  legacy-shape response (i.e. the envelope wasn't fully populated
     *  by the backend). Renders an extra hint in the panel. */
    isFallback: boolean;
}

interface AxiosLikeError {
    response?: {
        status?: number;
        data?: unknown;
    };
    message?: string;
}

/**
 * Normalize ANY thrown value into an ``ErrorEnvelope``.
 *
 * Handles:
 *   * The full envelope shape (preferred — most endpoints now)
 *   * Legacy ``{detail: "..."}`` responses from endpoints not yet
 *     migrated to the envelope
 *   * FastAPI 422 validation arrays (``{detail: [{loc, msg, ...}]}``)
 *   * Network errors / timeouts (no response object)
 *   * Plain ``Error`` instances or strings thrown elsewhere
 *
 * Never throws. Always returns a fully-populated envelope so the
 * caller can render without null checks.
 */
export function parseErrorEnvelope(err: unknown): ErrorEnvelope {
    // Network / unknown errors — no response object at all.
    if (typeof err !== 'object' || err === null) {
        return makeFallback({
            message: typeof err === 'string' ? err : 'An unknown error occurred.',
            statusCode: 0,
        });
    }
    const axiosErr = err as AxiosLikeError;
    const status = axiosErr.response?.status ?? 0;
    const data = axiosErr.response?.data;

    // No response body — probably a network error.
    if (!data || typeof data !== 'object') {
        return makeFallback({
            message: axiosErr.message || 'Request failed.',
            statusCode: status,
        });
    }

    const obj = data as Record<string, unknown>;
    // Detect the new envelope by ``troubleshooting_id`` (more specific
    // than ``error_code`` which a few legacy endpoints also set).
    if (typeof obj.troubleshooting_id === 'string') {
        return {
            errorCode: stringOr(obj.error_code, 'UNKNOWN'),
            stage: stringOr(obj.stage, 'general'),
            message: stringOr(obj.message ?? obj.detail, 'Request failed.'),
            actionableFix: stringOr(obj.actionable_fix, ''),
            docsUrl: stringOr(obj.docs_url, ''),
            troubleshootingId: stringOr(obj.troubleshooting_id, ''),
            metadata: isPlainObject(obj.metadata) ? obj.metadata : null,
            statusCode: status,
            isFallback: false,
        };
    }

    // Legacy ``{detail: "..."}`` — most endpoints that haven't been
    // migrated yet. Or 422 with array-of-validation-errors.
    const detailValue = obj.detail;
    if (Array.isArray(detailValue)) {
        // FastAPI validation errors. Surface the first message + stash
        // the full list under metadata.
        const first = detailValue[0] as Record<string, unknown> | undefined;
        const firstMsg = first ? stringOr(first.msg, '') : '';
        return makeFallback({
            message: firstMsg || 'Request validation failed.',
            statusCode: status,
            metadata: { validation_errors: detailValue },
        });
    }
    return makeFallback({
        message: stringOr(detailValue, axiosErr.message || 'Request failed.'),
        statusCode: status,
    });
}


function makeFallback(opts: {
    message: string;
    statusCode: number;
    metadata?: Record<string, unknown> | null;
}): ErrorEnvelope {
    return {
        errorCode: defaultCodeFromStatus(opts.statusCode),
        stage: 'general',
        message: opts.message,
        actionableFix: defaultRemediationFromStatus(opts.statusCode),
        docsUrl: '/docs/troubleshooting',
        // Client-side trace id when the server didn't provide one. Lets
        // the panel still show *something* to copy-paste even on a
        // network error.
        troubleshootingId: `local_${randomToken()}`,
        metadata: opts.metadata ?? null,
        statusCode: opts.statusCode,
        isFallback: true,
    };
}


function defaultCodeFromStatus(status: number): string {
    if (status === 0) return 'NETWORK_ERROR';
    if (status === 401) return 'UNAUTHORIZED';
    if (status === 403) return 'FORBIDDEN';
    if (status === 404) return 'NOT_FOUND';
    if (status === 409) return 'CONFLICT';
    if (status === 422) return 'VALIDATION_ERROR';
    if (status >= 500) return 'SERVER_ERROR';
    if (status >= 400) return 'REQUEST_FAILED';
    return 'UNKNOWN';
}


function defaultRemediationFromStatus(status: number): string {
    if (status === 0) return 'Check your connection and retry. The server may be down.';
    if (status === 401) return 'Sign in again and retry.';
    if (status === 403) return "You don't have permission for this action. Contact the project owner.";
    if (status === 404) return 'Verify the resource identifier and retry.';
    if (status === 422) return 'Review the request inputs and retry.';
    if (status >= 500) return 'The server hit an unexpected error. Retry; if it persists, copy the troubleshooting id and report it.';
    return 'Retry the action.';
}


function stringOr(value: unknown, fallback: string): string {
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    return fallback;
}


function isPlainObject(value: unknown): value is Record<string, unknown> {
    return (
        typeof value === 'object'
        && value !== null
        && !Array.isArray(value)
    );
}


function randomToken(): string {
    // 9 chars from the url-safe alphabet. Crypto-grade isn't required
    // here — collisions in the client-side fallback bucket are
    // harmless (we just want a copy-pasteable token).
    const alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_';
    let out = '';
    for (let i = 0; i < 9; i++) {
        out += alphabet[Math.floor(Math.random() * alphabet.length)];
    }
    return out;
}
