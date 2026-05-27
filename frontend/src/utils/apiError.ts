/**
 * Shared extractor for FastAPI-style structured error responses.
 *
 * The backend convention (see backend/app/api/*.py) is:
 *   * Caller-fixable problems → 400 with `{detail: {error_code, message}}`
 *   * Upstream / 502 relay errors → 400/502 with `{detail: "plain string"}`
 *   * Connection-level failures → no response body, axios reports
 *     `message: "Network Error"` or `code: "ECONNABORTED"`
 *
 * This helper handles the first two shapes uniformly. The third
 * (network/timeout) is caller-specific because the useful copy
 * depends on context (LLM-gen "tokens may be billed" vs add-form
 * "couldn't reach the backend") — callers pass through their own
 * fallback string for the no-detail case.
 */

export interface ApiErrorInfo {
    /** Structured error code from the backend, or a synthetic tag. */
    code: string;
    /** Human-readable message from the backend's ``detail.message``,
     *  or the raw string detail, or the caller-supplied fallback. */
    message: string;
}


/** Extract a {code, message} pair from an axios rejection.
 *
 *  Returns null when the rejection has no response body — caller
 *  decides how to surface network/timeout failures with context-
 *  appropriate copy. */
export function parseApiErrorDetail(err: unknown): ApiErrorInfo | null {
    const detail = (
        err as { response?: { data?: { detail?: unknown } } }
    )?.response?.data?.detail;
    if (detail && typeof detail === 'object') {
        const d = detail as { error_code?: unknown; message?: unknown };
        return {
            code: String(d.error_code || 'UNKNOWN'),
            message: String(d.message || 'Request failed'),
        };
    }
    if (typeof detail === 'string') {
        return { code: 'UPSTREAM_ERROR', message: detail };
    }
    return null;
}


/** Convenience for callers that want a guaranteed-non-null result —
 *  surfaces the backend's structured detail when present, otherwise
 *  falls back to the axios ``message`` string (e.g. "Network Error",
 *  "Request failed with status code 400") or the caller's literal
 *  fallback string. */
export function extractApiErrorMessage(
    err: unknown,
    fallback: string,
): ApiErrorInfo {
    const parsed = parseApiErrorDetail(err);
    if (parsed) return parsed;
    const rawMessage = (err as { message?: string })?.message;
    return {
        code: 'UNKNOWN',
        message: rawMessage || fallback,
    };
}
