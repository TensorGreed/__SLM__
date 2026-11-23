/**
 * Wakes the jobs store's poll when an API call starts a Job, so a Job started from any page is watched to the end.
 *
 * The jobs store stops its 4 s ``/jobs/active`` poll when nothing is in flight and the bell is closed. Only a few
 * call sites used to restart it, so a Job started elsewhere (Train > Launch) ran unwatched and the pages that react to
 * a finished Job (Results) never heard of it. The axios client calls ``noticeResponse`` on every successful response;
 * the jobs store registers its wake-up with ``setJobStartListener``. Kept apart from ``client.ts`` (which tests mock
 * wholesale) and from the store (which imports the client), so neither imports the other.
 */

/** The parts of an axios response this module reads. */
export interface ResponseLike {
    status: number;
    data?: unknown;
    config?: { method?: string; url?: string };
}

/** Endpoints that start a Job server-side but answer with something else (not 202, not a Job body). */
const SERVER_SIDE_JOB_STARTERS: RegExp[] = [
    // Train > Launch (and the kill switch's clone + start): the backend spawns a ``training_start`` watcher Job and
    // answers 200 with the experiment's task id.
    /\/training\/experiments\/\d+\/start$/,
];

function looksLikeJob(value: unknown): boolean {
    if (!value || typeof value !== 'object') return false;
    const v = value as Record<string, unknown>;
    return typeof v.id === 'number' && typeof v.kind === 'string' && typeof v.status === 'string' && 'progress' in v;
}

/** True when this response says a Job was (or may have been) started: a write that answered 202 (every
 *  ``?async_job=true`` endpoint and the other Job starters), a write whose body is a Job (or carries one under
 *  ``job``), or a known endpoint that starts a Job server-side. Reads never count. */
export function responseStartsJob(response: ResponseLike): boolean {
    const method = String(response.config?.method || 'get').toLowerCase();
    if (method === 'get' || method === 'head' || method === 'options') return false;
    if (response.status === 202) return true;
    const data = response.data as Record<string, unknown> | null | undefined;
    if (looksLikeJob(data) || looksLikeJob(data?.job)) return true;
    const url = String(response.config?.url || '').split('?')[0];
    return SERVER_SIDE_JOB_STARTERS.some((re) => re.test(url));
}

let listener: (() => void) | null = null;

/** The jobs store registers its wake-up here (one listener; a later call replaces it). */
export function setJobStartListener(fn: (() => void) | null): void {
    listener = fn;
}

/** Called by the axios client on every successful response. Never throws into the caller's request. */
export function noticeResponse(response: ResponseLike): void {
    if (!listener) return;
    try {
        if (responseStartsJob(response)) listener();
    } catch {
        // A failed wake-up must not fail the request that started the Job.
    }
}
