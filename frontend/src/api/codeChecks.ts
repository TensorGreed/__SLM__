/**
 * Typed client for the code-check registry (syntax / parse / lint checks that
 * run on code, in-process or through the sandbox).
 *
 *   GET  /code-checks
 *   POST /code-checks/run
 */

import api from './client';

export type CodeCheckKind = 'builtin' | 'external' | 'detector';

export interface CodeCheckInfo {
    id: string;
    languages: string[];
    kind: CodeCheckKind;
    binary: string | null;
    /** The binary resolves on this host (always true for built-ins). */
    available: boolean;
    /** Allow-listed in CODE_CHECK_ALLOWED (always true for built-ins). */
    allowed: boolean;
}

export interface NetworkIsolationInfo {
    mode: 'auto' | 'off' | 'require' | string;
    available: boolean;
}

export interface CodeChecksResponse {
    checks: CodeCheckInfo[];
    network_isolation: NetworkIsolationInfo;
}

export interface CodeCheckResult {
    check_id: string;
    ok: boolean | null;
    messages: string[];
    /** unavailable | not_allowed | language_mismatch | timeout */
    skipped: string | null;
}

export interface CodeChecksRunResult {
    ok: boolean | null;
    results: CodeCheckResult[];
}

export async function listCodeChecks(): Promise<CodeChecksResponse> {
    const res = await api.get<CodeChecksResponse>('/code-checks');
    return res.data;
}

export async function runCodeChecks(body: {
    language: string;
    code: string;
    check_ids?: string[];
}): Promise<CodeChecksRunResult> {
    const res = await api.post<CodeChecksRunResult>('/code-checks/run', body);
    return res.data;
}

/** Why a check is not usable, or null when it is. */
export function checkStatus(check: Pick<CodeCheckInfo, 'available' | 'allowed'>): 'ready' | 'not_installed' | 'not_allowed' {
    if (!check.available) return 'not_installed';
    if (!check.allowed) return 'not_allowed';
    return 'ready';
}
