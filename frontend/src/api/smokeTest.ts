/**
 * Typed client for the project smoke-test endpoint (Diagnostics
 * Intervention C). One button on the project page runs N parallel
 * read-only checks; this client + the matching component render
 * the result.
 */

import api from './client';
import type { ErrorEnvelope } from './errors';


export type SmokeStatus = 'ok' | 'warn' | 'fail' | 'skip';


/** One check's result. Mirrors ``SmokeCheckResult`` in the backend
 *  service. The frontend renders these in a checklist; failures drop
 *  the ``envelope`` field into the shared ``<ErrorPanel>``. */
export interface SmokeCheckResult {
    name: string;
    status: SmokeStatus;
    elapsedMs: number;
    message: string;
    remediation: string | null;
    /** When status=fail, an ErrorEnvelope-shape dict the frontend
     *  can render via the shared ``<ErrorPanel>``. The shape exactly
     *  matches the backend's structured error envelope so the same
     *  parser/component handles both. */
    envelope: ErrorEnvelope | null;
    /** Optional context (counts, sample row, etc.) the panel shows
     *  under "Technical details" — populated for ok/warn checks too. */
    metadata: Record<string, unknown>;
}


export interface SmokeTestSummary {
    projectId: number;
    overall: SmokeStatus;
    elapsedMs: number;
    counts: Record<SmokeStatus, number>;
    checks: SmokeCheckResult[];
}


/** Backend wire shape (snake_case → camelCase normalization happens
 *  in ``runSmokeTest``). */
interface WireSmokeCheck {
    name: string;
    status: SmokeStatus;
    elapsed_ms: number;
    message: string;
    remediation: string | null;
    envelope: unknown;
    metadata: Record<string, unknown>;
}

interface WireSmokeSummary {
    project_id: number;
    overall: SmokeStatus;
    elapsed_ms: number;
    counts: Record<SmokeStatus, number>;
    checks: WireSmokeCheck[];
}


function normalizeEnvelope(value: unknown): ErrorEnvelope | null {
    // Backend wire envelope uses snake_case; map to the
    // ErrorEnvelope's camelCase fields so the same <ErrorPanel> works.
    if (!value || typeof value !== 'object') return null;
    const obj = value as Record<string, unknown>;
    if (typeof obj.troubleshooting_id !== 'string') return null;
    return {
        errorCode: String(obj.error_code ?? 'UNKNOWN'),
        stage: String(obj.stage ?? 'general'),
        message: String(obj.message ?? obj.detail ?? 'Check failed.'),
        actionableFix: String(obj.actionable_fix ?? ''),
        docsUrl: String(obj.docs_url ?? ''),
        troubleshootingId: String(obj.troubleshooting_id ?? ''),
        metadata: (obj.metadata && typeof obj.metadata === 'object'
            && !Array.isArray(obj.metadata))
            ? obj.metadata as Record<string, unknown>
            : null,
        statusCode: 500,
        isFallback: false,
    };
}


export async function runSmokeTest(projectId: number): Promise<SmokeTestSummary> {
    const resp = await api.post<WireSmokeSummary>(
        `/projects/${projectId}/smoke-test`,
    );
    const wire = resp.data;
    return {
        projectId: wire.project_id,
        overall: wire.overall,
        elapsedMs: wire.elapsed_ms,
        counts: wire.counts,
        checks: wire.checks.map((c) => ({
            name: c.name,
            status: c.status,
            elapsedMs: c.elapsed_ms,
            message: c.message,
            remediation: c.remediation,
            envelope: normalizeEnvelope(c.envelope),
            metadata: c.metadata || {},
        })),
    };
}
