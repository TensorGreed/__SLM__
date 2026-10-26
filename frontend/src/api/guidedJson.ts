/**
 * Structured-output hint for serving (code-findings recipe).
 * GET /api/projects/{id}/guided-json -> { guided_json: {...} | null }
 */

import api from './client';

export interface GuidedJsonHint {
    /** JSON Schema for the model's output; label enum = taxonomy ids when known. */
    schema: Record<string, unknown>;
    /** e.g. "recipe:code-findings". */
    source: string;
    taxonomy_version: string | null;
    updated_at?: string | null;
}

export async function getGuidedJson(projectId: number): Promise<GuidedJsonHint | null> {
    const res = await api.get<{ guided_json: GuidedJsonHint | null }>(
        `/projects/${projectId}/guided-json`,
    );
    return res?.data?.guided_json ?? null;
}

/** Number of allowed labels in the schema's `label` enum, or null when open. */
export function guidedJsonLabelCount(schema: Record<string, unknown>): number | null {
    const seen = new Set<unknown>();
    const walk = (node: unknown): number | null => {
        if (!node || typeof node !== 'object' || seen.has(node)) return null;
        seen.add(node);
        const obj = node as Record<string, unknown>;
        const props = obj.properties as Record<string, unknown> | undefined;
        const label = props?.label as { enum?: unknown[] } | undefined;
        if (label && Array.isArray(label.enum)) return label.enum.length;
        for (const v of Object.values(obj)) {
            const found = walk(v);
            if (found !== null) return found;
        }
        return null;
    };
    return walk(schema);
}
