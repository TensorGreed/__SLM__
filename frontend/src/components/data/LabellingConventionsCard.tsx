/**
 * Labels step — the labelling conventions of a span project (Epic U phase
 * U4): compliance bundles, per-type definitions, custom infotypes. Rendered
 * only when the project's recipe scores span sets; other task shapes have
 * nothing to configure here.
 */

import { useState } from 'react';

import ComplianceBundlePicker from './ComplianceBundlePicker';
import SpanAddTypeForm from './SpanAddTypeForm';
import SpanTypeDefinitionsEditor from './SpanTypeDefinitionsEditor';

interface LabellingConventionsCardProps {
    projectId: number;
}

/** A span project: its recipe scores span sets, or it already carries span
 *  conventions / custom patterns (a project 31-style tagger whose recipe
 *  snapshot predates the span recipe). */
export function isSpanProject(project: { selected_recipe?: Record<string, unknown> | null; runtime_config?: Record<string, unknown> | null }): boolean {
    const snapshot = project.selected_recipe;
    if (snapshot && typeof snapshot === 'object' && String(snapshot.scoring_mode ?? '').toLowerCase() === 'span_set') return true;
    const rc = project.runtime_config;
    if (rc && typeof rc === 'object') {
        const defs = rc.span_type_definitions;
        if (defs && typeof defs === 'object' && Object.keys(defs as object).length > 0) return true;
        const patterns = rc.span_custom_patterns;
        if (Array.isArray(patterns) && patterns.length > 0) return true;
        if (Array.isArray((rc.compliance as { bundles?: unknown[] } | undefined)?.bundles) && ((rc.compliance as { bundles: unknown[] }).bundles.length > 0)) return true;
    }
    return false;
}

export default function LabellingConventionsCard({ projectId }: LabellingConventionsCardProps) {
    const [definitionsVersion, setDefinitionsVersion] = useState(0);
    // The picker reports the bundle's types; here there is no generation form
    // to feed, so the list is kept only to show what a bundle added.
    const [entityTypes, setEntityTypes] = useState<string[]>([]);
    return (
        <section className="card" data-testid="labelling-conventions">
            <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-sm)' }}>Labelling conventions</h3>
            <p className="form-hint" style={{ marginBottom: 'var(--space-md)' }}>
                Every labeller reads these: what each type means, where a span starts and ends, which compliance regime it serves.
            </p>
            <ComplianceBundlePicker projectId={projectId} entityTypes={entityTypes} onEntityTypesChange={setEntityTypes} />
            <SpanTypeDefinitionsEditor key={definitionsVersion} projectId={projectId} />
            <SpanAddTypeForm projectId={projectId} onTypeAdded={() => setDefinitionsVersion((v) => v + 1)} />
        </section>
    );
}
