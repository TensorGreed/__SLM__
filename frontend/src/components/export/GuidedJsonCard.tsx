/**
 * GuidedJsonCard — read-only "Structured output hint" on Ship > Package.
 * The schema a serving stack can hand to a guided-decoding option so the
 * model can only answer in the findings shape. Renders nothing when the
 * project has no hint (only the code-findings recipe writes one).
 */

import { useEffect, useState } from 'react';

import './GuidedJsonCard.css';
import { getGuidedJson, guidedJsonLabelCount, type GuidedJsonHint } from '../../api/guidedJson';

export default function GuidedJsonCard({ projectId }: { projectId: number }) {
    const [hint, setHint] = useState<GuidedJsonHint | null>(null);

    useEffect(() => {
        let cancelled = false;
        getGuidedJson(projectId)
            .then((data) => { if (!cancelled) setHint(data); })
            .catch(() => { if (!cancelled) setHint(null); });
        return () => { cancelled = true; };
    }, [projectId]);

    if (!hint) return null;
    const labelCount = guidedJsonLabelCount(hint.schema);
    return (
        <details className="card" data-testid="guided-json-card">
            <summary className="u-card-title">Structured output hint</summary>
            <p className="u-hint u-mt0 u-mb-md">
                Pass this schema to a serving stack&apos;s guided-decoding option so the model can only answer in the findings shape.
                Read-only: it follows the recipe and the label taxonomy.
            </p>
            <ul className="guided-json__facts">
                <li data-testid="guided-json-source">Source: <code>{hint.source}</code></li>
                <li data-testid="guided-json-taxonomy">
                    Label taxonomy version: {hint.taxonomy_version ? <code>{hint.taxonomy_version}</code> : 'none (labels are open)'}
                </li>
                <li data-testid="guided-json-labels">
                    Labels allowed: {labelCount === null ? 'any' : labelCount}
                </li>
                {hint.updated_at && <li>Updated: {hint.updated_at}</li>}
            </ul>
            <pre className="guided-json__schema" data-testid="guided-json-schema">
                {JSON.stringify(hint.schema, null, 2)}
            </pre>
        </details>
    );
}
