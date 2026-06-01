/**
 * Shared error rendering component (Diagnostics Intervention A).
 *
 * One renderer for every API error across the platform. Replaces the
 * scattered ad-hoc toasts / banners / inline ``<p className="error">``
 * spans that have grown organically and made errors harder to
 * diagnose end-to-end.
 *
 * Usage:
 *   try { await api.post(...) }
 *   catch (err) {
 *     setError(parseErrorEnvelope(err));
 *   }
 *   ...
 *   {error && <ErrorPanel envelope={error} onDismiss={() => setError(null)} />}
 *
 * Shape:
 *   ┌─[Status badge] Message headline ─────────────────[× dismiss]┐
 *   │ Actionable fix sentence.                                    │
 *   │                                                              │
 *   │ ▶ Show technical details                                     │
 *   │ (collapsed by default; expands to show metadata + stage)    │
 *   │                                                              │
 *   │ Troubleshooting id: err_a3f8d4e2  [copy]   [Docs ↗]         │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * The troubleshooting_id is selectable + copyable so a user can drop
 * it in a bug report. The developer greps server logs for that id to
 * find the full traceback (the last-resort handler logs it under the
 * same id).
 */

import { Fragment, useState } from 'react';

import type { ErrorEnvelope } from '../../api/errors';
import './ErrorPanel.css';


export interface ErrorPanelProps {
    envelope: ErrorEnvelope;
    onDismiss?: () => void;
    /** Optional: render extra action buttons in the panel footer
     *  (e.g. "Retry with Qwen 2.5" for the synth refusal case). */
    actions?: React.ReactNode;
    /** Optional: override the data-testid prefix so multiple panels
     *  on the same page can be addressed independently. */
    testIdPrefix?: string;
}


export default function ErrorPanel({
    envelope,
    onDismiss,
    actions,
    testIdPrefix = 'error-panel',
}: ErrorPanelProps) {
    const [detailsOpen, setDetailsOpen] = useState(false);
    const [copied, setCopied] = useState(false);

    const severity = envelope.statusCode >= 500 ? 'critical' : 'warning';

    const copyTroubleshootingId = async () => {
        try {
            await navigator.clipboard.writeText(envelope.troubleshootingId);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Clipboard may be blocked in some browser contexts. The
            // id is selectable in the DOM so the user can fall back
            // to manual copy. Silent failure here is fine.
        }
    };

    const hasMetadata = envelope.metadata !== null
        && Object.keys(envelope.metadata).length > 0;

    return (
        <div
            className={`error-panel error-panel--${severity}`}
            role="alert"
            data-testid={testIdPrefix}
            data-error-code={envelope.errorCode}
        >
            <header className="error-panel__head">
                <span
                    className={`error-panel__badge error-panel__badge--${severity}`}
                    data-testid={`${testIdPrefix}-status`}
                >
                    {envelope.statusCode === 0 ? 'Network' : envelope.statusCode}
                </span>
                <p
                    className="error-panel__headline"
                    data-testid={`${testIdPrefix}-message`}
                >
                    {envelope.message}
                </p>
                {onDismiss && (
                    <button
                        type="button"
                        className="error-panel__dismiss"
                        onClick={onDismiss}
                        aria-label="Dismiss error"
                        data-testid={`${testIdPrefix}-dismiss`}
                    >
                        ×
                    </button>
                )}
            </header>

            {envelope.actionableFix && (
                <p
                    className="error-panel__remediation"
                    data-testid={`${testIdPrefix}-remediation`}
                >
                    {envelope.actionableFix}
                </p>
            )}

            {envelope.isFallback && (
                <p
                    className="error-panel__fallback-note"
                    data-testid={`${testIdPrefix}-fallback-note`}
                >
                    The server didn't return a structured error envelope —
                    this rendering is a best-effort fallback. The
                    troubleshooting id below is generated client-side.
                </p>
            )}

            {hasMetadata && (
                <details
                    className="error-panel__details"
                    open={detailsOpen}
                    onToggle={(e) => setDetailsOpen((e.target as HTMLDetailsElement).open)}
                    data-testid={`${testIdPrefix}-details`}
                >
                    <summary>
                        {detailsOpen ? '▼' : '▶'} Technical details
                    </summary>
                    <dl className="error-panel__details-grid">
                        <dt>Stage</dt><dd>{envelope.stage}</dd>
                        <dt>Error code</dt><dd><code>{envelope.errorCode}</code></dd>
                        {Object.entries(envelope.metadata!).map(([key, value]) => (
                            <Fragment key={key}>
                                <dt>{key}</dt>
                                <dd>
                                    <code className="error-panel__metadata-value">
                                        {formatMetadataValue(value)}
                                    </code>
                                </dd>
                            </Fragment>
                        ))}
                    </dl>
                </details>
            )}

            <footer className="error-panel__foot">
                <span
                    className="error-panel__trace-id"
                    data-testid={`${testIdPrefix}-trace-id`}
                >
                    <span className="error-panel__trace-label">
                        Troubleshooting id:
                    </span>{' '}
                    <code>{envelope.troubleshootingId}</code>
                </span>
                <button
                    type="button"
                    className="error-panel__copy"
                    onClick={() => void copyTroubleshootingId()}
                    data-testid={`${testIdPrefix}-copy`}
                >
                    {copied ? '✓ copied' : 'copy'}
                </button>
                {envelope.docsUrl && envelope.docsUrl !== '/docs/troubleshooting' && (
                    <a
                        href={envelope.docsUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="error-panel__docs-link"
                        data-testid={`${testIdPrefix}-docs-link`}
                    >
                        Docs ↗
                    </a>
                )}
                {actions && (
                    <span className="error-panel__actions">{actions}</span>
                )}
            </footer>
        </div>
    );
}


/**
 * Format a metadata value for the technical-details grid. Strings + numbers
 * + booleans render as-is; objects + arrays render as compact JSON so the
 * grid doesn't blow up vertically.
 */
function formatMetadataValue(value: unknown): string {
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean' || value === null) {
        return String(value);
    }
    try {
        const json = JSON.stringify(value);
        return json.length > 200 ? `${json.slice(0, 200)}…` : json;
    } catch {
        return '(unserializable)';
    }
}
