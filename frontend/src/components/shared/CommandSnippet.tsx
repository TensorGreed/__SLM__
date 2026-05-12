/**
 * CommandSnippet — collapsed "Show as CLI / API" disclosure rendered
 * next to a primary action button. When opened, surfaces the
 * equivalent `brewslm` invocation + `curl` call so a UI-first user
 * picks up the other two surfaces by osmosis.
 *
 * Stateless — caller passes the live CLI string + API descriptor.
 * Caller is responsible for keeping those in sync with form state;
 * the `buildSnippet` helper does the typical case (cli + curl from
 * one shape) so most call sites are one line.
 */

import { useCallback, useMemo, useState } from 'react';

import './CommandSnippet.css';

export interface ApiSnippet {
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
    path: string;
    body?: unknown;
}

interface Props {
    cli: string;
    api: ApiSnippet;
    /** Optional override for the API host shown in the curl example. */
    apiBaseHint?: string;
    /** Optional label override. Defaults to "Show as CLI / API". */
    label?: string;
}

type ActiveTab = 'cli' | 'api';

const DEFAULT_API_BASE = 'http://localhost:8000/api';

function formatCurl(api: ApiSnippet, apiBase: string): string {
    const url = `${apiBase}${api.path}`;
    const lines: string[] = [`curl -X ${api.method} ${url}`];
    if (api.body !== undefined && api.body !== null && api.method !== 'GET') {
        lines.push("  -H 'Content-Type: application/json' \\");
        const json = JSON.stringify(api.body, null, 2);
        lines.push(`  -d '${json}'`);
        // Glue the first line with backslash continuation.
        lines[0] = lines[0] + ' \\';
    }
    return lines.join('\n');
}

async function copyToClipboard(text: string): Promise<boolean> {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            return false;
        }
    }
    return false;
}

export default function CommandSnippet({
    cli,
    api,
    apiBaseHint = DEFAULT_API_BASE,
    label = 'Show as CLI / API',
}: Props) {
    const [open, setOpen] = useState(false);
    const [activeTab, setActiveTab] = useState<ActiveTab>('cli');
    const [copyHint, setCopyHint] = useState<string | null>(null);

    const curl = useMemo(() => formatCurl(api, apiBaseHint), [api, apiBaseHint]);

    const handleCopy = useCallback(
        async (kind: ActiveTab) => {
            const payload = kind === 'cli' ? cli : curl;
            const ok = await copyToClipboard(payload);
            setCopyHint(ok ? 'Copied!' : 'Copy failed');
            window.setTimeout(() => setCopyHint(null), 1400);
        },
        [cli, curl],
    );

    if (!open) {
        return (
            <button
                type="button"
                className="command-snippet-toggle"
                onClick={() => setOpen(true)}
                aria-expanded="false"
                aria-label="Show this action as CLI or API"
            >
                <span>{label}</span>
                <span className="command-snippet-toggle-chevron">⌄</span>
            </button>
        );
    }

    return (
        <div className="command-snippet" role="region" aria-label="Equivalent CLI and API">
            <div className="command-snippet-head">
                <div
                    className="command-snippet-tabs"
                    role="tablist"
                    aria-label="Snippet surface"
                >
                    <button
                        type="button"
                        role="tab"
                        aria-selected={activeTab === 'cli'}
                        className={`command-snippet-tab ${activeTab === 'cli' ? 'is-active' : ''}`}
                        onClick={() => setActiveTab('cli')}
                    >
                        CLI
                    </button>
                    <button
                        type="button"
                        role="tab"
                        aria-selected={activeTab === 'api'}
                        className={`command-snippet-tab ${activeTab === 'api' ? 'is-active' : ''}`}
                        onClick={() => setActiveTab('api')}
                    >
                        API
                    </button>
                </div>
                <div className="command-snippet-actions">
                    {copyHint && (
                        <span className="command-snippet-copy-hint" role="status">
                            {copyHint}
                        </span>
                    )}
                    <button
                        type="button"
                        className="command-snippet-copy"
                        onClick={() => void handleCopy(activeTab)}
                        aria-label={`Copy ${activeTab.toUpperCase()} snippet`}
                    >
                        Copy
                    </button>
                    <button
                        type="button"
                        className="command-snippet-close"
                        onClick={() => setOpen(false)}
                        aria-label="Hide snippet"
                    >
                        ✕
                    </button>
                </div>
            </div>
            <pre className="command-snippet-code" role="tabpanel">
                <code>{activeTab === 'cli' ? cli : curl}</code>
            </pre>
        </div>
    );
}
