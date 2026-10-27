/**
 * Data page — the project's tool catalog: the tools a model may call, imported
 * from a Model Context Protocol server (a local process or an HTTP endpoint) or
 * an OpenAPI document. Plans and tool calls are scored against it, and the
 * allowed tools are listed in the prompt for training and eval alike.
 *
 * Secret values (environment variables, request headers) are write-only: they
 * are stored server side and only their names come back. A blank value on
 * re-import keeps the stored secret. Visible for every project; collapsed by
 * default with the catalog version and tool count in the header.
 */

import { ChevronDown, ChevronRight } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import {
    clearToolCatalog,
    getToolCatalog,
    importMcpTools,
    importOpenApiTools,
    patchToolFlags,
    restoreToolCatalog,
    toolCatalogError,
    toolRiskLabel,
    toolSignature,
    type ToolCatalogErrorInfo,
    type ToolCatalogStatus,
} from '../../api/toolCatalog';
import { toast } from '../../stores/toastStore';
import './ToolCatalogCard.css';

interface Props {
    projectId: number;
}

type ImportKind = 'mcp_stdio' | 'mcp_http' | 'openapi';

interface Row {
    name: string;
    value: string;
}

const SOURCE_LABELS: Record<string, string> = {
    mcp_stdio: 'MCP server (local process)',
    mcp_http: 'MCP server (HTTP)',
    openapi: 'OpenAPI document',
    manual: 'entered by hand',
    domain_pack: 'domain pack',
};

export function sourceKindLabel(kind: string | null | undefined): string {
    return (kind && SOURCE_LABELS[kind]) || kind || 'unknown source';
}

function readText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ''));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

function SecretRows({ rows, setRows, kind }: { rows: Row[]; setRows: (rows: Row[]) => void; kind: 'env' | 'header' }) {
    const update = (i: number, patch: Partial<Row>) => setRows(rows.map((r, n) => (n === i ? { ...r, ...patch } : r)));
    return (
        <div data-testid={`tc-${kind}-rows`}>
            {rows.map((row, i) => (
                <div className="tc-row" key={i}>
                    <input
                        type="text"
                        value={row.name}
                        aria-label={`${kind === 'env' ? 'Environment variable' : 'Header'} name ${i + 1}`}
                        placeholder={kind === 'env' ? 'NAME' : 'Header-Name'}
                        onChange={(e) => update(i, { name: e.target.value })}
                        data-testid={`tc-${kind}-name-${i}`}
                    />
                    <input
                        type="password"
                        autoComplete="off"
                        value={row.value}
                        aria-label={`${kind === 'env' ? 'Environment variable' : 'Header'} value ${i + 1}`}
                        placeholder="blank keeps the stored value"
                        onChange={(e) => update(i, { value: e.target.value })}
                        data-testid={`tc-${kind}-value-${i}`}
                    />
                    <button type="button" className="btn btn-sm btn-secondary" onClick={() => setRows(rows.filter((_, n) => n !== i))}>
                        Remove
                    </button>
                </div>
            ))}
            <button type="button" className="btn btn-sm btn-secondary" onClick={() => setRows([...rows, { name: '', value: '' }])} data-testid={`tc-${kind}-add`}>
                {kind === 'env' ? 'Add variable' : 'Add header'}
            </button>
        </div>
    );
}

function toSecretMap(rows: Row[]): Record<string, string | null> {
    const out: Record<string, string | null> = {};
    for (const r of rows) {
        const name = r.name.trim();
        if (name) out[name] = r.value === '' ? null : r.value;
    }
    return out;
}

export default function ToolCatalogCard({ projectId }: Props) {
    const [status, setStatus] = useState<ToolCatalogStatus | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [open, setOpen] = useState(false);
    const [kind, setKind] = useState<ImportKind>('mcp_stdio');
    const [label, setLabel] = useState('');
    const [commandText, setCommandText] = useState('');
    const [envRows, setEnvRows] = useState<Row[]>([]);
    const [url, setUrl] = useState('');
    const [headerRows, setHeaderRows] = useState<Row[]>([]);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<ToolCatalogErrorInfo | null>(null);

    const prefill = useCallback((res: ToolCatalogStatus) => {
        const src = res.catalog?.source;
        if (!src) return;
        if (src.kind === 'mcp_stdio') {
            setKind('mcp_stdio');
            setCommandText((src.command ?? []).join('\n'));
            setEnvRows((src.env_names ?? []).map((name) => ({ name, value: '' })));
            setLabel(src.label ?? '');
        } else if (src.kind === 'mcp_http') {
            setKind('mcp_http');
            setUrl(src.url ?? '');
            setHeaderRows((src.header_names ?? []).map((name) => ({ name, value: '' })));
            setLabel(src.label ?? '');
        }
    }, []);

    useEffect(() => {
        let cancelled = false;
        getToolCatalog(projectId)
            .then((res) => { if (!cancelled) { setStatus(res); prefill(res); } })
            .catch(() => { if (!cancelled) setLoadError('Could not load the tool catalog.'); });
        return () => { cancelled = true; };
    }, [projectId, prefill]);

    const run = async (action: () => Promise<ToolCatalogStatus>, done: string) => {
        setBusy(true);
        setError(null);
        try {
            const res = await action();
            if (res) setStatus(res);
            toast.success(done, 3000);
        } catch (err) {
            setError(toolCatalogError(err));
        } finally {
            setBusy(false);
        }
    };

    const importMcp = () => {
        if (kind === 'mcp_stdio') {
            const command = commandText.split('\n').map((l) => l.trim()).filter(Boolean);
            void run(() => importMcpTools(projectId, { kind: 'mcp_stdio', command, env: toSecretMap(envRows), label: label.trim() || undefined }), 'Tools imported.');
        } else {
            void run(() => importMcpTools(projectId, { kind: 'mcp_http', url: url.trim(), headers: toSecretMap(headerRows), label: label.trim() || undefined }), 'Tools imported.');
        }
    };

    const onOpenApiFile = async (list: FileList | null) => {
        const file = list?.[0];
        if (!file) return;
        let content: string;
        try {
            content = await readText(file);
        } catch {
            setError({ code: null, message: `${file.name} could not be read.`, errors: [] });
            return;
        }
        void run(() => importOpenApiTools(projectId, file.name, content), 'Tools imported.');
    };

    const setFlags = (flags: Parameters<typeof patchToolFlags>[1], done = 'Tool settings saved.') =>
        run(() => patchToolFlags(projectId, flags), done);

    if (loadError) return <p className="u-hint" data-testid="tc-load-error">{loadError}</p>;
    if (!status) return <p className="u-hint">Loading tool catalog…</p>;

    const catalog = status.catalog;
    const tools = catalog?.tools ?? [];
    const allowedCount = tools.filter((t) => t.allowed).length;
    const src = catalog?.source ?? null;
    const server = catalog?.server ?? null;

    return (
        <section className="card tc-card" id="tool-catalog" data-testid="tool-catalog-card">
            <div className="tc-header">
                <button type="button" className="btn btn-secondary" onClick={() => setOpen(!open)} data-testid="tc-toggle">
                    {open ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />}
                    <span>Tool catalog</span>
                </button>
                {catalog ? (
                    <span className="tc-chip" data-testid="tc-summary">
                        {tools.length} tool{tools.length === 1 ? '' : 's'} · {allowedCount} allowed · version {catalog.version}
                    </span>
                ) : (
                    <span className="tc-chip" data-testid="tc-summary">no catalog</span>
                )}
            </div>

            {open && (
                <div className="tc-body">
                    <p className="u-hint u-mt0">
                        The tools a model may call. Import them from a tool server or an API description; tool calls and plans are then
                        checked against these tools and their argument schemas. Only the tools you allow are listed in the prompt.
                        Tools are never run unless you turn on dry runs, and then only read-only tools.
                    </p>

                    {catalog && (
                        <p className="u-hint" data-testid="tc-provenance">
                            Catalog version <code>{catalog.version}</code>
                            {' · '}source: {sourceKindLabel(src?.kind)}{src?.label ? ` "${src.label}"` : ''}
                            {src?.kind === 'openapi' && src.filename ? ` (${src.filename}${src.version ? `, API version ${src.version}` : ''})` : ''}
                            {src?.kind === 'mcp_http' && src.url ? ` (${src.url})` : ''}
                            {src?.kind === 'mcp_stdio' && src.command ? ` (${src.command.join(' ')})` : ''}
                            {server?.name ? ` · server ${server.name}${server.version ? ` ${server.version}` : ''}` : ''}
                            {catalog.imported_at ? ` · imported ${catalog.imported_at}` : ''}
                            {catalog.flags_version ? ` · flags ${catalog.flags_version}` : ''}
                        </p>
                    )}

                    <fieldset className="tc-group">
                        <legend>Import tools</legend>
                        <div className="tc-tabs">
                            {(['mcp_stdio', 'mcp_http', 'openapi'] as ImportKind[]).map((k) => (
                                <button
                                    key={k}
                                    type="button"
                                    className={`btn btn-sm ${kind === k ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setKind(k)}
                                    data-testid={`tc-tab-${k}`}
                                >
                                    {k === 'mcp_stdio' ? 'Local server' : k === 'mcp_http' ? 'HTTP server' : 'OpenAPI file'}
                                </button>
                            ))}
                        </div>

                        {kind !== 'openapi' && (
                            <div className="tc-row">
                                <label htmlFor="tc-label">Label</label>
                                <input id="tc-label" type="text" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="shown as provenance" data-testid="tc-label" />
                            </div>
                        )}

                        {kind === 'mcp_stdio' && (
                            <>
                                <label htmlFor="tc-command" className="u-hint u-mt0">
                                    Command and arguments, one per line. The program must be on the server's allow-list.
                                </label>
                                <textarea id="tc-command" className="tc-textarea" value={commandText} onChange={(e) => setCommandText(e.target.value)} data-testid="tc-command" />
                                <p className="u-hint u-mt0">Environment variables (values are stored as project secrets and never shown again)</p>
                                <SecretRows rows={envRows} setRows={setEnvRows} kind="env" />
                                <div className="tc-row">
                                    <button type="button" className="btn btn-primary" onClick={importMcp} disabled={busy || !commandText.trim()} data-testid="tc-import">
                                        {busy ? 'Importing…' : 'Import tools'}
                                    </button>
                                </div>
                            </>
                        )}

                        {kind === 'mcp_http' && (
                            <>
                                <div className="tc-row">
                                    <label htmlFor="tc-url">Server URL</label>
                                    <input id="tc-url" type="url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://host/mcp" data-testid="tc-url" />
                                </div>
                                <p className="u-hint u-mt0">Request headers (values are stored as project secrets and never shown again)</p>
                                <SecretRows rows={headerRows} setRows={setHeaderRows} kind="header" />
                                <div className="tc-row">
                                    <button type="button" className="btn btn-primary" onClick={importMcp} disabled={busy || !url.trim()} data-testid="tc-import">
                                        {busy ? 'Importing…' : 'Import tools'}
                                    </button>
                                </div>
                            </>
                        )}

                        {kind === 'openapi' && (
                            <div className="tc-row">
                                <label htmlFor="tc-openapi">API description (JSON or YAML)</label>
                                <input id="tc-openapi" type="file" accept=".json,.yaml,.yml,application/json" disabled={busy} data-testid="tc-openapi-upload"
                                    onChange={(e) => { void onOpenApiFile(e.target.files); e.target.value = ''; }} />
                            </div>
                        )}

                        {error && (
                            <div className="tc-error" data-testid="tc-error">
                                {error.code && <strong><code>{error.code}</code> </strong>}
                                {error.message}
                                {error.errors.length > 0 && (
                                    <ul>{error.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
                                )}
                            </div>
                        )}
                    </fieldset>

                    {catalog && tools.length > 0 && (
                        <>
                            <div className="tc-row">
                                <button type="button" className="btn btn-sm btn-secondary" disabled={busy} data-testid="tc-allow-all"
                                    onClick={() => void setFlags(Object.fromEntries(tools.map((t) => [t.name, { allowed: true }])), 'All tools allowed.')}>
                                    Allow all
                                </button>
                                <button type="button" className="btn btn-sm btn-secondary" disabled={busy} data-testid="tc-deny-all"
                                    onClick={() => void setFlags(Object.fromEntries(tools.map((t) => [t.name, { allowed: false }])), 'All tools denied.')}>
                                    Deny all
                                </button>
                                <button type="button" className="btn btn-sm btn-secondary" disabled={busy} data-testid="tc-deny-mutating"
                                    onClick={() => void setFlags(Object.fromEntries(tools.filter((t) => t.mutating).map((t) => [t.name, { allowed: false }])), 'Mutating tools denied.')}>
                                    Deny mutating
                                </button>
                            </div>
                            <table className="tc-table" data-testid="tc-table">
                                <thead>
                                    <tr><th>Tool</th><th>Signature</th><th>Kind</th><th>Allowed</th><th>Read-only override</th></tr>
                                </thead>
                                <tbody>
                                    {tools.map((t) => {
                                        const risk = toolRiskLabel(t);
                                        const destructive = t.destructive === true;
                                        const override = t.read_only_effective === (t.read_only === true) ? 'server' : t.read_only_effective ? 'yes' : 'no';
                                        return (
                                            <tr key={t.name} data-testid={`tc-tool-${t.name}`}>
                                                <td>
                                                    <code>{t.name}</code>
                                                    {t.description && <span className="tc-sub">{t.description}</span>}
                                                    {t.schema_error && <span className="tc-bad" data-testid={`tc-schema-error-${t.name}`}>schema problem: {t.schema_error}</span>}
                                                </td>
                                                <td><code className="tc-sig">{toolSignature(t)}</code></td>
                                                <td><span className={`tc-badge tc-badge--${risk}`} data-testid={`tc-risk-${t.name}`}>{risk}</span></td>
                                                <td>
                                                    <input type="checkbox" checked={t.allowed} disabled={busy}
                                                        aria-label={`Allow ${t.name}`}
                                                        onChange={(e) => void setFlags({ [t.name]: { allowed: e.target.checked } })}
                                                        data-testid={`tc-allowed-${t.name}`} />
                                                </td>
                                                <td>
                                                    <select value={override} disabled={busy || destructive} aria-label={`Read-only override for ${t.name}`}
                                                        onChange={(e) => {
                                                            const v = e.target.value;
                                                            void setFlags({ [t.name]: { read_only: v === 'server' ? null : v === 'yes' } });
                                                        }}
                                                        data-testid={`tc-override-${t.name}`}>
                                                        <option value="server">use server hint</option>
                                                        <option value="yes">treat as read-only</option>
                                                        <option value="no">treat as mutating</option>
                                                    </select>
                                                    {destructive && <span className="tc-sub">destructive tools are never read-only</span>}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </>
                    )}
                    {catalog && tools.length === 0 && <p className="u-hint" data-testid="tc-empty">The catalog has no tools.</p>}

                    {status.history.length > 0 && (
                        <div data-testid="tc-history">
                            <h4 className="tc-h4">History</h4>
                            <ul className="tc-history">
                                {status.history.map((h) => (
                                    <li key={h.version} data-testid={`tc-history-${h.version}`}>
                                        <code>{h.version}</code>
                                        <span>{sourceKindLabel(h.source_kind)} · {h.tool_count} tools{h.imported_at ? ` · ${h.imported_at}` : ''}</span>
                                        {h.current
                                            ? <span className="tc-chip">current</span>
                                            : h.restorable
                                                ? <button type="button" className="btn btn-sm btn-secondary" disabled={busy}
                                                    onClick={() => void run(() => restoreToolCatalog(projectId, h.version), 'Catalog restored.')}>Restore</button>
                                                : <span className="tc-sub">content trimmed</span>}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {catalog && (
                        <div className="tc-row">
                            <button type="button" className="btn btn-sm btn-secondary" disabled={busy} data-testid="tc-clear"
                                onClick={() => void run(async () => { await clearToolCatalog(projectId); return { catalog: null, history: status.history }; }, 'Catalog cleared.')}>
                                Clear catalog
                            </button>
                        </div>
                    )}
                </div>
            )}
        </section>
    );
}
