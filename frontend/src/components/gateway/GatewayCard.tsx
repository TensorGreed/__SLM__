/**
 * GatewayCard — Ship > Deploy "Use from Claude Code": the model ids this
 * project serves (with project + run + base provenance), copy-ready
 * snippets, API keys, gateway settings, named tools and recent requests.
 */

import { useCallback, useEffect, useState } from 'react';

import './gateway.css';
import {
    createGatewayKey, getGatewaySettings, getGatewaySnippets, getGatewayTelemetry,
    listGatewayKeys, parseGatewayError, putGatewaySettings, revokeGatewayKey,
    type FieldError, type GatewayKey, type GatewaySettingsResponse, type GatewaySnippets,
    type GatewayTelemetry, type GatewayTool, type GatewayUpstream, type GatewayUpstreamKind,
    type ServableRun,
} from '../../api/gateway';
import { toast } from '../../stores/toastStore';

export function runLabel(run: ServableRun): string {
    return `run #${run.id} · ${run.name} · ${run.base_model ?? 'unknown base'}`;
}

export async function copyText(text: string, what: string): Promise<void> {
    try {
        await navigator.clipboard.writeText(text);
        toast.success(`Copied ${what}`);
    } catch {
        toast.error('Could not copy to the clipboard');
    }
}

function CodeBlock({ id, label, text }: { id: string; label: string; text: string }) {
    return (
        <div>
            <div className="gateway__row">
                <strong>{label}</strong>
                <button type="button" className="btn btn-sm btn-secondary" data-testid={`gateway-copy-${id}`}
                    onClick={() => void copyText(text, label)}>Copy</button>
            </div>
            <pre className="gateway__code" data-testid={`gateway-snippet-${id}`}>{text}</pre>
        </div>
    );
}

function ErrorList({ message, errors, prefix }: { message: string | null; errors: FieldError[]; prefix?: string }) {
    const shown = prefix === undefined ? errors : errors.filter((e) => e.field.startsWith(prefix));
    if (!message && shown.length === 0) return null;
    return (
        <div className="gateway__error" role="alert" data-testid="gateway-errors">
            {message && <p>{message}</p>}
            <ul>
                {shown.map((e, i) => <li key={i} data-testid={`gateway-error-${e.field}`}>{e.field}: {e.message}</li>)}
            </ul>
        </div>
    );
}

/* ---------- API keys ---------- */

function KeysSection({ projectId }: { projectId: number }) {
    const [keys, setKeys] = useState<GatewayKey[]>([]);
    const [name, setName] = useState('');
    const [rate, setRate] = useState('60');
    const [secret, setSecret] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(() => {
        listGatewayKeys()
            .then((all) => setKeys(all.filter((k) => k.project_ids.includes(projectId))))
            .catch(() => setKeys([]));
    }, [projectId]);
    useEffect(() => { load(); }, [load]);

    const create = async () => {
        setError(null);
        try {
            const res = await createGatewayKey({
                name: name.trim(), project_ids: [projectId], rate_limit_per_minute: Number(rate) || 60,
            });
            setSecret(res.key);
            setName('');
            load();
        } catch (err) {
            setError(parseGatewayError(err).message);
        }
    };
    const revoke = async (k: GatewayKey) => {
        if (!window.confirm(`Revoke key ${k.key_prefix}... (${k.name})? Clients using it stop working.`)) return;
        try {
            await revokeGatewayKey(k.id);
            load();
        } catch (err) {
            setError(parseGatewayError(err).message);
        }
    };

    return (
        <fieldset className="gateway__section" data-testid="gateway-keys">
            <h4>API keys</h4>
            <p className="u-hint u-mt0">
                Clients send the key as <code>Authorization: Bearer</code> or <code>X-API-Key</code>. The snippets
                above use the placeholder <code>{'${BREWSLM_API_KEY}'}</code>: export your real key under that name.
            </p>
            {secret && (
                <div className="gateway__secret" data-testid="gateway-secret">
                    <p className="u-mt0">Copy this key now. You won&apos;t see it again.</p>
                    <code data-testid="gateway-secret-value">{secret}</code>
                    <div className="gateway__row">
                        <button type="button" className="btn btn-sm btn-secondary" data-testid="gateway-secret-copy"
                            onClick={() => void copyText(secret, 'the key')}>Copy key</button>
                        <button type="button" className="btn btn-sm btn-primary" data-testid="gateway-secret-dismiss"
                            onClick={() => setSecret(null)}>I saved it</button>
                    </div>
                </div>
            )}
            {keys.length === 0 ? (
                <p className="u-hint" data-testid="gateway-keys-empty">No keys for this project yet.</p>
            ) : (
                <table className="gateway__table">
                    <thead><tr><th>Prefix</th><th>Name</th><th>Projects</th><th>Rate / min</th><th>Last used</th><th>Status</th><th /></tr></thead>
                    <tbody>
                        {keys.map((k) => (
                            <tr key={k.id} data-testid={`gateway-key-${k.id}`}>
                                <td><code>{k.key_prefix}</code></td>
                                <td>{k.name}</td>
                                <td>{k.project_ids.join(', ')}</td>
                                <td>{k.rate_limit_per_minute}</td>
                                <td>{k.last_used_at ?? 'never'}</td>
                                <td>{k.revoked_at || !k.is_active ? 'revoked' : 'active'}</td>
                                <td>
                                    {k.is_active && !k.revoked_at && (
                                        <button type="button" className="btn btn-sm btn-danger" data-testid={`gateway-revoke-${k.id}`}
                                            onClick={() => void revoke(k)}>Revoke</button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            <div className="gateway__row">
                <label className="gateway__field">Key name
                    <input type="text" value={name} onChange={(e) => setName(e.target.value)} data-testid="gateway-key-name" />
                </label>
                <label className="gateway__field">Requests per minute
                    <input type="number" min={1} value={rate} onChange={(e) => setRate(e.target.value)} data-testid="gateway-key-rate" />
                </label>
                <button type="button" className="btn btn-primary" disabled={!name.trim()} data-testid="gateway-key-create"
                    onClick={() => void create()}>Create key</button>
            </div>
            {error && <p className="gateway__error" role="alert">{error}</p>}
        </fieldset>
    );
}

/* ---------- Settings ---------- */

function SettingsSection({ projectId, data, onSaved }: {
    projectId: number; data: GatewaySettingsResponse; onSaved: (d: GatewaySettingsResponse) => void;
}) {
    const s = data.settings;
    const [upstream, setUpstream] = useState<GatewayUpstream>(s.upstream ?? 'in_process');
    const [baseUrl, setBaseUrl] = useState(s.base_url ?? '');
    const [kind, setKind] = useState<GatewayUpstreamKind>(s.upstream_kind ?? 'vllm');
    const [defaultRun, setDefaultRun] = useState(s.default_run_id ? String(s.default_run_id) : '');
    const [message, setMessage] = useState<string | null>(null);
    const [errors, setErrors] = useState<FieldError[]>([]);

    const save = async () => {
        setMessage(null);
        setErrors([]);
        const body: Record<string, unknown> = {
            upstream, default_run_id: defaultRun ? Number(defaultRun) : null,
        };
        if (upstream === 'openai_compatible') {
            body.base_url = baseUrl.trim();
            body.upstream_kind = kind;
        }
        try {
            onSaved(await putGatewaySettings(projectId, body));
            toast.success('Gateway settings saved');
        } catch (err) {
            const parsed = parseGatewayError(err);
            setMessage(parsed.message);
            setErrors(parsed.errors);
        }
    };

    return (
        <fieldset className="gateway__section" data-testid="gateway-settings">
            <h4>Gateway settings</h4>
            <div className="gateway__row">
                <label className="gateway__field">Upstream
                    <select value={upstream} onChange={(e) => setUpstream(e.target.value as GatewayUpstream)} data-testid="gateway-upstream">
                        <option value="in_process">in_process (this backend runs the model)</option>
                        <option value="openai_compatible">openai_compatible (vLLM, Ollama, ...)</option>
                    </select>
                </label>
                <label className="gateway__field">Default run for <code>{data.slug}</code>
                    <select value={defaultRun} onChange={(e) => setDefaultRun(e.target.value)} data-testid="gateway-default-run">
                        <option value="">Latest completed run</option>
                        {data.servable_runs.map((r) => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
                    </select>
                </label>
            </div>
            {upstream === 'openai_compatible' && (
                <div className="gateway__row">
                    <label className="gateway__field">Base URL
                        <input type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)}
                            placeholder="http://localhost:8001" data-testid="gateway-base-url" />
                    </label>
                    <label className="gateway__field">Server kind
                        <select value={kind} onChange={(e) => setKind(e.target.value as GatewayUpstreamKind)} data-testid="gateway-kind">
                            <option value="vllm">vllm</option>
                            <option value="ollama">ollama</option>
                            <option value="openai">openai</option>
                        </select>
                    </label>
                </div>
            )}
            <ErrorList message={message} errors={errors} />
            <button type="button" className="btn btn-primary" data-testid="gateway-settings-save" onClick={() => void save()}>
                Save settings
            </button>
        </fieldset>
    );
}

/* ---------- Named tools ---------- */

interface ToolDraft {
    index: number | null;
    name: string;
    description: string;
    schema: string;
    model: string;
    template: string;
    system: string;
    retrieval: 'default' | 'on' | 'off';
    maxTokens: string;
}

const EMPTY_SCHEMA = '{\n  "type": "object",\n  "properties": {\n    "input": { "type": "string" }\n  },\n  "required": ["input"]\n}';

function newDraft(): ToolDraft {
    return { index: null, name: '', description: '', schema: EMPTY_SCHEMA, model: '', template: '{input}', system: '', retrieval: 'default', maxTokens: '' };
}

function toolToDraft(t: GatewayTool, index: number): ToolDraft {
    return {
        index, name: t.name, description: t.description, schema: JSON.stringify(t.input_schema, null, 2),
        model: t.model ?? '', template: t.prompt_template, system: t.system_prompt ?? '',
        retrieval: t.retrieval === true ? 'on' : t.retrieval === false ? 'off' : 'default', maxTokens: t.max_tokens ? String(t.max_tokens) : '',
    };
}

function ToolsSection({ projectId, data, onSaved }: {
    projectId: number; data: GatewaySettingsResponse; onSaved: (d: GatewaySettingsResponse) => void;
}) {
    const tools = data.settings.tools ?? [];
    const [draft, setDraft] = useState<ToolDraft | null>(null);
    const [localError, setLocalError] = useState<string | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const [errors, setErrors] = useState<FieldError[]>([]);

    const put = async (next: GatewayTool[]): Promise<boolean> => {
        setMessage(null);
        setErrors([]);
        try {
            onSaved(await putGatewaySettings(projectId, { tools: next }));
            return true;
        } catch (err) {
            const parsed = parseGatewayError(err);
            setMessage(parsed.message);
            setErrors(parsed.errors);
            return false;
        }
    };

    const saveDraft = async () => {
        if (!draft) return;
        setLocalError(null);
        let schema: Record<string, unknown>;
        try {
            const parsed = JSON.parse(draft.schema);
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('not an object');
            schema = parsed as Record<string, unknown>;
        } catch {
            setLocalError('Input schema must be a valid JSON object.');
            return;
        }
        const tool: GatewayTool = {
            name: draft.name.trim(), description: draft.description.trim(), input_schema: schema,
            model: draft.model, prompt_template: draft.template,
        };
        if (draft.retrieval !== 'default') tool.retrieval = draft.retrieval === 'on';
        if (draft.system.trim()) tool.system_prompt = draft.system;
        if (draft.maxTokens.trim()) tool.max_tokens = Number(draft.maxTokens);
        const next = [...tools];
        if (draft.index === null) next.push(tool); else next[draft.index] = tool;
        if (await put(next)) setDraft(null);
    };

    const remove = async (i: number) => {
        if (!window.confirm(`Delete tool ${tools[i].name}?`)) return;
        await put(tools.filter((_, j) => j !== i));
    };

    const set = (patch: Partial<ToolDraft>) => setDraft((d) => (d ? { ...d, ...patch } : d));

    return (
        <fieldset className="gateway__section" data-testid="gateway-tools">
            <h4>Named tools</h4>
            <p className="u-hint u-mt0" data-testid="gateway-guided-line">
                {data.guided_json_available
                    ? "Replies are constrained to the project's output schema."
                    : 'This project has no output schema, so replies are not constrained.'}
            </p>
            {tools.length === 0 && !draft && <p className="u-hint" data-testid="gateway-tools-empty">No named tools yet.</p>}
            {tools.map((t, i) => (
                <div key={`${t.name}-${i}`} className="gateway__row" data-testid={`gateway-tool-${t.name}`}>
                    <span><code>{t.name}</code> {t.description}</span>
                    <button type="button" className="btn btn-sm btn-secondary" data-testid={`gateway-tool-edit-${t.name}`}
                        onClick={() => setDraft(toolToDraft(t, i))}>Edit</button>
                    <button type="button" className="btn btn-sm btn-danger" data-testid={`gateway-tool-delete-${t.name}`}
                        onClick={() => void remove(i)}>Delete</button>
                </div>
            ))}
            {!draft && (
                <button type="button" className="btn btn-sm btn-secondary" data-testid="gateway-tool-add"
                    onClick={() => setDraft(newDraft())}>Add tool</button>
            )}
            {draft && (
                <div className="gateway__tool" data-testid="gateway-tool-form">
                    <div className="gateway__row">
                        <label className="gateway__field">Name
                            <input type="text" value={draft.name} onChange={(e) => set({ name: e.target.value })} data-testid="gateway-tool-name" />
                        </label>
                        <label className="gateway__field">Model
                            <select value={draft.model} onChange={(e) => set({ model: e.target.value })} data-testid="gateway-tool-model">
                                <option value="">Project default</option>
                                <option value=":base">Base model</option>
                                {data.servable_runs.map((r) => <option key={r.id} value={`:run-${r.id}`}>{runLabel(r)}</option>)}
                            </select>
                        </label>
                        <label className="gateway__field">Max tokens
                            <input type="number" min={1} value={draft.maxTokens} onChange={(e) => set({ maxTokens: e.target.value })} data-testid="gateway-tool-max-tokens" />
                        </label>
                    </div>
                    <label className="gateway__field">Description
                        <input type="text" value={draft.description} onChange={(e) => set({ description: e.target.value })} data-testid="gateway-tool-description" />
                    </label>
                    <label className="gateway__field">Input schema (JSON)
                        <textarea rows={6} value={draft.schema} onChange={(e) => set({ schema: e.target.value })} data-testid="gateway-tool-schema" />
                    </label>
                    <label>
                        <input type="checkbox" checked={draft.template === '@recipe'} data-testid="gateway-tool-recipe"
                            onChange={(e) => set({ template: e.target.checked ? '@recipe' : '{input}' })} />
                        {' '}Use the recipe&apos;s prompt (@recipe)
                    </label>
                    <label className="gateway__field">Prompt template ({'{field}'} placeholders)
                        <textarea rows={3} value={draft.template} disabled={draft.template === '@recipe'}
                            onChange={(e) => set({ template: e.target.value })} data-testid="gateway-tool-template" />
                    </label>
                    <label className="gateway__field">System prompt (optional)
                        <textarea rows={2} value={draft.system} onChange={(e) => set({ system: e.target.value })} data-testid="gateway-tool-system" />
                    </label>
                    <label className="gateway__field">Retrieve passages
                        <select value={draft.retrieval} data-testid="gateway-tool-retrieval"
                            onChange={(e) => set({ retrieval: e.target.value as ToolDraft['retrieval'] })}>
                            <option value="default">Model default (on for passage-trained runs)</option>
                            <option value="on">Always on</option>
                            <option value="off">Always off</option>
                        </select>
                    </label>
                    {localError && <p className="gateway__error" role="alert" data-testid="gateway-tool-local-error">{localError}</p>}
                    <ErrorList message={message} errors={errors} />
                    <div className="gateway__row">
                        <button type="button" className="btn btn-primary" data-testid="gateway-tool-save" onClick={() => void saveDraft()}>Save tool</button>
                        <button type="button" className="btn btn-secondary" onClick={() => { setDraft(null); setLocalError(null); }}>Cancel</button>
                    </div>
                </div>
            )}
            {!draft && <ErrorList message={message} errors={errors} />}
        </fieldset>
    );
}

/* ---------- Telemetry ---------- */

function TelemetrySection({ telemetry }: { telemetry: GatewayTelemetry | null }) {
    const rows = telemetry?.summary.by_model ?? [];
    const recent = telemetry?.recent ?? [];
    return (
        <fieldset className="gateway__section" data-testid="gateway-telemetry">
            <h4>Recent requests</h4>
            {rows.length === 0 && recent.length === 0 ? (
                <p className="u-hint" data-testid="gateway-telemetry-empty">
                    No requests yet. Prompts and replies are never stored; only timings, status and token counts.
                </p>
            ) : (
                <>
                    <table className="gateway__table">
                        <thead><tr><th>Model</th><th>Requests</th><th>Error rate</th><th>p50 ms</th><th>p95 ms</th></tr></thead>
                        <tbody>
                            {rows.map((r) => (
                                <tr key={r.model_id}>
                                    <td><code>{r.model_id}</code></td><td>{r.requests}</td>
                                    <td>{(r.error_rate * 100).toFixed(1)}%</td><td>{r.p50_ms ?? '-'}</td><td>{r.p95_ms ?? '-'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <table className="gateway__table">
                        <thead><tr><th>Time</th><th>Model</th><th>Run</th><th>Status</th><th>ms</th><th>Retrieval</th><th>Guided JSON</th></tr></thead>
                        <tbody>
                            {recent.slice(0, 10).map((r, i) => (
                                <tr key={i}>
                                    <td>{r.ts}</td><td><code>{r.model_id}</code></td><td>{r.run_id ?? '-'}</td>
                                    <td>{r.status_code}</td><td>{r.latency_ms}</td>
                                    <td>{r.retrieval_applied ? 'yes' : 'no'}</td><td>{r.guided_json_mode ?? '-'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </>
            )}
        </fieldset>
    );
}

/* ---------- Card ---------- */

export default function GatewayCard({ projectId }: { projectId: number }) {
    const [data, setData] = useState<GatewaySettingsResponse | null>(null);
    const [snippets, setSnippets] = useState<GatewaySnippets | null>(null);
    const [telemetry, setTelemetry] = useState<GatewayTelemetry | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        getGatewaySettings(projectId)
            .then((d) => {
                if (cancelled) return;
                if (d?.settings && Array.isArray(d.servable_runs) && Array.isArray(d.model_ids)) setData(d);
                else setLoadError('Gateway settings unavailable');
            })
            .catch((err) => { if (!cancelled) setLoadError(parseGatewayError(err).message); });
        getGatewaySnippets(projectId).then((d) => { if (!cancelled) setSnippets(d); }).catch(() => undefined);
        getGatewayTelemetry(projectId).then((d) => { if (!cancelled) setTelemetry(d); }).catch(() => undefined);
        return () => { cancelled = true; };
    }, [projectId]);

    const refreshSnippets = (d: GatewaySettingsResponse) => {
        setData(d);
        getGatewaySnippets(projectId).then(setSnippets).catch(() => undefined);
    };

    if (loadError) return <section className="card" data-testid="gateway-card"><p className="gateway__error">{loadError}</p></section>;
    if (!data) return <section className="card" data-testid="gateway-card"><p className="u-hint">Loading gateway...</p></section>;

    const defaultRunId = data.settings.default_run_id ?? null;
    const resolvedRunId = data.default_run_resolved ?? null;
    const resolved = resolvedRunId ? data.servable_runs.find((r) => r.id === resolvedRunId) : undefined;
    const resolvesTo = data.default_resolves_to;
    const slugNote = resolvesTo?.kind === 'run' && resolvedRunId
        ? `${defaultRunId ? 'default run' : 'latest completed run'} #${resolvedRunId}`
        : resolvesTo?.kind === 'base_with_retrieval'
            ? 'base model + your passages (no trained run)'
            : 'no servable run yet';
    const runFor = (modelId: string): ServableRun | null => {
        const m = /:run-(\d+)$/.exec(modelId);
        if (m) return data.servable_runs.find((r) => r.id === Number(m[1])) ?? null;
        if (modelId === data.slug) return resolved ?? null;
        return null;
    };

    return (
        <section className="card" data-testid="gateway-card">
            <h3 className="u-card-title u-mb-sm">Use from Claude Code</h3>
            <p className="u-hint u-mt0 u-mb-md">
                Serve this project&apos;s model through an OpenAI-compatible endpoint and an MCP server, with retrieval and citations.
            </p>

            <fieldset className="gateway__section" data-testid="gateway-models">
                <h4>Model ids</h4>
                <ul className="gateway__list">
                    {data.model_ids.map((id) => {
                        const run = runFor(id);
                        const isBase = id.endsWith(':base');
                        const base = isBase ? data.base_model : run?.base_model;
                        return (
                            <li key={id} data-testid={`gateway-model-${id}`}>
                                <code>{id}</code> · project {projectId}
                                {isBase ? ' · base model, no adapter' : run ? ` · ${runLabel(run)}` : ''}
                                {isBase && base ? ` (${base})` : ''}
                                {id === data.slug && (
                                    <span data-testid="gateway-slug-note">
                                        {' '}· {slugNote}
                                    </span>
                                )}
                            </li>
                        );
                    })}
                </ul>
            </fieldset>

            {snippets && (
                <fieldset className="gateway__section" data-testid="gateway-snippets">
                    <h4>Snippets</h4>
                    <CodeBlock id="mcp" label=".mcp.json" text={JSON.stringify(snippets.mcp_json, null, 2)} />
                    <CodeBlock id="curl" label="curl" text={snippets.curl} />
                    <CodeBlock id="python" label="OpenAI Python" text={snippets.openai_python} />
                </fieldset>
            )}

            <KeysSection projectId={projectId} />
            <SettingsSection key={`s-${JSON.stringify(data.settings)}`} projectId={projectId} data={data} onSaved={refreshSnippets} />
            <ToolsSection projectId={projectId} data={data} onSaved={refreshSnippets} />
            <TelemetrySection telemetry={telemetry} />
        </section>
    );
}
