/**
 * ValidatePanel — run the P37 contract suite against a module path
 * supplied by the operator. Surfaces each check (module_importable,
 * module_interface, schema_compliance, version_metadata, safe_reload)
 * so the operator can see exactly what failed. ``onForceReload`` is
 * for the rare case where the module is already imported and changes
 * landed on disk.
 */

import { useCallback, useMemo, useState } from 'react';

import type {
    PluginContractReport,
    PluginKind,
} from '../../types/extensions';
import { PLUGIN_KIND_ALIAS } from '../../types/extensions';
import CommandSnippet, { type ApiSnippet } from '../shared/CommandSnippet';

interface Props {
    kind: PluginKind;
    onValidate: (
        body: { kind: PluginKind; module: string; force_reload: boolean },
    ) => Promise<PluginContractReport>;
}

export default function ValidatePanel({ kind, onValidate }: Props) {
    const [module, setModule] = useState('');
    const [forceReload, setForceReload] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [report, setReport] = useState<PluginContractReport | null>(null);
    const [error, setError] = useState<string | null>(null);

    const snippet = useMemo(() => {
        const moduleName = module.trim() || '<module-path>';
        const alias = PLUGIN_KIND_ALIAS[kind];
        const cliParts = [
            `brewslm extensions validate`,
            `  --kind ${alias}`,
            `  --module ${moduleName}`,
        ];
        if (forceReload) cliParts.push('  --force-reload');
        const api: ApiSnippet = {
            method: 'POST',
            path: '/extensions/validate',
            body: {
                kind,
                module: moduleName,
                force_reload: forceReload,
            },
        };
        return { cli: cliParts.join(' \\\n'), api };
    }, [forceReload, kind, module]);

    const handleSubmit = useCallback(
        async (event: React.FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            const moduleName = module.trim();
            if (!moduleName) {
                setError('Module path is required.');
                return;
            }
            setSubmitting(true);
            setError(null);
            try {
                const result = await onValidate({
                    kind,
                    module: moduleName,
                    force_reload: forceReload,
                });
                setReport(result);
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Validate failed.';
                setError(message);
                setReport(null);
            } finally {
                setSubmitting(false);
            }
        },
        [forceReload, kind, module, onValidate],
    );

    return (
        <div className="extension-validate-panel">
            <form className="extension-validate-form" onSubmit={handleSubmit}>
                <div className="scaffold-form-row scaffold-form-row-inline">
                    <label className="scaffold-form-label" htmlFor="validate-module">
                        Importable Python module path
                    </label>
                    <input
                        id="validate-module"
                        className="input"
                        type="text"
                        placeholder="example.plugins.my_adapter"
                        value={module}
                        onChange={(event) => setModule(event.target.value)}
                        required
                        autoComplete="off"
                    />
                </div>
                <div className="scaffold-form-row scaffold-form-row-inline">
                    <label
                        className="scaffold-form-label"
                        htmlFor="validate-force-reload"
                    >
                        <input
                            id="validate-force-reload"
                            type="checkbox"
                            checked={forceReload}
                            onChange={(event) => setForceReload(event.target.checked)}
                        />{' '}
                        Force reload if already imported
                    </label>
                </div>
                <div className="scaffold-form-actions">
                    <button
                        type="submit"
                        className="btn btn-primary btn-sm"
                        disabled={submitting}
                    >
                        {submitting ? 'Validating…' : 'Validate'}
                    </button>
                </div>
                <CommandSnippet cli={snippet.cli} api={snippet.api} />
            </form>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            {report && (
                <div
                    className={`extension-validate-report ${report.ok ? 'is-ok' : 'is-error'}`}
                    aria-label="Contract validation report"
                >
                    <div className="extension-validate-report-head">
                        <span
                            className={`badge ${report.ok ? 'badge-success' : 'badge-danger'}`}
                        >
                            {report.ok ? 'contract ok' : 'contract failed'}
                        </span>
                        <code>{report.module}</code>
                        <span className="dim">{report.contract_version}</span>
                        {report.declared_version && (
                            <span className="dim">
                                declared {report.declared_version}
                            </span>
                        )}
                    </div>
                    {report.declared_ids.length > 0 && (
                        <div className="dim extension-validate-report-ids">
                            Declared ids:{' '}
                            {report.declared_ids.map((id) => (
                                <code key={id}>{id}</code>
                            ))}
                        </div>
                    )}
                    <ul className="extension-validate-checks">
                        {report.checks.map((check) => (
                            <li
                                key={check.name}
                                className={`extension-validate-check ${check.ok ? 'is-ok' : 'is-error'}`}
                            >
                                <span
                                    className={`badge ${check.ok ? 'badge-success' : 'badge-danger'}`}
                                >
                                    {check.ok ? 'pass' : 'fail'}
                                </span>
                                <code>{check.name}</code>
                                <span className="dim">{check.message}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
