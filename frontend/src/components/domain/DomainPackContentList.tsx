/**
 * Domain packs that carry content: versions, counts, "create a project from
 * this pack", and "export this project as a pack".
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import {
    exportProjectPack,
    fetchPackContent,
    instantiatePack,
} from '../../api/domainPacks';
import type { IncludeDocuments, InstantiateSummary, PackContent } from '../../api/domainPacks';
import type { DomainPackSummary } from '../../types';
import './DomainPackImportPanel.css';

interface Props {
    /** Current project, for the export action. */
    projectId: number;
    /** Bump to reload (after an import). */
    refreshKey?: number;
}

function PackRow({ content }: { content: PackContent }) {
    const [name, setName] = useState('');
    const [version, setVersion] = useState(content.version);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [summary, setSummary] = useState<InstantiateSummary | null>(null);
    const s = content.summary;

    const create = async () => {
        setBusy(true);
        setError(null);
        try {
            const res = await instantiatePack(content.pack_id, {
                project_name: name.trim() || undefined,
                version,
            });
            setSummary(res.summary);
            window.location.assign(`/project/${res.project.id}`);
        } catch (err) {
            const e = err as { response?: { data?: { detail?: unknown } } };
            const detail = e.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Could not create the project from this pack.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <li className="dp-content__row" data-testid={`dp-content-${content.pack_id}`}>
            <div className="dp-content__head">
                <strong>{content.pack_id}@{content.version}</strong>
                <span className="dp-import__muted">
                    {s.taxonomy_entries} labels, {s.probes} checks ({s.critical_probes} critical),{' '}
                    {s.documents} documents, {s.gold_dev_rows + s.gold_test_rows} gold rows
                </span>
            </div>
            <div className="dp-content__create">
                <input
                    type="text"
                    value={name}
                    placeholder="New project name"
                    aria-label={`Project name for ${content.pack_id}`}
                    onChange={(e) => setName(e.target.value)}
                />
                <select
                    value={version}
                    aria-label={`Version of ${content.pack_id}`}
                    onChange={(e) => setVersion(e.target.value)}
                >
                    {content.versions.map((v) => (
                        <option key={v.version} value={v.version}>{v.version}</option>
                    ))}
                </select>
                <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy}
                    onClick={() => void create()}
                    data-testid={`dp-create-${content.pack_id}`}
                >
                    Create project from this pack
                </button>
            </div>
            {error && <p className="dp-import__error" role="alert">{error}</p>}
            {summary && summary.warnings.length > 0 && (
                <ul className="dp-import__issue-list">
                    {summary.warnings.map((w, n) => <li key={n}>{w.code}: {w.message}</li>)}
                </ul>
            )}
        </li>
    );
}

export default function DomainPackContentList({ projectId, refreshKey = 0 }: Props) {
    const [contents, setContents] = useState<PackContent[]>([]);
    const [loading, setLoading] = useState(true);
    const [includeDocs, setIncludeDocs] = useState<IncludeDocuments>('pack');
    const [exporting, setExporting] = useState(false);
    const [exportError, setExportError] = useState<string | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        try {
            const res = await api.get<{ packs: DomainPackSummary[] }>('/domain-packs');
            const found = await Promise.all(
                (res.data.packs || [])
                    .filter((p) => !p.is_system)
                    .map((p) => fetchPackContent(p.pack_id).catch(() => null)),
            );
            setContents(found.filter((c): c is PackContent => c !== null));
        } catch {
            setContents([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void load();
    }, [load, refreshKey]);

    const doExport = async () => {
        setExporting(true);
        setExportError(null);
        try {
            const { blob, filename } = await exportProjectPack(projectId, includeDocs);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch {
            setExportError('Could not export this project as a pack.');
        } finally {
            setExporting(false);
        }
    };

    return (
        <section className="dp-content card" data-testid="dp-content-list">
            <h3 className="dp-import__title">Packs with content</h3>
            {loading ? (
                <p className="dp-import__muted">Loading...</p>
            ) : contents.length === 0 ? (
                <p className="dp-import__muted" data-testid="dp-content-empty">
                    No imported pack carries content yet.
                </p>
            ) : (
                <ul className="dp-content__list">
                    {contents.map((c) => <PackRow key={c.pack_id} content={c} />)}
                </ul>
            )}
            <div className="dp-content__export">
                <select
                    value={includeDocs}
                    aria-label="Documents to include in the export"
                    data-testid="dp-export-docs"
                    onChange={(e) => setIncludeDocs(e.target.value as IncludeDocuments)}
                >
                    <option value="pack">Documents from packs</option>
                    <option value="all">All documents</option>
                    <option value="none">No documents</option>
                </select>
                <button
                    type="button"
                    className="btn btn-sm"
                    disabled={exporting}
                    onClick={() => void doExport()}
                    data-testid="dp-export"
                >
                    Export this project as a pack
                </button>
            </div>
            {exportError && <p className="dp-import__error" role="alert">{exportError}</p>}
        </section>
    );
}
