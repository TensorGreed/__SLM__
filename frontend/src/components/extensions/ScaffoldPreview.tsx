/**
 * ScaffoldPreview — display the generated scaffold files inline with
 * per-file download buttons (priority.md P40). Since the frontend has
 * no JSZip dependency, "download zip" is split into N file-by-file
 * blob downloads — same outcome, one extra click.
 */

import { useMemo, useState } from 'react';

import type { ScaffoldResponse } from '../../types/extensions';

interface Props {
    scaffold: ScaffoldResponse;
}

function downloadBlob(filename: string, content: string): void {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

export default function ScaffoldPreview({ scaffold }: Props) {
    const fileEntries = useMemo(
        () => Object.entries(scaffold.files),
        [scaffold.files],
    );
    const initialActive = fileEntries[0]?.[0] ?? null;
    const [activeFile, setActiveFile] = useState<string | null>(initialActive);
    const active = activeFile ?? fileEntries[0]?.[0] ?? null;
    const activeContent = active ? scaffold.files[active] : '';

    return (
        <div className="scaffold-preview">
            <div className="scaffold-preview-header">
                <div>
                    <div className="scaffold-preview-title">
                        <span className="badge badge-success">scaffold ready</span>
                        <code>{scaffold.plugin_id}</code>
                        <span className="dim">
                            ({scaffold.contract_version})
                        </span>
                    </div>
                    <div className="dim scaffold-preview-meta">
                        Output dir:{' '}
                        <code aria-label="scaffold output directory">
                            {scaffold.output_dir}
                        </code>
                        {scaffold.written_files.length > 0 && (
                            <>
                                {' · '}
                                {scaffold.written_files.length} file(s) written
                            </>
                        )}
                    </div>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                        for (const [filename, content] of fileEntries) {
                            downloadBlob(filename, content);
                        }
                    }}
                    aria-label="Download every scaffold file"
                >
                    Download all files
                </button>
            </div>

            <div className="scaffold-preview-body">
                <ul
                    className="scaffold-preview-tabs"
                    role="tablist"
                    aria-label="Scaffold files"
                >
                    {fileEntries.map(([filename]) => (
                        <li key={filename}>
                            <button
                                type="button"
                                role="tab"
                                aria-selected={filename === active}
                                className={`scaffold-preview-tab ${filename === active ? 'is-active' : ''}`}
                                onClick={() => setActiveFile(filename)}
                            >
                                {filename}
                            </button>
                        </li>
                    ))}
                </ul>
                <div className="scaffold-preview-pane" role="tabpanel">
                    <div className="scaffold-preview-pane-toolbar">
                        <span className="dim">{active ?? '—'}</span>
                        {active && (
                            <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => downloadBlob(active, activeContent)}
                            >
                                Download
                            </button>
                        )}
                    </div>
                    <pre className="scaffold-preview-code">
                        <code>{activeContent}</code>
                    </pre>
                </div>
            </div>
        </div>
    );
}
