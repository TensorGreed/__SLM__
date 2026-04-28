/**
 * ManifestExportButton — small reusable button that exports the project's
 * current state as a `brewslm.yaml` file (priority.md P24).
 *
 * GETs `/api/projects/{id}/manifest/export?format=yaml` and triggers a
 * client-side download named after the project. After a successful
 * download it briefly toggles `confirmation` so the parent can render an
 * inline "Exported" hint without needing toast plumbing.
 */

import { useCallback, useState } from 'react';
import { Download } from 'lucide-react';

import api from '../../api/client';

interface ManifestExportButtonProps {
    projectId: number;
    projectName: string;
    label?: string;
    className?: string;
    variant?: 'primary' | 'secondary' | 'ghost';
    size?: 'default' | 'sm';
    onExported?: () => void;
}

function slugifyProjectName(name: string): string {
    const cleaned = (name || 'project').trim().toLowerCase();
    return cleaned.replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'project';
}

function triggerDownload(content: string, filename: string): void {
    const blob = new Blob([content], { type: 'application/x-yaml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

export default function ManifestExportButton({
    projectId,
    projectName,
    label = 'Export YAML',
    className,
    variant = 'secondary',
    size = 'default',
    onExported,
}: ManifestExportButtonProps) {
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [confirmed, setConfirmed] = useState(false);

    const handleClick = useCallback(async () => {
        setBusy(true);
        setError(null);
        try {
            const response = await api.get<string>(
                `/projects/${projectId}/manifest/export`,
                {
                    params: { format: 'yaml' },
                    responseType: 'text',
                    transformResponse: [(data) => (typeof data === 'string' ? data : String(data ?? ''))],
                },
            );
            const body = typeof response.data === 'string' ? response.data : String(response.data ?? '');
            triggerDownload(body, `${slugifyProjectName(projectName)}.brewslm.yaml`);
            setConfirmed(true);
            onExported?.();
            window.setTimeout(() => setConfirmed(false), 2500);
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' && detail ? detail : 'Export failed.');
        } finally {
            setBusy(false);
        }
    }, [projectId, projectName, onExported]);

    const buttonClass = [
        'btn',
        variant === 'primary' ? 'btn-primary' : variant === 'ghost' ? 'btn-ghost' : 'btn-secondary',
        size === 'sm' ? 'btn-sm' : '',
        'manifest-export-button',
        className || '',
    ]
        .filter(Boolean)
        .join(' ');

    return (
        <span className="manifest-export-button-wrap">
            <button
                type="button"
                className={buttonClass}
                onClick={() => void handleClick()}
                disabled={busy}
                title="Download brewslm.yaml"
            >
                <Download size={14} />
                <span>{busy ? 'Exporting…' : label}</span>
            </button>
            {confirmed && (
                <span className="badge badge-success" role="status" style={{ marginLeft: 8 }}>
                    Exported
                </span>
            )}
            {error && (
                <span className="badge badge-danger" role="alert" style={{ marginLeft: 8 }}>
                    {error}
                </span>
            )}
        </span>
    );
}
