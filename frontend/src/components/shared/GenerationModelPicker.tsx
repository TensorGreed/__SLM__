/**
 * GenerationModelPicker — "which model writes this?" for the guided flows
 * (documents → Q&A, seed documents). Default = the first reachable local
 * model; or a cloud provider + model whose API key is saved on the project
 * (Data → Generate → "Save key to this project"). Emits the backend pin the
 * flow routes take: `null` (local) or `cloud:<provider>:<model>`.
 *
 * Reads the same catalog as the Synthetic tab's cloud picker
 * (`/synthetic/backends/cloud/models`), so the model names match.
 */

import { useEffect, useState } from 'react';

import { listCloudModels, type CloudProviderEntry } from '../../api/synthPlaybook';

interface Props {
    projectId: number;
    /** The current pin (`null` = local model). */
    value: string | null;
    onChange: (pin: string | null) => void;
    disabled?: boolean;
    /** Prefix for data-testids (one picker per card). */
    testIdPrefix: string;
}

export function cloudPin(provider: string, model: string): string {
    return `cloud:${provider}:${model}`;
}

export function parsePin(pin: string | null): { provider: string; model: string } | null {
    if (!pin || !pin.startsWith('cloud:')) return null;
    const [, provider, ...rest] = pin.split(':');
    const model = rest.join(':');
    return provider && model ? { provider, model } : null;
}

export default function GenerationModelPicker({ projectId, value, onChange, disabled, testIdPrefix }: Props) {
    const [providers, setProviders] = useState<CloudProviderEntry[]>([]);
    const parsed = parsePin(value);
    const [provider, setProvider] = useState<string>(parsed?.provider ?? '');

    useEffect(() => {
        let cancelled = false;
        listCloudModels(projectId)
            .then((resp) => {
                if (!cancelled) setProviders(resp.providers || []);
            })
            .catch(() => undefined);
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    const active = providers.find((p) => p.provider === provider) ?? null;

    return (
        <div className="u-row" data-testid={`${testIdPrefix}-model-picker`}>
            <label>
                Generation model{' '}
                <select
                    value={provider}
                    disabled={disabled}
                    onChange={(e) => {
                        setProvider(e.target.value);
                        onChange(null);
                    }}
                    data-testid={`${testIdPrefix}-provider`}
                >
                    <option value="">Local model (first reachable)</option>
                    {providers.map((p) => (
                        <option key={p.provider} value={p.provider}>
                            {p.provider}{p.key_saved ? ' (key saved)' : ' (no key saved)'}
                        </option>
                    ))}
                </select>
            </label>
            {active && (
                <select
                    value={parsed?.provider === active.provider ? parsed.model : ''}
                    disabled={disabled || !active.key_saved}
                    onChange={(e) => onChange(e.target.value ? cloudPin(active.provider, e.target.value) : null)}
                    aria-label={`${active.provider} model`}
                    data-testid={`${testIdPrefix}-cloud-model`}
                >
                    <option value="">Pick a model…</option>
                    {active.models.map((m) => (
                        <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                </select>
            )}
            {active && !active.key_saved && (
                <span className="u-hint" data-testid={`${testIdPrefix}-no-key`}>
                    No {active.provider} key saved on this project — save one on Data → Generate ("Save key to this project").
                </span>
            )}
        </div>
    );
}
