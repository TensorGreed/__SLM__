/**
 * GenerationModelPicker — "which model writes this?" for the guided flows
 * (documents → Q&A, seed documents). Default = the first reachable local
 * model; or a cloud provider + model whose API key is saved on the project
 * (Data → Generate → "Save key to this project"). Emits the backend pin the
 * flow routes take: `null` (local), `cloud:<provider>:<model>`, or the
 * local-CLI subscription backend's pin (`command`, `command:<model>`).
 *
 * Reads the same catalog as the Synthetic tab's cloud picker
 * (`/synthetic/backends/cloud/models`), so the model names match.
 */

import { useEffect, useState } from 'react';

import { listCloudModels, listSynthBackends, type CloudProviderEntry, type SynthBackendInfo } from '../../api/synthPlaybook';

interface Props {
    projectId: number;
    /** The current pin (`null` = local model). */
    value: string | null;
    onChange: (pin: string | null) => void;
    disabled?: boolean;
    /** Prefix for data-testids (one picker per card). */
    testIdPrefix: string;
    /** The field's label (default "Generation model"). */
    label?: string;
    /** What the `null` (no cloud pin) choice means for this picker. */
    localOptionLabel?: string;
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

/** The `provider` select value standing for the local-CLI subscription backend. */
const SUBSCRIPTION_OPTION = '__subscription';

export default function GenerationModelPicker({
    projectId, value, onChange, disabled, testIdPrefix, label = 'Generation model', localOptionLabel = 'Local model (first reachable)',
}: Props) {
    const [providers, setProviders] = useState<CloudProviderEntry[]>([]);
    const parsed = parsePin(value);
    const [cli, setCli] = useState<SynthBackendInfo | null>(null);
    const [provider, setProvider] = useState<string>(parsed?.provider ?? '');
    const cliSelected = cli !== null && value === cli.describe;

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

    useEffect(() => {
        let cancelled = false;
        listSynthBackends(projectId)
            .then((resp) => {
                if (cancelled) return;
                const found = (resp?.backends ?? []).find((b) => b.available && b.billing === 'subscription');
                setCli(found ?? null);
            })
            .catch(() => undefined);
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    const selectValue = cliSelected ? SUBSCRIPTION_OPTION : provider;
    const active = selectValue === SUBSCRIPTION_OPTION ? null : providers.find((p) => p.provider === provider) ?? null;

    return (
        <div className="u-row" data-testid={`${testIdPrefix}-model-picker`}>
            <label>
                {label}{' '}
                <select
                    value={selectValue}
                    disabled={disabled}
                    onChange={(e) => {
                        if (e.target.value === SUBSCRIPTION_OPTION && cli) {
                            setProvider('');
                            onChange(cli.describe);
                            return;
                        }
                        setProvider(e.target.value);
                        onChange(null);
                    }}
                    data-testid={`${testIdPrefix}-provider`}
                >
                    <option value="">{localOptionLabel}</option>
                    {cli && (
                        <option value={SUBSCRIPTION_OPTION}>
                            Local CLI (subscription){cli.detail ? ` — ${cli.detail}` : ''}
                        </option>
                    )}
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
            {cliSelected && (
                <span className="u-hint" data-testid={`${testIdPrefix}-subscription-note`}>
                    Runs on {cli?.describe} — subscription, not billed per call and not counted against the API budget.
                    Prompts leave this machine for the CLI's service.
                </span>
            )}
            {active && !active.key_saved && (
                <span className="u-hint" data-testid={`${testIdPrefix}-no-key`}>
                    No {active.provider} key saved on this project — save one on Data → Generate ("Save key to this project").
                </span>
            )}
        </div>
    );
}
