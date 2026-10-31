/**
 * CheckedDecodingSettings (Epic Y4): the project's "Checked decoding" setting, shown with the
 * Results eval settings. Off by default. Saved per project, so test-example evals and lift checks
 * (both the base model and the fine-tuned run) use it. Backend: ``GET/PUT .../checked-decoding``.
 */

import { useEffect, useState } from 'react';

import {
    fetchCheckedDecoding,
    saveCheckedDecoding,
    type CheckedDecodingMode,
    type CheckedDecodingSettings as Settings,
} from '../../api/checkedDecoding';
import './CheckedDecoding.css';

interface Props {
    projectId: number;
}

const MODE_TEXT: Record<CheckedDecodingMode, string> = {
    resample: 'Try again (a new sampled answer)',
    repair: 'Ask the model to fix the listed problems',
};

function errorText(err: unknown): string {
    const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : 'Could not save the setting.';
}

export default function CheckedDecodingSettings({ projectId }: Props) {
    const [settings, setSettings] = useState<Settings | null>(null);
    const [limit, setLimit] = useState(5);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        fetchCheckedDecoding(projectId)
            .then((payload) => {
                if (!alive) return;
                setSettings(payload.settings);
                setLimit(payload.max_attempts_limit || 5);
            })
            .catch(() => {
                if (alive) setError('Could not load the checked-decoding setting.');
            });
        return () => {
            alive = false;
        };
    }, [projectId]);

    const save = async (patch: Partial<Settings>) => {
        if (!settings) return;
        const next = { ...settings, ...patch };
        setSettings(next);
        setSaving(true);
        setError(null);
        try {
            const payload = await saveCheckedDecoding(projectId, patch);
            setSettings(payload.settings);
        } catch (err) {
            setError(errorText(err));
        } finally {
            setSaving(false);
        }
    };

    if (!settings) {
        return error ? <p className="checked-decoding-settings__error" data-testid="checked-decoding-settings-error">{error}</p> : null;
    }

    return (
        <div className="form-group checked-decoding-settings" data-testid="checked-decoding-settings">
            <label className="eval-panel__toggle">
                <input
                    type="checkbox"
                    checked={settings.enabled}
                    disabled={saving}
                    onChange={(e) => void save({ enabled: e.target.checked })}
                    data-testid="checked-decoding-toggle"
                />
                Checked decoding
            </label>
            <p className="u-hint u-mt0">
                After the model answers a row, quick automatic checks look at its answer and the question only (never
                the reference answer): does the code parse, does the JSON match the declared fields, is the label one of
                the allowed ones, are the cited passages real. When a check fails, the row is tried again. Off by
                default; saved for this project, so test-example evals and lift checks (the base model and the
                fine-tuned run alike) use it. Results are labelled, and Compare will not pair an eval decoded this way
                with one that was not.
            </p>
            {settings.enabled && (
                <div className="checked-decoding-settings__row">
                    <div className="form-group checked-decoding-settings__field">
                        <label className="form-label" htmlFor="checked-decoding-mode">When a check fails</label>
                        <select
                            id="checked-decoding-mode"
                            className="input"
                            value={settings.mode}
                            disabled={saving}
                            onChange={(e) => void save({ mode: e.target.value as CheckedDecodingMode })}
                            data-testid="checked-decoding-mode"
                        >
                            <option value="resample">{MODE_TEXT.resample}</option>
                            <option value="repair">{MODE_TEXT.repair}</option>
                        </select>
                    </div>
                    <div className="form-group checked-decoding-settings__field">
                        <label className="form-label" htmlFor="checked-decoding-max">Max tries per row</label>
                        <input
                            id="checked-decoding-max"
                            className="input"
                            type="number"
                            min={1}
                            max={limit}
                            value={settings.max_attempts}
                            disabled={saving}
                            onChange={(e) => {
                                const value = Math.round(Number(e.target.value));
                                if (Number.isFinite(value) && value >= 1 && value <= limit) {
                                    void save({ max_attempts: value });
                                }
                            }}
                            data-testid="checked-decoding-max"
                        />
                    </div>
                </div>
            )}
            {settings.enabled && (
                <p className="u-hint u-mt0" data-testid="checked-decoding-summary">
                    The first answer is always the normal one; a retry happens only when a check fails, and if no try
                    passes the first answer is kept. Each retry adds generation time.
                </p>
            )}
            {error && <p className="checked-decoding-settings__error" data-testid="checked-decoding-settings-error">{error}</p>}
        </div>
    );
}
