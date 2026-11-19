/**
 * AutoLiftSettingCard — per-project switch for the automatic base-vs-fine-tuned check that runs when
 * a training run finishes. On by default. The server-wide AUTO_LIFT_EVAL_ENABLED switch wins when it
 * is off (the checkbox is then disabled and says why). A check started by hand is never affected.
 */

import { useCallback, useEffect, useState } from 'react';

import { getAutoLiftSetting, putAutoLiftSetting, type AutoLiftSetting } from '../../api/autoLift';
import { toast } from '../../stores/toastStore';

export default function AutoLiftSettingCard({ projectId }: { projectId: number }) {
    const [state, setState] = useState<AutoLiftSetting | null>(null);
    const [loadError, setLoadError] = useState(false);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        let cancelled = false;
        getAutoLiftSetting(projectId)
            .then((res) => { if (!cancelled) setState(res); })
            .catch(() => { if (!cancelled) setLoadError(true); });
        return () => { cancelled = true; };
    }, [projectId]);

    const save = useCallback(async (enabled: boolean | null) => {
        setSaving(true);
        try {
            setState(await putAutoLiftSetting(projectId, enabled));
        } catch {
            toast.error('Could not save the automatic check setting.', 4000);
        } finally {
            setSaving(false);
        }
    }, [projectId]);

    if (loadError) return <p className="u-hint" data-testid="al-error">Could not load the automatic check setting.</p>;
    if (!state) return <p className="u-hint">Loading the automatic check setting…</p>;

    const globalOff = !state.global_enabled;
    const projectOn = state.enabled !== false;

    return (
        <div className="ts-config" data-testid="al-config">
            <label className="ts-config__row">
                <input
                    type="checkbox"
                    checked={projectOn && !globalOff}
                    disabled={saving || globalOff}
                    onChange={(e) => { void save(e.target.checked); }}
                    data-testid="al-enabled"
                />
                Run the base-vs-fine-tuned check automatically when training finishes
            </label>
            <p className="u-hint u-mt0" data-testid="al-status">
                {globalOff
                    ? 'Off for every project: the server setting AUTO_LIFT_EVAL_ENABLED is false and overrides this switch.'
                    : state.effective
                        ? 'On: a finished run is compared with its base model on the held-out test rows.'
                        : 'Off for this project: finished runs are not compared automatically. You can still run the check by hand from the Results page.'}
                {!globalOff && state.enabled !== null && (
                    <>
                        {' '}
                        <button
                            type="button"
                            className="btn btn-link"
                            disabled={saving}
                            onClick={() => { void save(null); }}
                            data-testid="al-reset"
                        >
                            Use the default
                        </button>
                    </>
                )}
            </p>
        </div>
    );
}
