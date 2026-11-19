/**
 * NullAsAbsentSetting: the opt-in "null counts as absent" JSON scoring setting, inside a closed
 * "More options" disclosure of the eval run form (nothing changes until someone opens it).
 *
 * - Project setting (saved): every eval and lift check of this project scores with it.
 * - This run: follow the project setting (default), or turn it on / off for this eval only.
 * Backend: `null_scoring` + `GET/PUT .../settings/null-as-absent`; the run value goes in the eval payload.
 */
import { useEffect, useState } from 'react';

import { fetchNullAsAbsent, saveNullAsAbsent } from '../../api/nullScoring';

/** '' = follow the project setting; 'on' / 'off' = override for this run. */
export type NullRunChoice = '' | 'on' | 'off';

interface Props {
    projectId: number;
    runChoice: NullRunChoice;
    onRunChoice: (choice: NullRunChoice) => void;
}

/** The eval payload field for a run choice (undefined = follow the project). */
export function nullAsAbsentPayload(choice: NullRunChoice): boolean | undefined {
    return choice === 'on' ? true : choice === 'off' ? false : undefined;
}

export default function NullAsAbsentSetting({ projectId, runChoice, onRunChoice }: Props) {
    const [open, setOpen] = useState(false);
    const [enabled, setEnabled] = useState<boolean | null>(null);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!open || enabled !== null) return;
        let alive = true;
        fetchNullAsAbsent(projectId)
            .then((s) => { if (alive) setEnabled(Boolean(s.enabled)); })
            .catch(() => { if (alive) setError('Could not load the setting.'); });
        return () => { alive = false; };
    }, [open, enabled, projectId]);

    const save = async (next: boolean) => {
        setSaving(true);
        setError(null);
        setEnabled(next);
        try {
            const s = await saveNullAsAbsent(projectId, next ? true : null);
            setEnabled(Boolean(s.enabled));
        } catch {
            setError('Could not save the setting.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <details
            className="eval-collapsible"
            data-testid="null-as-absent-more-options"
            onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
        >
            <summary>More options: JSON scoring (null counts as absent)</summary>
            <p className="u-hint u-mt0">
                For answers in JSON (extraction, tool-call arguments): count <code>&quot;key&quot;: null</code> the same as
                the key being left out when comparing with the answer key. Off by default. Results scored with it are
                labelled, and Compare and the lift check do not pair them with results scored without it.
            </p>
            {enabled !== null && (
                <label className="eval-panel__toggle">
                    <input
                        type="checkbox"
                        checked={enabled}
                        disabled={saving}
                        onChange={(e) => void save(e.target.checked)}
                        data-testid="null-as-absent-project-toggle"
                    />
                    Project setting: null counts as absent
                </label>
            )}
            <div className="form-group">
                <label className="form-label" htmlFor="null-as-absent-run">This eval</label>
                <select
                    id="null-as-absent-run"
                    className="input"
                    value={runChoice}
                    onChange={(e) => onRunChoice(e.target.value as NullRunChoice)}
                    data-testid="null-as-absent-run-select"
                >
                    <option value="">Use the project setting{enabled === null ? '' : enabled ? ' (on)' : ' (off)'}</option>
                    <option value="on">On for this eval</option>
                    <option value="off">Off for this eval</option>
                </select>
            </div>
            {error && <p className="u-hint u-mt0" data-testid="null-as-absent-error">{error}</p>}
        </details>
    );
}
