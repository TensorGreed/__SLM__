/**
 * RemediationConfigCard — how a code-remediation project's patches are
 * judged: the detector that decides "finding resolved", the code checks run
 * on patched code (and which of them are usable on this machine), the judge,
 * and passage retrieval for rows without their own passages.
 *
 * Shown on the Train page for `code-remediation` projects. Saved settings
 * apply at dataset prep, held-out eval and in the synthetic playbooks.
 */

import { useCallback, useEffect, useState } from 'react';

import { checkStatus, listCodeChecks, type CodeCheckInfo, type NetworkIsolationInfo } from '../../api/codeChecks';
import {
    detectorOptionLabel,
    getRemediationConfig,
    putRemediationConfig,
    type RemediationConfig,
    type RemediationConfigResponse,
    type RemediationDetector,
} from '../../api/remediation';
import { toast } from '../../stores/toastStore';
import './RemediationConfigCard.css';

interface Props {
    projectId: number;
}

const STATUS_TEXT = { ready: 'ready', not_installed: 'not installed', not_allowed: 'not allowed' } as const;

function detectorKey(d: RemediationDetector | null): string {
    if (!d) return '';
    return d.kind === 'project' ? `project:${d.project_id}:${d.experiment_id ?? ''}` : `command:${d.check_id}`;
}

function keyToDetector(key: string): RemediationDetector | null {
    if (key.startsWith('project:')) {
        const [, pid, exp] = key.split(':');
        return { kind: 'project', project_id: Number(pid), experiment_id: exp ? Number(exp) : null };
    }
    if (key.startsWith('command:')) return { kind: 'command', check_id: key.slice('command:'.length) };
    return null;
}

/** The select value for the stored detector. A project detector stored without a run id follows the
 *  project's latest run, so it maps to that project's available option when there is one. */
function resolveDetectorKey(res: RemediationConfigResponse): string {
    const stored = res.config.detector ?? null;
    if (stored?.kind === 'project' && stored.experiment_id == null) {
        const match = res.available_detectors.find((d) => d.project_id === stored.project_id);
        if (match) return `project:${match.project_id}:${match.experiment_id}`;
    }
    return detectorKey(stored);
}

function errorText(err: unknown): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: string } } })?.response?.data;
    return data?.message ?? (typeof data?.detail === 'string' ? data.detail : 'Could not save the settings.');
}

export default function RemediationConfigCard({ projectId }: Props) {
    const [state, setState] = useState<RemediationConfigResponse | null>(null);
    const [checks, setChecks] = useState<CodeCheckInfo[]>([]);
    const [isolation, setIsolation] = useState<NetworkIsolationInfo | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [detector, setDetector] = useState('');
    const [allChecks, setAllChecks] = useState(true);
    const [picked, setPicked] = useState<string[]>([]);
    const [judge, setJudge] = useState(true);
    const [retrieve, setRetrieve] = useState(true);
    const [topK, setTopK] = useState(3);
    const [scoreExamples, setScoreExamples] = useState(false);
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);

    const apply = useCallback((res: RemediationConfigResponse) => {
        setState(res);
        setDetector(resolveDetectorKey(res));
        setAllChecks(res.config.checks == null);
        setPicked(res.config.checks ?? []);
        setJudge(res.config.judge ?? res.effective.judge);
        setRetrieve(res.config.retrieve_passages ?? res.effective.retrieve_passages);
        setTopK(res.config.top_k ?? res.effective.top_k);
        setScoreExamples(res.config.score_code_examples ?? res.effective.score_code_examples ?? false);
    }, []);

    useEffect(() => {
        let cancelled = false;
        getRemediationConfig(projectId)
            .then((res) => { if (!cancelled) apply(res); })
            .catch(() => { if (!cancelled) setLoadError('Could not load the remediation settings.'); });
        listCodeChecks()
            .then((res) => {
                if (cancelled) return;
                setChecks(res.checks ?? []);
                setIsolation(res.network_isolation ?? null);
            })
            .catch(() => { /* the list is informational */ });
        return () => { cancelled = true; };
    }, [projectId, apply]);

    const save = async () => {
        setSaving(true);
        setSaveError(null);
        const body: RemediationConfig = {
            detector: keyToDetector(detector),
            checks: allChecks ? null : picked,
            judge,
            retrieve_passages: retrieve,
            top_k: topK,
            // PUT replaces the settings: leaving it out stores the default (off).
            ...(scoreExamples ? { score_code_examples: true } : {}),
        };
        try {
            apply(await putRemediationConfig(projectId, body));
            toast.success('Remediation settings saved.', 3000);
        } catch (err) {
            setSaveError(errorText(err));
        } finally {
            setSaving(false);
        }
    };

    if (loadError) return <p className="u-hint" data-testid="remediation-config-error">{loadError}</p>;
    if (!state) return <p className="u-hint">Loading remediation settings…</p>;

    const commandDetectors = checks.filter((c) => c.kind === 'detector' && checkStatus(c) === 'ready');
    const usableChecks = checks.filter((c) => c.kind !== 'detector');
    const projectLabel = state.detector_label;
    const knownKeys = new Set<string>([
        '',
        ...state.available_detectors.map((d) => `project:${d.project_id}:${d.experiment_id}`),
        ...commandDetectors.map((c) => `command:${c.id}`),
    ]);
    // A stored detector that no option matches (an older run, or one that is no longer usable) is
    // still shown, so the select reflects what is stored.
    const storedDetector = keyToDetector(detector);
    const extraOptionLabel = detector && !knownKeys.has(detector) && storedDetector
        ? (projectLabel
            ?? (storedDetector.kind === 'project'
                ? `project ${storedDetector.project_id} · run ${storedDetector.experiment_id ?? 'latest'} (not available)`
                : `command · ${storedDetector.check_id} (not available)`))
        : null;
    const isolationOff = isolation != null && !isolation.available;

    return (
        <div className="remediation-config" data-testid="remediation-config">
            <p className="u-hint u-mt0">
                These settings apply when patches are prepared, scored and generated. A patch is checked by applying it
                to the code, running the code checks, and asking the detector whether the finding is gone.
            </p>

            <fieldset className="remediation-config__group">
                <legend>Detector (is the finding resolved?)</legend>
                <div className="remediation-config__row">
                    <label htmlFor="rem-detector">Detector</label>
                    <select id="rem-detector" value={detector} onChange={(e) => setDetector(e.target.value)} data-testid="rem-detector-select">
                        <option value="">None (resolution not evaluated)</option>
                        {state.available_detectors.map((d) => (
                            <option
                                key={`${d.project_id}:${d.experiment_id}`}
                                value={`project:${d.project_id}:${d.experiment_id}`}
                            >
                                {detectorOptionLabel(d)}
                            </option>
                        ))}
                        {commandDetectors.map((c) => (
                            <option key={c.id} value={`command:${c.id}`}>command · {c.id}</option>
                        ))}
                        {extraOptionLabel && (
                            <option value={detector} data-testid="rem-detector-stored-option">{extraOptionLabel}</option>
                        )}
                    </select>
                </div>
                {projectLabel && <p className="u-hint u-mt0" data-testid="rem-detector-label">Current detector: {projectLabel}</p>}
                {state.available_detectors.length === 0 && (
                    <p className="u-hint u-mt0" data-testid="rem-no-detectors">
                        No code-findings project has a finished run with weights yet, so no model detector is available.
                    </p>
                )}
            </fieldset>

            <fieldset className="remediation-config__group">
                <legend>Code checks on patched code</legend>
                <label className="remediation-config__row">
                    <input type="checkbox" checked={allChecks} onChange={(e) => setAllChecks(e.target.checked)} data-testid="rem-all-checks" />
                    Run every ready check that applies to the language
                </label>
                <ul className="remediation-config__checks" data-testid="rem-checks">
                    {usableChecks.map((c) => {
                        const status = checkStatus(c);
                        return (
                            <li key={c.id} data-testid={`rem-check-${c.id}`}>
                                {!allChecks && (
                                    <input
                                        type="checkbox"
                                        aria-label={`Use check ${c.id}`}
                                        checked={picked.includes(c.id)}
                                        disabled={status !== 'ready'}
                                        onChange={(e) => setPicked(e.target.checked ? [...picked, c.id] : picked.filter((x) => x !== c.id))}
                                    />
                                )}
                                <code>{c.id}</code>
                                <span>{c.languages.join(', ')}</span>
                                <span className={`remediation-config__status ${status === 'ready' ? 'is-ready' : 'is-off'}`}>
                                    {STATUS_TEXT[status]}
                                </span>
                                {isolationOff && c.kind !== 'builtin' && status === 'ready' && (
                                    <span className="remediation-config__status is-off" data-testid={`rem-check-${c.id}-network`}>
                                        network not isolated
                                    </span>
                                )}
                            </li>
                        );
                    })}
                </ul>
                {isolation && (
                    <p className="u-hint" data-testid="rem-isolation">
                        External checks run sandboxed. Network isolation: {isolation.mode}
                        {isolation.available
                            ? ' (available)'
                            : ' (not available on this machine: external checks run without network isolation; set it to "require" to refuse to run them instead)'}.
                    </p>
                )}
            </fieldset>

            <fieldset className="remediation-config__group">
                <legend>Judge and passages</legend>
                <label className="remediation-config__row">
                    <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} data-testid="rem-judge-toggle" />
                    Score patches with the judge model when one is configured
                </label>
                <label className="remediation-config__row">
                    <input type="checkbox" checked={retrieve} onChange={(e) => setRetrieve(e.target.checked)} data-testid="rem-retrieve-toggle" />
                    Retrieve reference passages for rows that have none
                </label>
                <div className="remediation-config__row">
                    <label htmlFor="rem-top-k">Passages per row</label>
                    <input
                        id="rem-top-k"
                        type="number"
                        min={1}
                        max={10}
                        value={topK}
                        onChange={(e) => setTopK(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                        data-testid="rem-top-k"
                    />
                </div>
            </fieldset>

            <fieldset className="remediation-config__group">
                <legend>Held-out eval</legend>
                <label className="remediation-config__row">
                    <input type="checkbox" checked={scoreExamples} onChange={(e) => setScoreExamples(e.target.checked)} data-testid="rem-score-code-examples" />
                    Score task→code rows
                </label>
                <p className="u-hint u-mt0">
                    Off: held-out task→code rows are skipped. On: they pass when the code parses, passes the code checks,
                    cites only given passages and the judge calls it correct.
                </p>
            </fieldset>

            {saveError && <p className="remediation-config__error" data-testid="rem-save-error">{saveError}</p>}
            <div className="remediation-config__row">
                <button type="button" className="btn btn-primary" onClick={save} disabled={saving} data-testid="rem-save">
                    {saving ? 'Saving…' : 'Save settings'}
                </button>
            </div>
        </div>
    );
}
