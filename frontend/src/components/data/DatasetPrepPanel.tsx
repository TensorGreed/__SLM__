import { useEffect, useState } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { toast } from '../../stores/toastStore';
import './DatasetPrepPanel.css';

interface DatasetPrepPanelProps {
    projectId: number;
    onNextStep: () => void | Promise<void>;
}

interface ProfileResult {
    source: Record<string, unknown>;
    profile: {
        total_records: number;
        normalized_records: number;
        dropped_records: number;
        normalization_coverage: number;
        top_fields: [string, number][];
        text_length: { avg_chars: number; p50_chars: number; max_chars: number };
    };
    sample_records: Record<string, unknown>[];
    normalized_preview: Record<string, unknown>[];
    domain_hooks?: {
        normalizer?: { id?: string };
        validator?: { id?: string };
        evaluator?: { id?: string };
    };
    validator_report?: Record<string, unknown>;
}

interface SplitManifest {
    project_id: number;
    total_entries: number;
    splits: Record<string, number>;
    file_paths?: Record<string, string>;
    files?: Record<string, string>;
    chat_template: string;
    created_at: string;
    domain_pack_applied?: string | null;
    domain_pack_source?: string | null;
    domain_profile_applied?: string | null;
    domain_profile_source?: string | null;
    profile_split_defaults?: Record<string, unknown> | null;
    resolved_split_config?: {
        train_ratio?: number;
        val_ratio?: number;
        test_ratio?: number;
        seed?: number;
        chat_template?: string;
    } | null;
    profile_defaults_applied?: string[];
}

interface SplitEffectiveConfig {
    domain_pack_applied?: string | null;
    domain_pack_source?: string | null;
    domain_profile_applied?: string | null;
    domain_profile_source?: string | null;
    profile_split_defaults?: Record<string, unknown> | null;
    resolved_split_config?: {
        train_ratio?: number;
        val_ratio?: number;
        test_ratio?: number;
        seed?: number;
        chat_template?: string;
    } | null;
    profile_defaults_applied?: string[];
    include_types?: string[] | null;
}

const CHAT_TEMPLATES = [
    { value: 'llama3', label: 'Llama 3' },
    { value: 'chatml', label: 'ChatML' },
    { value: 'zephyr', label: 'Zephyr' },
    { value: 'phi3', label: 'Phi-3' },
];

export default function DatasetPrepPanel({ projectId, onNextStep }: DatasetPrepPanelProps) {
    // Preview state
    const [previewEntries, setPreviewEntries] = useState<Record<string, unknown>[]>([]);
    const [previewTotal, setPreviewTotal] = useState(0);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewTemplate, setPreviewTemplate] = useState('llama3');

    // Profile state
    const [profile, setProfile] = useState<ProfileResult | null>(null);
    const [profileLoading, setProfileLoading] = useState(false);
    const [profileType, setProfileType] = useState('cleaned');

    // Split state
    const [trainRatio, setTrainRatio] = useState(0.8);
    const [valRatio, setValRatio] = useState(0.1);
    const [testRatio, setTestRatio] = useState(0.1);
    const [splitSeed, setSplitSeed] = useState(42);
    const [splitTemplate, setSplitTemplate] = useState('llama3');
    const [useProfileDefaults, setUseProfileDefaults] = useState(true);
    const [splitTouched, setSplitTouched] = useState({
        train_ratio: false,
        val_ratio: false,
        test_ratio: false,
        seed: false,
        chat_template: false,
    });
    const [splitLoading, setSplitLoading] = useState(false);
    const [splitManifest, setSplitManifest] = useState<SplitManifest | null>(null);
    const [effectiveSplitConfig, setEffectiveSplitConfig] = useState<SplitEffectiveConfig | null>(null);
    const [effectiveSplitLoading, setEffectiveSplitLoading] = useState(false);
    const [effectiveSplitError, setEffectiveSplitError] = useState('');

    const buildSplitPayload = (): Record<string, unknown> => {
        const payload: Record<string, unknown> = {};
        if (!useProfileDefaults || splitTouched.train_ratio) payload.train_ratio = trainRatio;
        if (!useProfileDefaults || splitTouched.val_ratio) payload.val_ratio = valRatio;
        if (!useProfileDefaults || splitTouched.test_ratio) payload.test_ratio = testRatio;
        if (!useProfileDefaults || splitTouched.seed) payload.seed = splitSeed;
        if (!useProfileDefaults || splitTouched.chat_template) payload.chat_template = splitTemplate;
        return payload;
    };

    const previewEffectiveSplitConfig = async () => {
        setEffectiveSplitLoading(true);
        setEffectiveSplitError('');
        try {
            const payload = buildSplitPayload();
            const res = await api.post<SplitEffectiveConfig>(
                `/projects/${projectId}/dataset/split/effective-config`,
                payload,
            );
            setEffectiveSplitConfig(res.data);
        } catch (err: any) {
            setEffectiveSplitConfig(null);
            setEffectiveSplitError(err?.response?.data?.detail || 'Failed to preview effective split config');
        } finally {
            setEffectiveSplitLoading(false);
        }
    };

    useEffect(() => {
        void previewEffectiveSplitConfig();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    // ── Preview ────────────────────────────────────────────────
    const loadPreview = async () => {
        setPreviewLoading(true);
        try {
            const res = await api.get(`/projects/${projectId}/dataset/preview`, {
                params: { limit: 50, chat_template: previewTemplate },
            });
            setPreviewEntries(res.data.preview || []);
            setPreviewTotal(res.data.total || 0);
        } catch {
            setPreviewEntries([]);
            setPreviewTotal(0);
        } finally {
            setPreviewLoading(false);
        }
    };

    // ── Profile ────────────────────────────────────────────────
    const loadProfile = async () => {
        setProfileLoading(true);
        try {
            const res = await api.post(`/projects/${projectId}/dataset/profile`, {
                dataset_type: profileType,
                sample_size: 500,
            });
            setProfile(res.data);
        } catch {
            setProfile(null);
        } finally {
            setProfileLoading(false);
        }
    };

    // ── Split ──────────────────────────────────────────────────
    const runSplit = async () => {
        setSplitLoading(true);
        try {
            const payload = buildSplitPayload();

            const res = await api.post<SplitManifest>(`/projects/${projectId}/dataset/split`, payload);
            const manifest = res.data;
            setSplitManifest(manifest);

            const resolved = manifest.resolved_split_config || {};
            if (typeof resolved.train_ratio === 'number') {
                setTrainRatio(resolved.train_ratio);
            }
            if (typeof resolved.val_ratio === 'number') {
                setValRatio(resolved.val_ratio);
            }
            if (typeof resolved.test_ratio === 'number') {
                setTestRatio(resolved.test_ratio);
            }
            if (typeof resolved.seed === 'number') {
                setSplitSeed(resolved.seed);
            }
            if (typeof resolved.chat_template === 'string') {
                setSplitTemplate(resolved.chat_template);
            }
            setEffectiveSplitConfig({
                domain_pack_applied: manifest.domain_pack_applied,
                domain_pack_source: manifest.domain_pack_source,
                domain_profile_applied: manifest.domain_profile_applied,
                domain_profile_source: manifest.domain_profile_source,
                profile_split_defaults: manifest.profile_split_defaults,
                resolved_split_config: manifest.resolved_split_config,
                profile_defaults_applied: manifest.profile_defaults_applied,
                include_types: null,
            });
            setSplitTouched({
                train_ratio: false,
                val_ratio: false,
                test_ratio: false,
                seed: false,
                chat_template: false,
            });
        } catch (err: any) {
            const msg = err?.response?.data?.detail || 'Split failed. Make sure you have ingested and processed data.';
            setSplitManifest(null);
            toast.error(msg);
        } finally {
            setSplitLoading(false);
        }
    };

    // ── Source breakdown ────────────────────────────────────────
    const sourceBreakdown = () => {
        const counts: Record<string, number> = {};
        for (const entry of previewEntries) {
            const src = String((entry as any)._source_dataset || 'unknown');
            counts[src] = (counts[src] || 0) + 1;
        }
        return Object.entries(counts);
    };

    const ratioSum = +(trainRatio + valRatio + testRatio).toFixed(4);

    return (
        <div className="dataprep-panel">
            {/* ── Preview Section ──────────────────────────────── */}
            <div className="dp-section">
                <h3><span className="icon">👁️</span> Dataset Preview</h3>
                <div className="dp-info">
                    <span className="info-icon">ℹ️</span>
                    Preview combines all available data sources (cleaned, synthetic, gold, raw) into a unified view before splitting.
                </div>
                <div className="dp-actions">
                    <select value={previewTemplate} onChange={e => setPreviewTemplate(e.target.value)}>
                        {CHAT_TEMPLATES.map(t => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                        ))}
                    </select>
                    <button className="btn-primary" onClick={loadPreview} disabled={previewLoading}>
                        {previewLoading ? '⏳ Loading...' : '🔍 Load Preview'}
                    </button>
                    {previewTotal > 0 && <span style={{ color: 'rgba(255,255,255,.5)', fontSize: '.85rem' }}>
                        {previewTotal} total entries
                    </span>}
                </div>

                {previewEntries.length > 0 && (
                    <>
                        <div className="dp-stats-grid">
                            <div className="dp-stat">
                                <div className="label">Total Entries</div>
                                <div className="value">{previewTotal}</div>
                            </div>
                            {sourceBreakdown().map(([src, count]) => (
                                <div className="dp-stat" key={src}>
                                    <div className="label">{src}</div>
                                    <div className="value">{count}</div>
                                </div>
                            ))}
                        </div>
                        <div className="dp-table-wrap">
                            <table className="dp-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Source</th>
                                        <th>Text / Q&A</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {previewEntries.slice(0, 50).map((entry: any, i) => (
                                        <tr key={i}>
                                            <td>{i + 1}</td>
                                            <td>
                                                <span className={`dp-source-badge ${entry._source_dataset || ''}`}>
                                                    {entry._source_dataset || '—'}
                                                </span>
                                            </td>
                                            <td title={entry.text || entry.question || ''}>
                                                {entry.question
                                                    ? `Q: ${entry.question?.slice(0, 80)}… → A: ${entry.answer?.slice(0, 60)}…`
                                                    : (entry.text || '').slice(0, 140)
                                                }
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </>
                )}
            </div>

            {/* ── Profile Section ──────────────────────────────── */}
            <div className="dp-section">
                <h3><span className="icon">📊</span> Schema Profile</h3>
                <div className="dp-actions">
                    <select value={profileType} onChange={e => setProfileType(e.target.value)}>
                        <option value="raw">Raw</option>
                        <option value="cleaned">Cleaned</option>
                        <option value="synthetic">Synthetic</option>
                        <option value="gold_dev">Gold Dev</option>
                    </select>
                    <button className="btn-primary" onClick={loadProfile} disabled={profileLoading}>
                        {profileLoading ? '⏳ Profiling...' : '📋 Run Profile'}
                    </button>
                </div>

                {profile && (
                    <>
                        <div className="dp-stats-grid">
                            <div className="dp-stat">
                                <div className="label">Total Records</div>
                                <div className="value">{profile.profile.total_records}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Normalized</div>
                                <div className="value">{profile.profile.normalized_records}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Dropped</div>
                                <div className="value" style={{ color: profile.profile.dropped_records > 0 ? '#ff6b6b' : undefined }}>
                                    {profile.profile.dropped_records}
                                </div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Coverage</div>
                                <div className="value">{profile.profile.normalization_coverage}%</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Avg Length</div>
                                <div className="value">{Math.round(profile.profile.text_length.avg_chars)}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">P50 Length</div>
                                <div className="value">{Math.round(profile.profile.text_length.p50_chars)}</div>
                            </div>
                        </div>

                        {profile.profile.top_fields.length > 0 && (
                            <>
                                <h4 style={{ fontSize: '.85rem', color: 'rgba(255,255,255,.5)', margin: '1rem 0 .5rem' }}>Field Frequency</h4>
                                <div className="dp-field-bars">
                                    {profile.profile.top_fields.slice(0, 12).map(([field, count]) => (
                                        <div className="dp-field-bar" key={field}>
                                            <span className="field-name">{field}</span>
                                            <div className="bar-track">
                                                <div
                                                    className="bar-fill"
                                                    style={{ width: `${(count / profile.profile.total_records) * 100}%` }}
                                                />
                                            </div>
                                            <span className="bar-count">{count}</span>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}

                        <div className="dp-resolved-panel" style={{ marginTop: '1rem' }}>
                            <div className="dp-resolved-title">Hook Validation</div>
                            <div className="dp-resolved-kv">
                                <span>Normalizer Hook</span>
                                <strong>{profile.domain_hooks?.normalizer?.id || 'default-normalizer'}</strong>
                            </div>
                            <div className="dp-resolved-kv">
                                <span>Validator Hook</span>
                                <strong>{profile.domain_hooks?.validator?.id || 'default-validator'}</strong>
                            </div>
                            <div className="dp-resolved-kv">
                                <span>Evaluator Hook</span>
                                <strong>{profile.domain_hooks?.evaluator?.id || 'default-evaluator'}</strong>
                            </div>
                            <pre className="dp-resolved-json">
                                {JSON.stringify(profile.validator_report || {}, null, 2)}
                            </pre>
                        </div>
                    </>
                )}
            </div>

            {/* ── Split Section ─────────────────────────────────── */}
            <div className="dp-section">
                <h3><span className="icon">✂️</span> Train / Val / Test Split</h3>
                <div className="dp-info">
                    <span className="info-icon">💡</span>
                    Splits your combined dataset into training, validation, and test JSONL files. Ratios must sum to 1.0.
                </div>
                <div className="dp-split-default-toggle">
                    <label>
                        <input
                            type="checkbox"
                            checked={useProfileDefaults}
                            onChange={(e) => setUseProfileDefaults(e.target.checked)}
                        />
                        Use active domain runtime defaults for fields left untouched
                    </label>
                </div>
                <div className="dp-split-row">
                    <div className="dp-split-field">
                        <label>Train Ratio</label>
                        <input type="number" step="0.05" min="0.1" max="0.95" value={trainRatio}
                            onChange={e => {
                                setTrainRatio(+e.target.value);
                                setSplitTouched((prev) => ({ ...prev, train_ratio: true }));
                            }} />
                    </div>
                    <div className="dp-split-field">
                        <label>Val Ratio</label>
                        <input type="number" step="0.05" min="0" max="0.5" value={valRatio}
                            onChange={e => {
                                setValRatio(+e.target.value);
                                setSplitTouched((prev) => ({ ...prev, val_ratio: true }));
                            }} />
                    </div>
                    <div className="dp-split-field">
                        <label>Test Ratio</label>
                        <input type="number" step="0.05" min="0" max="0.5" value={testRatio}
                            onChange={e => {
                                setTestRatio(+e.target.value);
                                setSplitTouched((prev) => ({ ...prev, test_ratio: true }));
                            }} />
                    </div>
                    <div className="dp-split-field">
                        <label>Seed</label>
                        <input
                            type="number"
                            value={splitSeed}
                            onChange={e => {
                                setSplitSeed(+e.target.value);
                                setSplitTouched((prev) => ({ ...prev, seed: true }));
                            }}
                        />
                    </div>
                    <div className="dp-split-field">
                        <label>Chat Template</label>
                        <select
                            value={splitTemplate}
                            onChange={e => {
                                setSplitTemplate(e.target.value);
                                setSplitTouched((prev) => ({ ...prev, chat_template: true }));
                            }}
                        >
                            {CHAT_TEMPLATES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                        </select>
                    </div>
                </div>
                {Math.abs(ratioSum - 1.0) > 0.001 && (
                    <p style={{ color: '#ff6b6b', fontSize: '.82rem', margin: '0 0 .75rem' }}>
                        ⚠️ Ratios sum to {ratioSum} — must equal 1.0
                    </p>
                )}
                <div className="dp-actions">
                    <button className="btn-primary" onClick={previewEffectiveSplitConfig} disabled={effectiveSplitLoading}>
                        {effectiveSplitLoading ? '⏳ Resolving...' : '🧭 Preview Effective Config'}
                    </button>
                    <button className="btn-primary" onClick={runSplit}
                        disabled={splitLoading || Math.abs(ratioSum - 1.0) > 0.001}>
                        {splitLoading ? '⏳ Splitting...' : '✂️ Run Split'}
                    </button>
                </div>
                {effectiveSplitError && (
                    <p style={{ color: '#ff6b6b', fontSize: '.82rem', margin: '0 0 .75rem' }}>{effectiveSplitError}</p>
                )}
                {effectiveSplitConfig && (
                    <div className="dp-resolved-panel">
                        <div className="dp-resolved-title">Effective Config Preview (Pre-run)</div>
                        <div className="dp-resolved-kv">
                            <span>Applied Pack</span>
                            <strong>
                                {effectiveSplitConfig.domain_pack_applied
                                    ? `${effectiveSplitConfig.domain_pack_applied} (${effectiveSplitConfig.domain_pack_source || 'unknown'})`
                                    : 'none'}
                            </strong>
                        </div>
                        <div className="dp-resolved-kv">
                            <span>Applied Profile</span>
                            <strong>
                                {effectiveSplitConfig.domain_profile_applied
                                    ? `${effectiveSplitConfig.domain_profile_applied} (${effectiveSplitConfig.domain_profile_source || 'unknown'})`
                                    : 'none'}
                            </strong>
                        </div>
                        <div className="dp-resolved-kv">
                            <span>Runtime Fields Applied</span>
                            <strong>
                                {effectiveSplitConfig.profile_defaults_applied && effectiveSplitConfig.profile_defaults_applied.length > 0
                                    ? effectiveSplitConfig.profile_defaults_applied.join(', ')
                                    : 'none'}
                            </strong>
                        </div>
                        <div className="dp-resolved-grid">
                            <div>
                                <div className="dp-resolved-subtitle">Resolved Split Config</div>
                                <pre className="dp-resolved-json">
                                    {JSON.stringify(effectiveSplitConfig.resolved_split_config || {}, null, 2)}
                                </pre>
                            </div>
                            <div>
                                <div className="dp-resolved-subtitle">Runtime Split Defaults</div>
                                <pre className="dp-resolved-json">
                                    {JSON.stringify(effectiveSplitConfig.profile_split_defaults || {}, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </div>
                )}

                {splitManifest && (
                    <div className="dp-manifest">
                        <h4>✅ Split Complete</h4>
                        <div className="dp-resolved-panel">
                            <div className="dp-resolved-title">Resolved Defaults</div>
                            <div className="dp-resolved-kv">
                                <span>Applied Pack</span>
                                <strong>
                                    {splitManifest.domain_pack_applied
                                        ? `${splitManifest.domain_pack_applied} (${splitManifest.domain_pack_source || 'unknown'})`
                                        : 'none'}
                                </strong>
                            </div>
                            <div className="dp-resolved-kv">
                                <span>Applied Profile</span>
                                <strong>
                                    {splitManifest.domain_profile_applied
                                        ? `${splitManifest.domain_profile_applied} (${splitManifest.domain_profile_source || 'unknown'})`
                                        : 'none'}
                                </strong>
                            </div>
                            <div className="dp-resolved-kv">
                                <span>Runtime Fields Applied</span>
                                <strong>
                                    {splitManifest.profile_defaults_applied && splitManifest.profile_defaults_applied.length > 0
                                        ? splitManifest.profile_defaults_applied.join(', ')
                                        : 'none'}
                                </strong>
                            </div>
                            <div className="dp-resolved-grid">
                                <div>
                                    <div className="dp-resolved-subtitle">Resolved Split Config</div>
                                    <pre className="dp-resolved-json">
                                        {JSON.stringify(splitManifest.resolved_split_config || {}, null, 2)}
                                    </pre>
                                </div>
                                <div>
                                    <div className="dp-resolved-subtitle">Runtime Split Defaults</div>
                                    <pre className="dp-resolved-json">
                                        {JSON.stringify(splitManifest.profile_split_defaults || {}, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        </div>
                        <div className="dp-stats-grid">
                            <div className="dp-stat">
                                <div className="label">Total</div>
                                <div className="value">{splitManifest.total_entries}</div>
                            </div>
                            {Object.entries(splitManifest.splits || {}).map(([split, count]) => (
                                <div className="dp-stat" key={split}>
                                    <div className="label">{split}</div>
                                    <div className="value">{count as number}</div>
                                </div>
                            ))}
                        </div>
                        <pre>{JSON.stringify(splitManifest, null, 2)}</pre>
                    </div>
                )}
            </div>

            <StepFooter
                currentStep="Dataset Preparation"
                nextStep="Tokenization"
                nextStepIcon="🔤"
                isComplete={!!splitManifest}
                hint="Split your dataset before continuing"
                onNext={onNextStep}
            />
        </div>
    );
}
