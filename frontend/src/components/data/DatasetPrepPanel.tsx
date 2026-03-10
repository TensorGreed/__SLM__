import { useEffect, useState } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { toast } from '../../stores/toastStore';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';
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

interface AdapterCatalogResponse {
    default_adapter: string;
    contract_version?: string;
    supported_task_profiles?: string[];
    adapters: Record<string, {
        description?: string;
        source?: string;
        is_default?: boolean;
        schema_hint?: Record<string, unknown>;
        contract?: {
            version?: string;
            adapter_id?: string;
            task_profiles?: string[];
            preferred_training_tasks?: string[];
            output_contract?: Record<string, unknown>;
        };
    }>;
    loaded_plugin_modules?: string[];
    plugin_load_errors?: Record<string, string>;
}

interface AdapterPreviewResult {
    project_id: number;
    source: Record<string, unknown>;
    requested_adapter_id: string;
    resolved_adapter_id: string;
    requested_task_profile?: string | null;
    resolved_task_profile?: string | null;
    task_profile_compatible?: boolean;
    compatibility_warnings?: string[];
    adapter_contract?: {
        version?: string;
        adapter_id?: string;
        task_profiles?: string[];
        preferred_training_tasks?: string[];
        output_contract?: Record<string, unknown>;
    };
    detection_scores: Record<string, number>;
    sampled_records: number;
    mapped_records: number;
    dropped_records: number;
    error_count: number;
    errors: Array<{ index: number; error: string }>;
    validation_report: Record<string, unknown>;
    conformance_report?: {
        sampled_records?: number;
        mapped_records?: number;
        mapping_success_rate?: number;
        required_fields?: string[];
        optional_fields?: string[];
        required_field_coverage?: Record<string, { present?: number; missing?: number; ratio?: number }>;
        optional_field_coverage?: Record<string, { present?: number; missing?: number; ratio?: number }>;
        required_fields_below_100?: string[];
        contract_pass?: boolean;
        failing_examples?: Array<{ mapped_index?: number; missing_required_fields?: string[] }>;
    };
    auto_fix_suggestions?: Array<{
        suggestion_id?: string;
        kind?: string;
        severity?: string;
        message?: string;
        confidence?: number;
        confidence_label?: string;
        expected_coverage_gain?: number;
        suggested_field_mapping?: Record<string, string>;
        top_raw_fields?: Array<[string, number]>;
    }>;
    auto_apply?: {
        confidence_threshold?: number;
        candidate_count?: number;
        applied_count?: number;
        suggested_field_mapping?: Record<string, string>;
        applied_suggestion_ids?: string[];
        min_confidence?: number | null;
    };
    inferred_task_profiles?: string[];
    raw_field_frequency?: Record<string, number>;
    preview_rows: Array<{
        index: number;
        raw: Record<string, unknown>;
        mapped: Record<string, unknown>;
    }>;
}

interface AdapterPreferenceResponse {
    project_id: number;
    source: string;
    adapter_id: string;
    adapter_config: Record<string, unknown>;
    field_mapping: Record<string, string>;
    task_profile?: string | null;
    domain_pack_applied?: string | null;
    domain_profile_applied?: string | null;
}

interface AdapterPreferenceAutoDetectResponse {
    preference: AdapterPreferenceResponse;
    preview: AdapterPreviewResult;
    saved: boolean;
}

interface SemanticIntelligenceResult {
    project_id: number;
    source?: {
        split?: string;
        source?: string;
    };
    sample_size_analyzed?: number;
    cluster_count?: number;
    semantic_diversity_score?: number;
    redundancy_ratio?: number;
    average_nearest_similarity?: number;
    suggestions?: string[];
    clusters?: Array<{
        cluster_id: number;
        size: number;
        share: number;
        representatives?: Array<{
            sample_index?: number;
            similarity_to_nearest?: number;
            text_preview?: string;
        }>;
    }>;
    report_path?: string;
}

const CHAT_TEMPLATES = [
    { value: 'llama3', label: 'Llama 3' },
    { value: 'chatml', label: 'ChatML' },
    { value: 'zephyr', label: 'Zephyr' },
    { value: 'phi3', label: 'Phi-3' },
];

type DatasetPrepView = 'overview' | 'adapters' | 'split';

const DEFAULT_TASK_PROFILE_OPTIONS = [
    'auto',
    'instruction_sft',
    'chat_sft',
    'qa',
    'rag_qa',
    'tool_calling',
    'structured_extraction',
    'summarization',
    'seq2seq',
    'classification',
    'preference',
    'language_modeling',
];

function parseJsonObjectInput(raw: string): { value: Record<string, unknown>; error: string } {
    const text = raw.trim();
    if (!text) {
        return { value: {}, error: '' };
    }
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return { value: {}, error: 'JSON config must be an object (e.g. {"key":"value"}).' };
        }
        return { value: parsed as Record<string, unknown>, error: '' };
    } catch (error) {
        if (error instanceof Error) {
            return { value: {}, error: error.message };
        }
        return { value: {}, error: 'Invalid JSON.' };
    }
}

function parseJsonStringMapInput(raw: string): { value: Record<string, string>; error: string } {
    const parsed = parseJsonObjectInput(raw);
    if (parsed.error) {
        return { value: {}, error: parsed.error };
    }
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed.value)) {
        const left = String(key || '').trim();
        const right = String(value ?? '').trim();
        if (!left || !right) {
            continue;
        }
        out[left] = right;
    }
    return { value: out, error: '' };
}

function normalizeStringMap(mapping: Record<string, unknown> | Record<string, string>): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [rawKey, rawValue] of Object.entries(mapping || {})) {
        const key = String(rawKey || '').trim();
        const value = String(rawValue ?? '').trim();
        if (!key || !value) {
            continue;
        }
        out[key] = value;
    }
    return out;
}

function asPercent(value: unknown): string {
    const numberValue = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(numberValue)) {
        return '0.0%';
    }
    return `${(numberValue * 100).toFixed(1)}%`;
}

export default function DatasetPrepPanel({ projectId, onNextStep }: DatasetPrepPanelProps) {
    const [activeView, setActiveView] = useState<DatasetPrepView>('overview');

    // Preview state
    const [previewEntries, setPreviewEntries] = useState<Record<string, unknown>[]>([]);
    const [previewTotal, setPreviewTotal] = useState(0);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewTemplate, setPreviewTemplate] = useState('llama3');

    // Profile state
    const [profile, setProfile] = useState<ProfileResult | null>(null);
    const [profileLoading, setProfileLoading] = useState(false);
    const [profileType, setProfileType] = useState('cleaned');
    const [semanticTargetSplit, setSemanticTargetSplit] = useState('train');
    const [semanticSampleSize, setSemanticSampleSize] = useState(400);
    const [semanticClusterCount, setSemanticClusterCount] = useState('');
    const [semanticThreshold, setSemanticThreshold] = useState('0.92');
    const [semanticLoading, setSemanticLoading] = useState(false);
    const [semanticError, setSemanticError] = useState('');
    const [semanticResult, setSemanticResult] = useState<SemanticIntelligenceResult | null>(null);
    const [semanticPrefillStage, setSemanticPrefillStage] = useState('');

    // Adapter preview state
    const [adapterCatalog, setAdapterCatalog] = useState<AdapterCatalogResponse | null>(null);
    const [adapterType, setAdapterType] = useState('raw');
    const [adapterId, setAdapterId] = useState('auto');
    const [adapterTaskProfile, setAdapterTaskProfile] = useState('auto');
    const [adapterConfigText, setAdapterConfigText] = useState('');
    const [adapterFieldMappingText, setAdapterFieldMappingText] = useState('');
    const [adapterSampleSize, setAdapterSampleSize] = useState(200);
    const [adapterPreviewLimit, setAdapterPreviewLimit] = useState(20);
    const [adapterPreviewLoading, setAdapterPreviewLoading] = useState(false);
    const [adapterPreviewError, setAdapterPreviewError] = useState('');
    const [adapterPreviewResult, setAdapterPreviewResult] = useState<AdapterPreviewResult | null>(null);
    const [adapterPreference, setAdapterPreference] = useState<AdapterPreferenceResponse | null>(null);
    const [adapterPreferenceLoading, setAdapterPreferenceLoading] = useState(false);

    // Split state
    const [trainRatio, setTrainRatio] = useState(0.8);
    const [valRatio, setValRatio] = useState(0.1);
    const [testRatio, setTestRatio] = useState(0.1);
    const [splitSeed, setSplitSeed] = useState(42);
    const [splitTemplate, setSplitTemplate] = useState('llama3');
    const [splitAdapterId, setSplitAdapterId] = useState('default-canonical');
    const [splitTaskProfile, setSplitTaskProfile] = useState('auto');
    const [splitAdapterConfigText, setSplitAdapterConfigText] = useState('');
    const [splitFieldMappingText, setSplitFieldMappingText] = useState('');
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
        payload.adapter_id = splitAdapterId || 'default-canonical';
        if (splitTaskProfile && splitTaskProfile !== 'auto') {
            payload.task_profile = splitTaskProfile;
        }
        const parsed = parseJsonObjectInput(splitAdapterConfigText);
        if (parsed.error) {
            payload.__adapter_config_error = parsed.error;
            return payload;
        }
        const parsedFieldMapping = parseJsonStringMapInput(splitFieldMappingText);
        if (parsedFieldMapping.error) {
            payload.__field_mapping_error = parsedFieldMapping.error;
            return payload;
        }
        if (Object.keys(parsed.value).length > 0) {
            payload.adapter_config = parsed.value;
        }
        if (Object.keys(parsedFieldMapping.value).length > 0) {
            payload.field_mapping = parsedFieldMapping.value;
        }
        return payload;
    };

    const previewEffectiveSplitConfig = async () => {
        setEffectiveSplitLoading(true);
        setEffectiveSplitError('');
        try {
            const payload = buildSplitPayload();
            const adapterConfigError = payload.__adapter_config_error;
            const fieldMappingError = payload.__field_mapping_error;
            delete payload.__adapter_config_error;
            delete payload.__field_mapping_error;
            delete payload.adapter_id;
            delete payload.adapter_config;
            delete payload.field_mapping;
            if (typeof adapterConfigError === 'string' && adapterConfigError.trim()) {
                setEffectiveSplitError(`Split adapter config JSON error: ${adapterConfigError}`);
                setEffectiveSplitConfig(null);
                return;
            }
            if (typeof fieldMappingError === 'string' && fieldMappingError.trim()) {
                setEffectiveSplitError(`Split field mapping JSON error: ${fieldMappingError}`);
                setEffectiveSplitConfig(null);
                return;
            }
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

    const loadAdapterPreference = async () => {
        setAdapterPreferenceLoading(true);
        try {
            const res = await api.get<AdapterPreferenceResponse>(`/projects/${projectId}/dataset/adapter-preference`);
            const pref = res.data;
            setAdapterPreference(pref);
            if (pref?.adapter_id) {
                setSplitAdapterId(pref.adapter_id);
            }
            setSplitTaskProfile(pref?.task_profile ? String(pref.task_profile) : 'auto');
            setAdapterTaskProfile(pref?.task_profile ? String(pref.task_profile) : 'auto');
            setSplitAdapterConfigText(
                pref?.adapter_config && Object.keys(pref.adapter_config).length > 0
                    ? JSON.stringify(pref.adapter_config, null, 2)
                    : '',
            );
            setSplitFieldMappingText(
                pref?.field_mapping && Object.keys(pref.field_mapping).length > 0
                    ? JSON.stringify(pref.field_mapping, null, 2)
                    : '',
            );
        } catch {
            setAdapterPreference(null);
        } finally {
            setAdapterPreferenceLoading(false);
        }
    };

    useEffect(() => {
        void previewEffectiveSplitConfig();
        void loadAdapterCatalog();
        void loadAdapterPreference();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    useEffect(() => {
        let cancelled = false;
        const applySemanticPrefill = async () => {
            const prefill = await loadWorkflowStagePrefill(projectId, ['semantic_curation']);
            if (cancelled || !prefill) {
                return;
            }
            const cfg = prefill.config || {};
            const splitToken = String(cfg.target_split || '').trim().toLowerCase();
            if (splitToken) {
                if (splitToken === 'val') {
                    setSemanticTargetSplit('validation');
                } else {
                    setSemanticTargetSplit(splitToken);
                }
            }
            const sampleValue = Number(cfg.sample_size);
            if (Number.isFinite(sampleValue) && sampleValue > 0) {
                setSemanticSampleSize(Math.max(20, Math.min(2000, Math.round(sampleValue))));
            }
            const clusterValue = Number(cfg.cluster_count);
            if (Number.isFinite(clusterValue) && clusterValue > 1) {
                setSemanticClusterCount(String(Math.max(2, Math.min(64, Math.round(clusterValue)))));
            }
            const thresholdValue = Number(cfg.similarity_threshold);
            if (Number.isFinite(thresholdValue) && thresholdValue >= 0.5) {
                setSemanticThreshold(String(Math.max(0.5, Math.min(0.999, thresholdValue))));
            }
            setSemanticPrefillStage(prefill.stage);
        };
        void applySemanticPrefill();
        return () => {
            cancelled = true;
        };
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

    const runSemanticIntelligence = async () => {
        setSemanticLoading(true);
        setSemanticError('');
        try {
            const thresholdValue = Number(semanticThreshold);
            const parsedClusterCount = Number.parseInt(semanticClusterCount, 10);
            const res = await api.post<SemanticIntelligenceResult>(
                `/projects/${projectId}/dataset/semantic-intelligence/analyze`,
                {
                    target_split: semanticTargetSplit,
                    sample_size: Math.max(20, Math.min(2000, semanticSampleSize || 20)),
                    cluster_count: Number.isFinite(parsedClusterCount) && parsedClusterCount > 1
                        ? parsedClusterCount
                        : undefined,
                    similarity_threshold: Number.isFinite(thresholdValue)
                        ? Math.max(0.5, Math.min(0.999, thresholdValue))
                        : 0.92,
                },
            );
            setSemanticResult(res.data);
        } catch (err: any) {
            setSemanticResult(null);
            setSemanticError(err?.response?.data?.detail || 'Semantic intelligence analysis failed');
        } finally {
            setSemanticLoading(false);
        }
    };

    // ── Adapter Preview ────────────────────────────────────────
    async function loadAdapterCatalog() {
        try {
            const res = await api.get<AdapterCatalogResponse>(`/projects/${projectId}/dataset/adapters/catalog`);
            setAdapterCatalog(res.data);
            if (res.data?.default_adapter) {
                setAdapterId((prev) => prev || res.data.default_adapter);
                setSplitAdapterId((prev) => prev || res.data.default_adapter);
            }
        } catch {
            setAdapterCatalog(null);
        }
    }

    const runAdapterPreview = async () => {
        setAdapterPreviewLoading(true);
        setAdapterPreviewError('');
        try {
            const parsedConfig = parseJsonObjectInput(adapterConfigText);
            if (parsedConfig.error) {
                setAdapterPreviewError(`Adapter config JSON error: ${parsedConfig.error}`);
                setAdapterPreviewResult(null);
                return;
            }
            const parsedFieldMapping = parseJsonStringMapInput(adapterFieldMappingText);
            if (parsedFieldMapping.error) {
                setAdapterPreviewError(`Field mapping JSON error: ${parsedFieldMapping.error}`);
                setAdapterPreviewResult(null);
                return;
            }
            const res = await api.post<AdapterPreviewResult>(`/projects/${projectId}/dataset/adapters/preview`, {
                dataset_type: adapterType,
                sample_size: adapterSampleSize,
                adapter_id: adapterId,
                task_profile: adapterTaskProfile !== 'auto' ? adapterTaskProfile : undefined,
                adapter_config: Object.keys(parsedConfig.value).length > 0 ? parsedConfig.value : undefined,
                field_mapping: Object.keys(parsedFieldMapping.value).length > 0 ? parsedFieldMapping.value : undefined,
                preview_limit: adapterPreviewLimit,
            });
            setAdapterPreviewResult(res.data);
        } catch (err: any) {
            setAdapterPreviewResult(null);
            setAdapterPreviewError(err?.response?.data?.detail || 'Adapter preview failed');
        } finally {
            setAdapterPreviewLoading(false);
        }
    };

    const emitMappingAcceptanceTelemetry = async (payload: {
        mode: 'single' | 'batch';
        mapping: Record<string, string>;
        suggestionCount?: number;
        confidenceAvg?: number;
        acceptedSuggestionIds?: string[];
    }) => {
        try {
            await api.post(`/projects/${projectId}/dataset/adapters/mapping-acceptance`, {
                mode: payload.mode,
                source: 'adapter_preview',
                adapter_id: adapterPreviewResult?.resolved_adapter_id || adapterId || undefined,
                task_profile: adapterPreviewResult?.resolved_task_profile || (adapterTaskProfile !== 'auto' ? adapterTaskProfile : undefined),
                suggestion_count: payload.suggestionCount,
                confidence_avg: payload.confidenceAvg,
                mapping: payload.mapping,
                accepted_suggestion_ids: payload.acceptedSuggestionIds,
            });
        } catch {
            // Best-effort telemetry should not block UX.
        }
    };

    const applySuggestedFieldMapping = (
        mapping: Record<string, string>,
        options?: {
            mode?: 'single' | 'batch';
            suggestionCount?: number;
            confidenceAvg?: number;
            acceptedSuggestionIds?: string[];
        },
    ) => {
        const nextMapping = normalizeStringMap(mapping);
        if (Object.keys(nextMapping).length === 0) {
            toast.error('Suggested mapping is empty.');
            return;
        }

        const parsedCurrent = parseJsonStringMapInput(adapterFieldMappingText);
        if (parsedCurrent.error) {
            setAdapterPreviewError(`Field mapping JSON error: ${parsedCurrent.error}`);
            return;
        }
        const merged = {
            ...parsedCurrent.value,
            ...nextMapping,
        };
        let changedCount = 0;
        for (const [key, value] of Object.entries(nextMapping)) {
            if (parsedCurrent.value[key] !== value) {
                changedCount += 1;
            }
        }
        if (changedCount === 0) {
            toast.success('Suggested mappings were already applied.');
            return;
        }

        const mergedText = JSON.stringify(merged, null, 2);
        setAdapterFieldMappingText(mergedText);
        setSplitFieldMappingText(mergedText);
        setAdapterPreviewError('');

        const mode = options?.mode || 'single';
        if (mode === 'batch') {
            toast.success(`Applied ${changedCount} suggested mappings to adapter and split field mappings.`);
        } else {
            toast.success('Applied suggested mapping to adapter and split field mappings.');
        }
        void emitMappingAcceptanceTelemetry({
            mode,
            mapping: nextMapping,
            suggestionCount: options?.suggestionCount,
            confidenceAvg: options?.confidenceAvg,
            acceptedSuggestionIds: options?.acceptedSuggestionIds,
        });
    };

    const applyAllSuggestedFieldMappings = () => {
        if (!adapterPreviewResult) {
            toast.error('Run adapter preview before applying suggestions.');
            return;
        }
        const suggestions = Array.isArray(adapterPreviewResult.auto_fix_suggestions)
            ? adapterPreviewResult.auto_fix_suggestions
            : [];
        const confidenceById = new Map<string, number>();
        for (const suggestion of suggestions) {
            const id = String(suggestion?.suggestion_id || '').trim();
            const confidence = Number(suggestion?.confidence);
            if (id && Number.isFinite(confidence)) {
                confidenceById.set(id, confidence);
            }
        }

        const preferredMapping = normalizeStringMap(
            (adapterPreviewResult.auto_apply?.suggested_field_mapping || {}) as Record<string, unknown>,
        );
        const preferredIds = Array.isArray(adapterPreviewResult.auto_apply?.applied_suggestion_ids)
            ? adapterPreviewResult.auto_apply?.applied_suggestion_ids?.map((id) => String(id || '').trim()).filter(Boolean) || []
            : [];

        let mergedMapping = { ...preferredMapping };
        let acceptedSuggestionIds = [...preferredIds];
        if (Object.keys(mergedMapping).length === 0) {
            const threshold = Number(adapterPreviewResult.auto_apply?.confidence_threshold ?? 0.72);
            const usedRawFields = new Set<string>();
            for (const suggestion of suggestions) {
                const suggested = normalizeStringMap((suggestion?.suggested_field_mapping || {}) as Record<string, unknown>);
                if (Object.keys(suggested).length === 0) {
                    continue;
                }
                const confidence = Number(suggestion?.confidence);
                if (Number.isFinite(confidence) && confidence < threshold) {
                    continue;
                }
                let applied = false;
                for (const [canonical, raw] of Object.entries(suggested)) {
                    if (mergedMapping[canonical] || usedRawFields.has(raw)) {
                        continue;
                    }
                    mergedMapping[canonical] = raw;
                    usedRawFields.add(raw);
                    applied = true;
                }
                if (applied && suggestion?.suggestion_id) {
                    acceptedSuggestionIds.push(String(suggestion.suggestion_id));
                }
            }
        }

        if (Object.keys(mergedMapping).length === 0) {
            toast.error('No high-confidence field mapping suggestions found.');
            return;
        }

        const confidenceValues = acceptedSuggestionIds
            .map((id) => confidenceById.get(id))
            .filter((value): value is number => Number.isFinite(value));
        const confidenceAvg = confidenceValues.length > 0
            ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
            : undefined;

        applySuggestedFieldMapping(mergedMapping, {
            mode: 'batch',
            suggestionCount: acceptedSuggestionIds.length || Number(adapterPreviewResult.auto_apply?.candidate_count || 0),
            confidenceAvg,
            acceptedSuggestionIds,
        });
    };

    // ── Split ──────────────────────────────────────────────────
    const runSplit = async () => {
        setSplitLoading(true);
        try {
            const payload = buildSplitPayload();
            const adapterConfigError = payload.__adapter_config_error;
            const fieldMappingError = payload.__field_mapping_error;
            if (typeof adapterConfigError === 'string' && adapterConfigError.trim()) {
                toast.error(`Split adapter config JSON error: ${adapterConfigError}`);
                return;
            }
            if (typeof fieldMappingError === 'string' && fieldMappingError.trim()) {
                toast.error(`Split field mapping JSON error: ${fieldMappingError}`);
                return;
            }
            delete payload.__adapter_config_error;
            delete payload.__field_mapping_error;

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

    const saveSplitAdapterPreference = async () => {
        try {
            const parsedConfig = parseJsonObjectInput(splitAdapterConfigText);
            if (parsedConfig.error) {
                toast.error(`Split adapter config JSON error: ${parsedConfig.error}`);
                return;
            }
            const parsedFieldMapping = parseJsonStringMapInput(splitFieldMappingText);
            if (parsedFieldMapping.error) {
                toast.error(`Split field mapping JSON error: ${parsedFieldMapping.error}`);
                return;
            }
            const res = await api.put<AdapterPreferenceResponse>(
                `/projects/${projectId}/dataset/adapter-preference`,
                {
                    adapter_id: splitAdapterId || 'default-canonical',
                    adapter_config: Object.keys(parsedConfig.value).length > 0 ? parsedConfig.value : {},
                    field_mapping: Object.keys(parsedFieldMapping.value).length > 0 ? parsedFieldMapping.value : {},
                    task_profile: splitTaskProfile !== 'auto' ? splitTaskProfile : undefined,
                },
            );
            setAdapterPreference(res.data);
            toast.success(`Saved adapter preset (${res.data.source}).`);
        } catch (err: any) {
            toast.error(err?.response?.data?.detail || 'Failed to save adapter preset.');
        }
    };

    const autoDetectAndSaveAdapterPreference = async () => {
        try {
            const parsedConfig = parseJsonObjectInput(adapterConfigText);
            if (parsedConfig.error) {
                setAdapterPreviewError(`Adapter config JSON error: ${parsedConfig.error}`);
                return;
            }
            const parsedFieldMapping = parseJsonStringMapInput(adapterFieldMappingText);
            if (parsedFieldMapping.error) {
                setAdapterPreviewError(`Field mapping JSON error: ${parsedFieldMapping.error}`);
                return;
            }
            const res = await api.post<AdapterPreferenceAutoDetectResponse>(
                `/projects/${projectId}/dataset/adapter-preference/auto-detect`,
                {
                    dataset_type: adapterType,
                    sample_size: adapterSampleSize,
                    task_profile: adapterTaskProfile !== 'auto' ? adapterTaskProfile : undefined,
                    adapter_config: Object.keys(parsedConfig.value).length > 0 ? parsedConfig.value : undefined,
                    field_mapping: Object.keys(parsedFieldMapping.value).length > 0 ? parsedFieldMapping.value : undefined,
                    save: true,
                },
            );
            const nextPref = res.data?.preference;
            const nextPreview = res.data?.preview;
            if (nextPref) {
                setAdapterPreference(nextPref);
                setSplitAdapterId(nextPref.adapter_id || 'default-canonical');
                setSplitTaskProfile(nextPref.task_profile ? String(nextPref.task_profile) : 'auto');
                setSplitAdapterConfigText(
                    nextPref.adapter_config && Object.keys(nextPref.adapter_config).length > 0
                        ? JSON.stringify(nextPref.adapter_config, null, 2)
                        : '',
                );
                setSplitFieldMappingText(
                    nextPref.field_mapping && Object.keys(nextPref.field_mapping).length > 0
                        ? JSON.stringify(nextPref.field_mapping, null, 2)
                        : '',
                );
                setAdapterId(nextPref.adapter_id || 'auto');
                setAdapterTaskProfile(nextPref.task_profile ? String(nextPref.task_profile) : 'auto');
            }
            if (nextPreview) {
                setAdapterPreviewResult(nextPreview);
                setAdapterPreviewError('');
                if (nextPreview.resolved_task_profile) {
                    setAdapterTaskProfile(String(nextPreview.resolved_task_profile));
                }
            }
            toast.success(`Auto-detected and saved adapter preset (${nextPref?.adapter_id || 'default-canonical'}).`);
        } catch (err: any) {
            setAdapterPreviewError(err?.response?.data?.detail || 'Auto-detect failed');
        }
    };

    const syncSplitAdapterFromPreview = () => {
        setSplitAdapterId(adapterId || 'default-canonical');
        setSplitTaskProfile(adapterTaskProfile || 'auto');
        setSplitAdapterConfigText(adapterConfigText);
        setSplitFieldMappingText(adapterFieldMappingText);
        toast.success('Copied adapter preview settings into split configuration.');
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
    const taskProfileOptions = [
        ...new Set([
            ...(adapterCatalog?.supported_task_profiles || []),
            ...DEFAULT_TASK_PROFILE_OPTIONS,
        ]),
    ];
    const autoFixSuggestions = Array.isArray(adapterPreviewResult?.auto_fix_suggestions)
        ? adapterPreviewResult?.auto_fix_suggestions || []
        : [];
    const suggestionsWithMapping = autoFixSuggestions.filter((item) => (
        item?.suggested_field_mapping && Object.keys(item.suggested_field_mapping).length > 0
    ));

    return (
        <div className="dataprep-panel">
            <div className="dp-view-tabs">
                <button
                    type="button"
                    className={`dp-view-tab ${activeView === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveView('overview')}
                >
                    Overview
                </button>
                <button
                    type="button"
                    className={`dp-view-tab ${activeView === 'adapters' ? 'active' : ''}`}
                    onClick={() => setActiveView('adapters')}
                >
                    Adapter Lab
                </button>
                <button
                    type="button"
                    className={`dp-view-tab ${activeView === 'split' ? 'active' : ''}`}
                    onClick={() => setActiveView('split')}
                >
                    Split
                </button>
            </div>

            {activeView === 'overview' && (
                <>
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

            <div className="dp-section">
                <h3><span className="icon">🧠</span> Semantic Intelligence</h3>
                {semanticPrefillStage && (
                    <div style={{ marginBottom: '.5rem', fontSize: '.78rem', color: 'rgba(255,255,255,.62)' }}>
                        Prefilled from workflow template stage: <strong>{semanticPrefillStage}</strong>
                    </div>
                )}
                <div className="dp-info">
                    <span className="info-icon">ℹ️</span>
                    Analyze semantic diversity, redundancy, and cluster spread in prepared data.
                </div>
                <div className="dp-actions" style={{ flexWrap: 'wrap' }}>
                    <select value={semanticTargetSplit} onChange={e => setSemanticTargetSplit(e.target.value)}>
                        <option value="train">Train</option>
                        <option value="validation">Validation</option>
                        <option value="test">Test</option>
                        <option value="combined">Combined</option>
                    </select>
                    <input
                        type="number"
                        min={20}
                        max={2000}
                        value={semanticSampleSize}
                        onChange={e => setSemanticSampleSize(Math.max(20, Math.min(2000, Number(e.target.value) || 20)))}
                        title="Sample Size"
                        style={{ width: 130 }}
                    />
                    <input
                        type="number"
                        min={2}
                        max={64}
                        value={semanticClusterCount}
                        onChange={e => setSemanticClusterCount(e.target.value)}
                        title="Cluster Count (optional)"
                        placeholder="clusters (opt)"
                        style={{ width: 150 }}
                    />
                    <input
                        type="number"
                        min={0.5}
                        max={0.999}
                        step={0.01}
                        value={semanticThreshold}
                        onChange={e => setSemanticThreshold(e.target.value)}
                        title="Similarity Threshold"
                        style={{ width: 150 }}
                    />
                    <button className="btn-primary" onClick={runSemanticIntelligence} disabled={semanticLoading}>
                        {semanticLoading ? '⏳ Analyzing...' : '🧪 Run Analysis'}
                    </button>
                </div>
                {semanticError && (
                    <p style={{ color: '#ff6b6b', fontSize: '.82rem', margin: '0 0 .75rem' }}>{semanticError}</p>
                )}
                {semanticResult && (
                    <>
                        <div className="dp-stats-grid">
                            <div className="dp-stat">
                                <div className="label">Sampled Rows</div>
                                <div className="value">{semanticResult.sample_size_analyzed || 0}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Clusters</div>
                                <div className="value">{semanticResult.cluster_count || 0}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Diversity Score</div>
                                <div className="value">{asPercent(semanticResult.semantic_diversity_score)}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Redundancy Ratio</div>
                                <div className="value">{asPercent(semanticResult.redundancy_ratio)}</div>
                            </div>
                        </div>
                        {Array.isArray(semanticResult.suggestions) && semanticResult.suggestions.length > 0 && (
                            <div className="dp-resolved-panel" style={{ marginBottom: '.75rem' }}>
                                <div className="dp-resolved-title">Suggestions</div>
                                <ul style={{ margin: 0, paddingLeft: '1rem', color: 'rgba(255,255,255,.78)' }}>
                                    {semanticResult.suggestions.map((item, index) => (
                                        <li key={`semantic-suggestion-${index}`}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {Array.isArray(semanticResult.clusters) && semanticResult.clusters.length > 0 && (
                            <div className="dp-resolved-panel">
                                <div className="dp-resolved-title">Top Clusters</div>
                                <div style={{ display: 'grid', gap: '.5rem' }}>
                                    {semanticResult.clusters.slice(0, 6).map((cluster) => (
                                        <div key={`cluster-${cluster.cluster_id}`} style={{ border: '1px solid rgba(255,255,255,.08)', borderRadius: 8, padding: '.55rem .65rem' }}>
                                            <div style={{ fontSize: '.8rem', fontWeight: 600 }}>
                                                Cluster #{cluster.cluster_id} • {cluster.size} rows ({asPercent(cluster.share)})
                                            </div>
                                            {Array.isArray(cluster.representatives) && cluster.representatives.length > 0 && (
                                                <div style={{ marginTop: '.4rem', display: 'grid', gap: '.4rem' }}>
                                                    {cluster.representatives.slice(0, 2).map((rep, repIndex) => (
                                                        <div key={`cluster-${cluster.cluster_id}-rep-${repIndex}`} style={{ fontSize: '.75rem', color: 'rgba(255,255,255,.72)' }}>
                                                            #{rep.sample_index ?? 0} • sim {asPercent(rep.similarity_to_nearest)} • {String(rep.text_preview || '')}
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                                {semanticResult.report_path && (
                                    <div style={{ marginTop: '.5rem', fontSize: '.74rem', color: 'rgba(255,255,255,.62)' }}>
                                        Report path: {semanticResult.report_path}
                                    </div>
                                )}
                            </div>
                        )}
                    </>
                )}
            </div>
                </>
            )}

            {/* ── Adapter Preview Section ──────────────────────── */}
            {activeView === 'adapters' && (
            <div className="dp-section">
                <h3><span className="icon">🧩</span> Adapter Preview</h3>
                <div className="dp-info">
                    <span className="info-icon">ℹ️</span>
                    Test adapter mapping against sampled rows before split/training. This validates row contracts for different task types and domains.
                </div>
                <div className="dp-actions" style={{ flexWrap: 'wrap' }}>
                    <select value={adapterType} onChange={e => setAdapterType(e.target.value)}>
                        <option value="raw">Raw</option>
                        <option value="cleaned">Cleaned</option>
                        <option value="synthetic">Synthetic</option>
                        <option value="gold_dev">Gold Dev</option>
                        <option value="train">Train</option>
                        <option value="validation">Validation</option>
                        <option value="test">Test</option>
                    </select>
                    <select value={adapterId} onChange={e => setAdapterId(e.target.value)}>
                        <option value="auto">auto</option>
                        {Object.keys(adapterCatalog?.adapters || {})
                            .filter((key) => key !== 'auto')
                            .map((key) => (
                            <option key={key} value={key}>{key}</option>
                        ))}
                    </select>
                    <select value={adapterTaskProfile} onChange={e => setAdapterTaskProfile(e.target.value)}>
                        {taskProfileOptions.map((profile) => (
                            <option key={profile} value={profile}>{profile}</option>
                        ))}
                    </select>
                    <input
                        type="number"
                        min={10}
                        max={5000}
                        value={adapterSampleSize}
                        onChange={e => setAdapterSampleSize(Math.max(10, Math.min(5000, Number(e.target.value) || 10)))}
                        style={{ width: 120 }}
                        title="Sample Size"
                    />
                    <input
                        type="number"
                        min={5}
                        max={100}
                        value={adapterPreviewLimit}
                        onChange={e => setAdapterPreviewLimit(Math.max(5, Math.min(100, Number(e.target.value) || 5)))}
                        style={{ width: 120 }}
                        title="Preview Rows"
                    />
                    <button className="btn-primary" onClick={runAdapterPreview} disabled={adapterPreviewLoading}>
                        {adapterPreviewLoading ? '⏳ Running...' : '🧪 Run Adapter Preview'}
                    </button>
                    <button className="btn-secondary" onClick={autoDetectAndSaveAdapterPreference} disabled={adapterPreviewLoading}>
                        🤖 Auto-Detect + Save Preset
                    </button>
                    <button className="btn-secondary" onClick={loadAdapterCatalog}>
                        🔄 Refresh Catalog
                    </button>
                </div>
                <div style={{ fontSize: '.8rem', color: 'rgba(255,255,255,.6)', marginBottom: '.75rem' }}>
                    Active preset source: <strong>{adapterPreference?.source || 'default'}</strong>
                    {adapterPreference?.adapter_id ? ` • adapter: ${adapterPreference.adapter_id}` : ''}
                    {adapterPreference?.task_profile ? ` • task_profile: ${adapterPreference.task_profile}` : ''}
                    {adapterPreferenceLoading ? ' • syncing...' : ''}
                </div>
                <label className="dp-json-field">
                    Adapter Config JSON (optional)
                    <textarea
                        className="dp-json-input"
                        value={adapterConfigText}
                        onChange={(e) => setAdapterConfigText(e.target.value)}
                        placeholder='{"source_fields":["prompt"],"target_fields":["completion"]}'
                    />
                </label>
                <label className="dp-json-field">
                    Field Mapping JSON (optional)
                    <textarea
                        className="dp-json-input"
                        value={adapterFieldMappingText}
                        onChange={(e) => setAdapterFieldMappingText(e.target.value)}
                        placeholder='{"question":"instruction","answer":"output","text":"content"}'
                    />
                </label>
                {adapterPreviewError && (
                    <p style={{ color: '#ff6b6b', fontSize: '.82rem', margin: '0 0 .75rem' }}>{adapterPreviewError}</p>
                )}
                {adapterPreviewResult && (
                    <>
                        <div className="dp-stats-grid">
                            <div className="dp-stat">
                                <div className="label">Resolved Adapter</div>
                                <div className="value">{adapterPreviewResult.resolved_adapter_id}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Sampled</div>
                                <div className="value">{adapterPreviewResult.sampled_records}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Mapped</div>
                                <div className="value">{adapterPreviewResult.mapped_records}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Dropped</div>
                                <div className="value" style={{ color: adapterPreviewResult.dropped_records > 0 ? '#ff6b6b' : undefined }}>
                                    {adapterPreviewResult.dropped_records}
                                </div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Errors</div>
                                <div className="value">{adapterPreviewResult.error_count}</div>
                            </div>
                            <div className="dp-stat">
                                <div className="label">Task Profile</div>
                                <div className="value">{adapterPreviewResult.resolved_task_profile || 'n/a'}</div>
                            </div>
                        </div>
                        {Array.isArray(adapterPreviewResult.compatibility_warnings) && adapterPreviewResult.compatibility_warnings.length > 0 && (
                            <div style={{ color: '#fbbf24', fontSize: '.8rem', marginBottom: '.75rem' }}>
                                {adapterPreviewResult.compatibility_warnings.join(' | ')}
                            </div>
                        )}
                        {adapterPreviewResult.conformance_report && (
                            <div className="dp-resolved-panel">
                                <div className="dp-resolved-title">Contract Conformance</div>
                                <div className="dp-resolved-kv">
                                    <span>Contract Pass</span>
                                    <strong style={{ color: adapterPreviewResult.conformance_report.contract_pass ? '#34d399' : '#f87171' }}>
                                        {adapterPreviewResult.conformance_report.contract_pass ? 'yes' : 'no'}
                                    </strong>
                                </div>
                                <div className="dp-resolved-kv">
                                    <span>Mapping Success</span>
                                    <strong>{asPercent(adapterPreviewResult.conformance_report.mapping_success_rate)}</strong>
                                </div>
                                <div className="dp-resolved-kv">
                                    <span>Inferred Profiles</span>
                                    <strong>
                                        {(adapterPreviewResult.inferred_task_profiles || []).length > 0
                                            ? adapterPreviewResult.inferred_task_profiles?.join(', ')
                                            : 'n/a'}
                                    </strong>
                                </div>
                                {Object.entries(adapterPreviewResult.conformance_report.required_field_coverage || {}).length > 0 && (
                                    <div className="dp-resolved-grid">
                                        {Object.entries(adapterPreviewResult.conformance_report.required_field_coverage || {}).map(([field, stats]) => (
                                            <div className="dp-stat" key={`required-${field}`}>
                                                <div className="label">{field}</div>
                                                <div className="value" style={{ fontSize: '1.05rem' }}>{asPercent(stats?.ratio)}</div>
                                                <div style={{ fontSize: '.72rem', color: 'rgba(255,255,255,.58)' }}>
                                                    {stats?.present || 0} present / {stats?.missing || 0} missing
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {Array.isArray(adapterPreviewResult.conformance_report.required_fields_below_100) &&
                                    adapterPreviewResult.conformance_report.required_fields_below_100.length > 0 && (
                                    <div style={{ fontSize: '.78rem', color: '#fca5a5' }}>
                                        Missing required coverage: {adapterPreviewResult.conformance_report.required_fields_below_100.join(', ')}
                                    </div>
                                )}
                            </div>
                        )}
                        {autoFixSuggestions.length > 0 && (
                            <div className="dp-resolved-panel">
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    gap: '.6rem',
                                    marginBottom: '.5rem',
                                }}>
                                    <div className="dp-resolved-title" style={{ marginBottom: 0 }}>Auto-Fix Suggestions</div>
                                    <button
                                        type="button"
                                        className="btn-secondary"
                                        onClick={applyAllSuggestedFieldMappings}
                                        disabled={suggestionsWithMapping.length === 0}
                                    >
                                        Apply All Suggested Mappings
                                    </button>
                                </div>
                                {adapterPreviewResult.auto_apply?.applied_count ? (
                                    <div style={{ fontSize: '.74rem', color: 'rgba(255,255,255,.62)', marginBottom: '.5rem' }}>
                                        Auto-apply ready: {adapterPreviewResult.auto_apply.applied_count} mapping(s) at or above
                                        {' '}{asPercent(adapterPreviewResult.auto_apply.confidence_threshold || 0)} confidence.
                                    </div>
                                ) : null}
                                <div style={{ display: 'grid', gap: '.55rem' }}>
                                    {autoFixSuggestions.map((suggestion, index) => (
                                        <div key={`suggestion-${index}`} style={{
                                            border: '1px solid rgba(255,255,255,.08)',
                                            borderRadius: 8,
                                            padding: '.5rem .65rem',
                                            background: 'rgba(13,18,31,.45)',
                                        }}>
                                            <div style={{ fontSize: '.78rem', color: 'rgba(255,255,255,.9)' }}>
                                                {suggestion.message || 'No message'}
                                            </div>
                                            <div style={{ fontSize: '.72rem', color: 'rgba(255,255,255,.62)', marginTop: '.3rem' }}>
                                                Confidence: {asPercent(suggestion.confidence)}
                                                {suggestion.confidence_label ? ` (${suggestion.confidence_label})` : ''}
                                                {typeof suggestion.expected_coverage_gain === 'number'
                                                    ? ` • expected gain ${asPercent(suggestion.expected_coverage_gain)}`
                                                    : ''}
                                            </div>
                                            {suggestion.suggested_field_mapping && Object.keys(suggestion.suggested_field_mapping).length > 0 && (
                                                <>
                                                    <pre className="dp-resolved-json" style={{ marginTop: '.45rem' }}>
                                                        {JSON.stringify({ field_mapping: suggestion.suggested_field_mapping }, null, 2)}
                                                    </pre>
                                                    <button
                                                        type="button"
                                                        className="btn-secondary"
                                                        style={{ marginTop: '.45rem' }}
                                                        onClick={() => applySuggestedFieldMapping(
                                                            suggestion.suggested_field_mapping || {},
                                                            {
                                                                mode: 'single',
                                                                suggestionCount: 1,
                                                                confidenceAvg: typeof suggestion.confidence === 'number'
                                                                    ? suggestion.confidence
                                                                    : undefined,
                                                                acceptedSuggestionIds: suggestion.suggestion_id
                                                                    ? [suggestion.suggestion_id]
                                                                    : undefined,
                                                            },
                                                        )}
                                                    >
                                                        Apply Mapping
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        <div className="dp-resolved-panel">
                            <div className="dp-resolved-title">Adapter Diagnostics</div>
                            <div className="dp-resolved-grid">
                                <div>
                                    <div className="dp-resolved-subtitle">Validation Report</div>
                                    <pre className="dp-resolved-json">
                                        {JSON.stringify(adapterPreviewResult.validation_report || {}, null, 2)}
                                    </pre>
                                </div>
                                <div>
                                    <div className="dp-resolved-subtitle">Detection Scores</div>
                                    <pre className="dp-resolved-json">
                                        {JSON.stringify(adapterPreviewResult.detection_scores || {}, null, 2)}
                                    </pre>
                                </div>
                                <div>
                                    <div className="dp-resolved-subtitle">
                                        Adapter Contract ({adapterPreviewResult.adapter_contract?.version || adapterCatalog?.contract_version || 'n/a'})
                                    </div>
                                    <pre className="dp-resolved-json">
                                        {JSON.stringify(adapterPreviewResult.adapter_contract || {}, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        </div>
                        {adapterPreviewResult.preview_rows?.length > 0 && (
                            <div className="dp-table-wrap">
                                <table className="dp-table">
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>Raw</th>
                                            <th>Mapped</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {adapterPreviewResult.preview_rows.map((row) => (
                                            <tr key={row.index}>
                                                <td>{row.index + 1}</td>
                                                <td>
                                                    <pre style={{ maxWidth: 360, whiteSpace: 'pre-wrap', fontSize: '.75rem' }}>
                                                        {JSON.stringify(row.raw, null, 2)}
                                                    </pre>
                                                </td>
                                                <td>
                                                    <pre style={{ maxWidth: 360, whiteSpace: 'pre-wrap', fontSize: '.75rem' }}>
                                                        {JSON.stringify(row.mapped, null, 2)}
                                                    </pre>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </>
                )}
            </div>
            )}

            {/* ── Split Section ─────────────────────────────────── */}
            {activeView === 'split' && (
            <div className="dp-section">
                <h3><span className="icon">✂️</span> Train / Val / Test Split</h3>
                <div className="dp-info">
                    <span className="info-icon">💡</span>
                    Splits your combined dataset into training, validation, and test JSONL files. Ratios must sum to 1.0.
                </div>
                <div className="dp-resolved-panel" style={{ marginBottom: '1rem' }}>
                    <div className="dp-resolved-title">Split Adapter</div>
                    <div className="dp-actions" style={{ marginBottom: '.5rem' }}>
                        <select value={splitAdapterId} onChange={e => setSplitAdapterId(e.target.value)}>
                            <option value="default-canonical">default-canonical</option>
                            <option value="auto">auto</option>
                            {Object.keys(adapterCatalog?.adapters || {})
                                .filter((key) => key !== 'auto' && key !== 'default-canonical')
                                .map((key) => (
                                <option key={key} value={key}>{key}</option>
                            ))}
                        </select>
                        <select value={splitTaskProfile} onChange={e => setSplitTaskProfile(e.target.value)}>
                            {taskProfileOptions.map((profile) => (
                                <option key={profile} value={profile}>{profile}</option>
                            ))}
                        </select>
                        <button type="button" className="btn-secondary" onClick={syncSplitAdapterFromPreview}>
                            Use Adapter Lab Settings
                        </button>
                        <button type="button" className="btn-secondary" onClick={saveSplitAdapterPreference}>
                            Save as Project Preset
                        </button>
                        <button
                            type="button"
                            className="btn-secondary"
                            onClick={() => void loadAdapterPreference()}
                            disabled={adapterPreferenceLoading}
                        >
                            {adapterPreferenceLoading ? 'Loading...' : 'Load Saved Preset'}
                        </button>
                    </div>
                    <div style={{ fontSize: '.8rem', color: 'rgba(255,255,255,.6)', marginBottom: '.5rem' }}>
                        Resolved preset source: <strong>{adapterPreference?.source || 'default'}</strong>
                        {adapterPreference?.adapter_id ? ` • ${adapterPreference.adapter_id}` : ''}
                        {adapterPreference?.task_profile ? ` • ${adapterPreference.task_profile}` : ''}
                    </div>
                    <label className="dp-json-field">
                        Adapter Config JSON (optional)
                        <textarea
                            className="dp-json-input"
                            value={splitAdapterConfigText}
                            onChange={(e) => setSplitAdapterConfigText(e.target.value)}
                            placeholder='{"field_mapping":{"instruction":"question","response":"answer"}}'
                        />
                    </label>
                    <label className="dp-json-field">
                        Field Mapping JSON (optional)
                        <textarea
                            className="dp-json-input"
                            value={splitFieldMappingText}
                            onChange={(e) => setSplitFieldMappingText(e.target.value)}
                            placeholder='{"question":"instruction","answer":"response"}'
                        />
                    </label>
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
            )}

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
