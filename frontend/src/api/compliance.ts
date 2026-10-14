import api from './client';

/** One built-in compliance bundle: a regime tag + entity types with their
 *  starting labelling definitions. */
export interface ComplianceBundle {
    id: string;
    regime: string;
    regime_label: string;
    name: string;
    description: string;
    types: Array<{ type: string; definition: string | null }>;
    applied: boolean;
}

export interface ComplianceRegime {
    regime: string;
    label: string;
    custom: boolean;
}

/** The project's compliance state (runtime_config.compliance). */
export interface ComplianceState {
    bundles: string[];
    custom: string[];
    type_regimes: Record<string, string[]>;
    entity_types: string[];
    definitions: Record<string, string>;
    regimes: ComplianceRegime[];
}

export interface ComplianceBundlesPayload extends ComplianceState {
    builtin: ComplianceBundle[];
}

export interface ComplianceApplyResult extends ComplianceState {
    definitions_added: string[];
    definitions_kept: string[];
}

export async function fetchComplianceBundles(projectId: number): Promise<ComplianceBundlesPayload> {
    const res = await api.get<ComplianceBundlesPayload>(`/projects/${projectId}/compliance/bundles`);
    return res.data;
}

export async function applyComplianceBundles(
    projectId: number,
    bundleIds: string[],
    removeBundleIds: string[] = [],
): Promise<ComplianceApplyResult> {
    const res = await api.post<ComplianceApplyResult>(`/projects/${projectId}/compliance/apply`, {
        bundle_ids: bundleIds,
        remove_bundle_ids: removeBundleIds,
    });
    return res.data;
}

export async function saveTypeRegimes(projectId: number, typeRegimes: Record<string, string[]>): Promise<ComplianceState> {
    const res = await api.put<ComplianceState>(`/projects/${projectId}/compliance/type-regimes`, { type_regimes: typeRegimes });
    return res.data;
}
