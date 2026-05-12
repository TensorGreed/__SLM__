/**
 * TypeScript shapes for the Wave H extensions plane (priority.md
 * P37 contracts, P38 scaffold generator, P40 Extension Studio frontend).
 *
 * Mirrors:
 *   backend/app/services/plugin_contracts.py
 *   backend/app/services/plugin_contract_service.py
 *   backend/app/services/scaffold_service.py
 *   backend/app/api/extensions.py
 */

export type PluginKind =
    | 'data_adapter'
    | 'training_runtime'
    | 'domain_pack'
    | 'eval_pack';

export const KNOWN_PLUGIN_KINDS: PluginKind[] = [
    'data_adapter',
    'training_runtime',
    'domain_pack',
    'eval_pack',
];

export type ScaffoldAlias =
    | 'adapter'
    | 'runtime'
    | 'domain-pack'
    | 'eval-pack';

export const PLUGIN_KIND_LABEL: Record<PluginKind, string> = {
    data_adapter: 'Data adapter',
    training_runtime: 'Training runtime',
    domain_pack: 'Domain pack',
    eval_pack: 'Evaluation pack',
};

export const PLUGIN_KIND_ALIAS: Record<PluginKind, ScaffoldAlias> = {
    data_adapter: 'adapter',
    training_runtime: 'runtime',
    domain_pack: 'domain-pack',
    eval_pack: 'eval-pack',
};

export const ALIAS_TO_KIND: Record<ScaffoldAlias, PluginKind> = {
    adapter: 'data_adapter',
    runtime: 'training_runtime',
    'domain-pack': 'domain_pack',
    'eval-pack': 'eval_pack',
};

// ----- GET /api/extensions ------------------------------------------------

export interface ExtensionKindStatus {
    kind: PluginKind;
    contract_version: string;
    supports_safe_reload: boolean;
    has_module_loader: boolean;
    settings_key: string | null;
    configured_modules: string[];
    loaded_modules: string[];
    load_errors: Record<string, string>;
    registered_count: number;
    recognized_exports: string[];
    note?: string;
}

export interface ExtensionListResponse {
    kinds: ExtensionKindStatus[];
    known_kinds: PluginKind[];
}

// ----- POST /api/extensions/validate --------------------------------------

export interface PluginContractCheck {
    name: string;
    ok: boolean;
    message: string;
}

export interface PluginContractReport {
    kind: PluginKind;
    module: string;
    contract_version: string;
    declared_version: string | null;
    declared_ids: string[];
    ok: boolean;
    checks: PluginContractCheck[];
    import_error: string | null;
}

// ----- POST /api/extensions/reload ----------------------------------------

export type ReloadStatus = 'ok' | 'partial' | 'error' | 'not_supported';

export interface ReloadKindResult {
    kind: PluginKind;
    status: ReloadStatus;
    requested_modules?: string[];
    loaded_modules?: string[];
    failed_modules?: Record<string, string>;
    registered_count?: number;
    message?: string;
}

export interface ReloadResponse {
    results: ReloadKindResult[];
}

// ----- POST /api/extensions/scaffold --------------------------------------

export interface ScaffoldRequest {
    kind: PluginKind;
    plugin_id: string;
    display_name?: string;
    description?: string;
    author?: string;
    version?: string;
    export_dir?: string;
    write?: boolean;
}

export interface ScaffoldResponse {
    kind: PluginKind;
    plugin_id: string;
    plugin_id_raw: string;
    module_basename: string;
    display_name: string;
    description: string;
    author: string;
    version: string;
    contract_version: string;
    output_dir: string;
    files: Record<string, string>;
    written_files: string[];
}
