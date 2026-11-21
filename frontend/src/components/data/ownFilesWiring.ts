/**
 * What the "I already have train / validation / test files" import sends besides the files
 * (UI simplification pass, phase 3).
 *
 * The import card lives in two places: Data › Sources (step 1) and Data › Split (step 5, inside
 * the split form). On Split the split form owns the chat template and the adapter wiring; on
 * Sources nothing is edited, so the same values are derived from the server here, exactly as the
 * split form derives them when it mounts (adapter preference + the active prepared manifest).
 * `ownFilesWiring.test.tsx` proves the two placements send the same request for the same server state.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import type { AdapterWiring } from '../../api/predefinedSplit';

export const CHAT_TEMPLATES = [
    { value: 'llama3', label: 'Llama 3' },
    { value: 'chatml', label: 'ChatML' },
    { value: 'zephyr', label: 'Zephyr' },
    { value: 'phi3', label: 'Phi-3' },
];

/** The split form's chat template before anything is hydrated. */
export const DEFAULT_CHAT_TEMPLATE = 'llama3';

export interface AdapterPreferenceLike {
    adapter_id?: string | null;
    adapter_config?: Record<string, unknown> | null;
    task_profile?: string | null;
}

export interface PreparedManifestLike {
    split_source?: string | null;
    predefined_test_active?: boolean;
    splits?: Record<string, unknown> | null;
    resolved_split_config?: Record<string, unknown> | null;
    ratios?: Record<string, unknown> | null;
    seed?: unknown;
    chat_template?: unknown;
}

/** Whether a re-split would replace the user's own test file. The backend sets
 *  ``predefined_test_active`` on every manifest while that file is still accepted
 *  (a keep-test re-split drops ``split_source`` but keeps the file); manifests
 *  from before that key fall back to a predefined import with test rows. */
export function ownTestFileActive(m: Pick<PreparedManifestLike, 'split_source' | 'predefined_test_active' | 'splits'> | null | undefined): boolean {
    if (!m) return false;
    if (typeof m.predefined_test_active === 'boolean') return m.predefined_test_active;
    return m.split_source === 'predefined' && Number((m.splits || {}).test ?? 0) > 0;
}

export interface OwnFilesWiring {
    chatTemplate: string;
    /** False while the profile's default template applies (nothing is sent then). */
    sendChatTemplate: boolean;
    adapter: AdapterWiring;
    predefinedActive: boolean;
    activeTestRows: number;
}

/** The split form's values on mount, before any edit (`DatasetPrepPanel` loadAdapterPreference +
 *  loadActivePreparedConfig). Null inputs are failed / absent GETs. */
export function wiringFromServer(
    pref: AdapterPreferenceLike | null | undefined,
    manifest: PreparedManifestLike | null | undefined,
): OwnFilesWiring {
    const m = manifest ?? null;
    const rc = (m?.resolved_split_config || {}) as Record<string, unknown>;
    const ratios = (m?.ratios || {}) as Record<string, unknown>;
    const trainR = typeof rc.train_ratio === 'number' ? rc.train_ratio : ratios.train;
    const valR = typeof rc.val_ratio === 'number' ? rc.val_ratio : ratios.val;
    const testR = typeof rc.test_ratio === 'number' ? rc.test_ratio : ratios.test;
    const seedV = typeof rc.seed === 'number' ? rc.seed : m?.seed;
    const templateV = typeof rc.chat_template === 'string' ? rc.chat_template : m?.chat_template;
    // Explicit ratios / seed on the active version switch the form off "use profile defaults", and
    // from then on the chosen template is sent with the import.
    const hydratedExplicit = [trainR, valR, testR, seedV].some((v) => typeof v === 'number');
    const config = pref?.adapter_config && Object.keys(pref.adapter_config).length > 0 ? pref.adapter_config : {};
    const activeSplits = (m?.splits || {}) as Record<string, unknown>;
    return {
        chatTemplate: typeof templateV === 'string' ? templateV : DEFAULT_CHAT_TEMPLATE,
        sendChatTemplate: hydratedExplicit,
        adapter: {
            adapterId: pref?.adapter_id || 'default-canonical',
            taskProfile: pref?.task_profile ? String(pref.task_profile) : 'auto',
            adapterConfig: config,
        },
        predefinedActive: ownTestFileActive(m),
        activeTestRows: typeof activeSplits.test === 'number' ? activeSplits.test : 0,
    };
}

/** Loads the wiring for the Sources placement; ``reload`` re-reads it after an import. */
export function useOwnFilesWiring(projectId: number): { wiring: OwnFilesWiring; loaded: boolean; reload: () => void } {
    const [wiring, setWiring] = useState<OwnFilesWiring>(() => wiringFromServer(null, null));
    const [loaded, setLoaded] = useState(false);
    const [nonce, setNonce] = useState(0);
    useEffect(() => {
        let cancelled = false;
        const read = async <T,>(url: string): Promise<T | null> => {
            try {
                return (await api.get<T>(url)).data ?? null;
            } catch {
                return null;
            }
        };
        void (async () => {
            const [pref, manifest] = await Promise.all([
                read<AdapterPreferenceLike>(`/projects/${projectId}/dataset/adapter-preference`),
                read<PreparedManifestLike>(`/projects/${projectId}/prepared-manifest`),
            ]);
            if (cancelled) return;
            setWiring(wiringFromServer(pref, manifest));
            setLoaded(true);
        })();
        return () => { cancelled = true; };
    }, [projectId, nonce]);
    const reload = useCallback(() => setNonce((n) => n + 1), []);
    return { wiring, loaded, reload };
}
