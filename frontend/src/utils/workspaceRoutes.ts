/**
 * The project workspace's information architecture (Epic U, phase U1).
 *
 * Six destinations — Overview, Data, Train, Results, Ship, Settings — and
 * the mapping from the old pipeline tabs (``/pipeline/<tab>``) onto them.
 * Until the later Epic U phases replace the panels, a destination renders
 * the old tab panels as *steps* (``?step=<tab>``); old URLs redirect here
 * (``legacyRedirectPath``), so the bell, the Coach and the docs keep
 * working while their targets are rewritten in U6.
 */

import type { TabKey } from '../types';

export type Destination = 'overview' | 'data' | 'train' | 'results' | 'ship' | 'settings';

/** Destinations that render the old pipeline tabs as steps. */
export type StageDestination = Exclude<Destination, 'overview' | 'settings'>;

export const DESTINATION_LABEL: Record<Destination, string> = {
    overview: 'Overview',
    data: 'Data',
    train: 'Train',
    results: 'Results',
    ship: 'Ship',
    settings: 'Settings',
};

export const DESTINATION_ORDER: Destination[] = ['overview', 'data', 'train', 'results', 'ship', 'settings'];

/** Which old tabs each stage destination carries as steps, in order. */
export const STEPS_BY_DESTINATION: Record<StageDestination, TabKey[]> = {
    data: ['data', 'cleaning', 'goldset', 'synthetic', 'dataprep'],
    // Train is a form (U2): one step; tokenization is a More-options section.
    train: ['training'],
    results: ['eval'],
    // Ship is a flow (U5): one step; compression is an option inside Package.
    ship: ['export'],
};

/** Ship stages (``/ship?stage=<id>``, Epic U phase U5). */
export const SHIP_STAGES = ['package', 'publish', 'deploy', 'monitor'] as const;
export type ShipStageId = (typeof SHIP_STAGES)[number];

export function isShipStage(value: string | null | undefined): value is ShipStageId {
    return !!value && (SHIP_STAGES as readonly string[]).includes(value);
}

/** ``/project/7/ship?stage=deploy`` (+ extra query). */
export function shipPath(projectId: number, stageId: ShipStageId, extraSearch?: string): string {
    const params = new URLSearchParams(extraSearch ?? '');
    params.set('stage', stageId);
    return `${projectPath(projectId, 'ship')}?${params.toString()}`;
}

/** ``/project/7/train?open=plan`` — a Train More-options section. */
export function trainSectionPath(projectId: number, section: string, extraSearch?: string): string {
    const params = new URLSearchParams(extraSearch ?? '');
    params.set('open', section);
    return `${projectPath(projectId, 'train')}?${params.toString()}`;
}

/** Train's More-options sections (``/train?open=<id>``) and the retired
 *  pages that redirect to them. */
export const TRAIN_OPEN_BY_LEGACY_PATH: Record<string, string> = {
    '/training-config': 'plan',
    '/models': 'models',
    '/recipe-picker': 'task',
    '/recipes': 'pipeline-presets',
    '/adapter-studio': 'adapters',
};

/** Data steps' Details sections and where the retired Data Studio sections
 *  and the annotation list landed (Epic U phase U4). */
export const DATA_OPEN_BY_STUDIO_SECTION: Record<string, { step: TabKey; open: string }> = {
    overview: { step: 'data', open: 'overview' },
    sources: { step: 'data', open: 'sources' },
    assist: { step: 'data', open: 'assist' },
    mapping: { step: 'dataprep', open: 'mapping' },
    domain: { step: 'cleaning', open: 'domain' },
    'quality-safety': { step: 'cleaning', open: 'quality' },
    'gold-set': { step: 'goldset', open: 'workbench' },
    'review-queue': { step: 'goldset', open: 'review-queue' },
    'synthetic-playbooks': { step: 'synthetic', open: 'playbook-center' },
    'synthetic-recommendations': { step: 'synthetic', open: 'recommendations' },
    'synthetic-quality': { step: 'synthetic', open: 'quality' },
    'prepare-dataset': { step: 'dataprep', open: 'prepare' },
    'dataset-versions': { step: 'dataprep', open: 'versions' },
};

/** ``/project/7/data?step=goldset&open=annotation``. */
export function dataStepPath(projectId: number, tab: TabKey, open?: string | null, extraSearch?: string): string {
    const params = new URLSearchParams(extraSearch ?? '');
    if (open) params.set('open', open);
    return stagePath(projectId, tab, params.toString());
}

/** Settings sections (``/settings?open=<id>``, Epic U phase U6): the long tail. */
export const SETTINGS_SECTIONS = [
    { id: 'autopilot', label: 'Autopilot Planner', hint: 'plan diffs, repair preview, rollback' },
    { id: 'workflow', label: 'Workflow Builder', hint: 'compose and run pipeline graphs' },
    { id: 'domain-packs', label: 'Domain Packs', hint: 'overlay a domain pack on the project' },
    { id: 'domain-profiles', label: 'Domain Profiles', hint: 'profiles that tune the pack' },
    { id: 'extensions', label: 'Extension Studio', hint: 'runtime and pack plugins' },
    { id: 'decisions', label: 'Autopilot decisions', hint: 'every step the Autopilot took, by run' },
] as const;
export type SettingsSectionId = (typeof SETTINGS_SECTIONS)[number]['id'];

export function isSettingsSection(value: string | null | undefined): value is SettingsSectionId {
    return !!value && SETTINGS_SECTIONS.some((s) => s.id === value);
}

/** ``/project/7/settings?open=domain-packs``. */
export function settingsPath(projectId: number, section?: SettingsSectionId | null): string {
    const base = projectPath(projectId, 'settings');
    return section ? `${base}?open=${section}` : base;
}

/** The retired long-tail pages and the Settings section each became. */
export const SETTINGS_SECTION_BY_LEGACY_PATH: Record<string, SettingsSectionId> = {
    '/autopilot': 'autopilot',
    '/workflow': 'workflow',
    '/domain': 'domain-packs',
    '/domain/packs': 'domain-packs',
    '/domain/profiles': 'domain-profiles',
    '/extensions': 'extensions',
};

/** Results workflows (``/results?tab=<id>``, Epic U phase U3). */
export const RESULTS_TABS = ['verdict', 'failures', 'compare', 'try'] as const;
export type ResultsTab = (typeof RESULTS_TABS)[number];

export function isResultsTab(value: string | null | undefined): value is ResultsTab {
    return !!value && (RESULTS_TABS as readonly string[]).includes(value);
}

/** Anchors inside the old data tabs that now live in a Details section. */
const DATA_OPEN_BY_HASH: Record<string, string> = {
    '#synth-review-queue': 'review-queue',
    '#label-noise-review': 'label-noise',
    '#seed-documents': '',
    '#split': '',
};

/** Anchors the Coach, the bell and the panels used on the eval tab. */
export const RESULTS_TAB_BY_HASH: Record<string, ResultsTab> = {
    '#eval-gaps-panel': 'failures',
    '#probe-pack-panel': 'failures',
    '#reroute-recommendation-panel': 'failures',
    '#eval-pack-editor': 'failures',
    '#failure-clusters-panel': 'failures',
    '#external-benchmarks': 'compare',
    '#experiment-scorecard': 'verdict',
};

/** ``/project/7/results?tab=failures`` (+ extra query). */
export function resultsPath(projectId: number, tab: ResultsTab, extraSearch?: string): string {
    const params = new URLSearchParams(extraSearch ?? '');
    params.set('tab', tab);
    return `${projectPath(projectId, 'results')}?${params.toString()}`;
}

/** Anchors the Coach and the gap panels used on /training-config. */
const TRAIN_OPEN_BY_HASH: Record<string, string> = {
    '#base-model': 'model',
    '#multi-seed': 'seeds',
    '#training-config-gaps-panel': 'plan',
    '#archetype-comparison-panel': 'plan',
    '#checkpoints-panel': 'plan',
};

/** Short step labels (the old tab labels, no emoji). */
export const STEP_LABEL: Record<TabKey, string> = {
    data: 'Sources',
    cleaning: 'Prepare',
    goldset: 'Labels',
    synthetic: 'Generate',
    dataprep: 'Split',
    tokenization: 'Tokenization',
    training: 'Training',
    eval: 'Evaluation',
    compression: 'Compression',
    export: 'Export',
};

const DESTINATION_BY_TAB: Record<TabKey, StageDestination> = {
    data: 'data',
    cleaning: 'data',
    goldset: 'data',
    synthetic: 'data',
    dataprep: 'data',
    tokenization: 'train',
    training: 'train',
    eval: 'results',
    compression: 'ship',
    export: 'ship',
};

const TAB_KEYS = new Set<string>(Object.keys(DESTINATION_BY_TAB));

export function isTabKey(value: string | null | undefined): value is TabKey {
    return !!value && TAB_KEYS.has(value);
}

export function destinationForTab(tab: TabKey): StageDestination {
    return DESTINATION_BY_TAB[tab];
}

export function projectPath(projectId: number, destination: Destination = 'overview'): string {
    const base = `/project/${projectId}`;
    return destination === 'overview' ? base : `${base}/${destination}`;
}

/** The canonical URL of an old pipeline tab: ``/project/7/data?step=cleaning``.
 *  The first step of a destination needs no ``step`` parameter. */
export function stagePath(projectId: number, tab: TabKey, extraSearch?: string): string {
    const destination = destinationForTab(tab);
    const params = new URLSearchParams(extraSearch ?? '');
    if (tab === 'tokenization') {
        params.set('open', 'tokenization');
    } else if (tab === 'compression') {
        params.set('open', 'compress');
    } else if (STEPS_BY_DESTINATION[destination][0] !== tab) {
        params.set('step', tab);
    }
    const query = params.toString();
    return `${projectPath(projectId, destination)}${query ? `?${query}` : ''}`;
}

/** Resolve ``?step=`` for a destination: a known step of that destination,
 *  else its first step. */
export function resolveStep(destination: StageDestination, step: string | null | undefined): TabKey {
    const steps = STEPS_BY_DESTINATION[destination];
    return isTabKey(step) && steps.includes(step) ? step : steps[0];
}

/** The step after ``tab`` in the pipeline order, across destinations
 *  (``dataprep`` → ``tokenization`` → ``training`` …); null after export. */
export function nextTab(tab: TabKey): TabKey | null {
    const order = (['data', 'train', 'results', 'ship'] as StageDestination[]).flatMap((d) => STEPS_BY_DESTINATION[d]);
    const index = order.indexOf(tab);
    return index >= 0 && index < order.length - 1 ? order[index + 1] : null;
}

/**
 * Where an old workspace URL lives now. ``null`` when the path is not a
 * retired one. Search + hash are carried over (``?import=auto`` opens the
 * importer, ``#seed-documents`` scrolls to the card).
 */
export function legacyRedirectPath(projectId: number, pathname: string, search = '', hash = ''): string | null {
    const base = `/project/${projectId}`;
    const rest = pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
    const clean = rest.replace(/\/+$/, '');
    if (clean === '/guide' || clean === '/wizard') {
        return projectPath(projectId);
    }
    if (clean === '/pipeline' || clean === '/pipeline/') {
        return projectPath(projectId, 'data') + search + hash;
    }
    const pipeline = clean.match(/^\/pipeline\/([^/]+)$/);
    if (pipeline) {
        const tab = pipeline[1];
        if (!isTabKey(tab)) return projectPath(projectId, 'data');
        if (tab === 'eval') {
            const resultsTab = RESULTS_TAB_BY_HASH[hash];
            return (resultsTab ? resultsPath(projectId, resultsTab, search) : stagePath(projectId, tab, search)) + hash;
        }
        const open = DATA_OPEN_BY_HASH[hash];
        if (open) return dataStepPath(projectId, tab, open, search) + hash;
        return stagePath(projectId, tab, search) + hash;
    }
    if (clean === '/data-studio') {
        const section = hash.replace(/^#/, '').trim().toLowerCase().replace(/_/g, '-');
        const landing = DATA_OPEN_BY_STUDIO_SECTION[section];
        return landing ? dataStepPath(projectId, landing.step, landing.open, search) : stagePath(projectId, 'data', search);
    }
    if (clean === '/annotate') {
        return dataStepPath(projectId, 'goldset', 'annotation', search) + hash;
    }
    const settingsSection = SETTINGS_SECTION_BY_LEGACY_PATH[clean];
    if (settingsSection) {
        return settingsPath(projectId, settingsSection) + hash;
    }
    if (clean === '/deployments') {
        return shipPath(projectId, 'deploy', search) + hash;
    }
    if (clean === '/observability') {
        return shipPath(projectId, 'monitor', search) + hash;
    }
    if (clean === '/manifest') {
        const params = new URLSearchParams(search);
        params.set('open', 'manifest');
        return shipPath(projectId, 'package', params.toString()) + hash;
    }
    if (clean === '/eval/compare') {
        return resultsPath(projectId, 'compare', search) + hash;
    }
    if (clean === '/playground') {
        return resultsPath(projectId, 'try', search) + hash;
    }
    const trainSection = TRAIN_OPEN_BY_LEGACY_PATH[clean];
    if (trainSection) {
        const params = new URLSearchParams(search);
        params.delete('return_to');
        const section = params.get('expand_multi_seed') === '1'
            ? 'seeds'
            : TRAIN_OPEN_BY_HASH[hash] ?? trainSection;
        return trainSectionPath(projectId, section, params.toString()) + (hash && !TRAIN_OPEN_BY_HASH[hash] ? hash : '');
    }
    return null;
}
