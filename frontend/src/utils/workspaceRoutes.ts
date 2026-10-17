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
    train: ['tokenization', 'training'],
    results: ['eval'],
    ship: ['compression', 'export'],
};

/** Short step labels (the old tab labels, no emoji). */
export const STEP_LABEL: Record<TabKey, string> = {
    data: 'Sources',
    cleaning: 'Prepare',
    goldset: 'Answer key',
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
    if (STEPS_BY_DESTINATION[destination][0] !== tab) {
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
        return stagePath(projectId, tab, search) + hash;
    }
    return null;
}
