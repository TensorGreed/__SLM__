/**
 * The workspace's command registry (Epic U, phase U6): every destination,
 * step, workflow, stage and section a user can jump to, generated from the
 * IA in ``workspaceRoutes``. The ⌘K palette renders this list; nothing is
 * hand-maintained there any more.
 */

import {
    DESTINATION_LABEL,
    DESTINATION_ORDER,
    dataStepPath,
    projectPath,
    RESULTS_TABS,
    resultsPath,
    SETTINGS_SECTIONS,
    settingsPath,
    SHIP_STAGES,
    shipPath,
    STEP_LABEL,
    STEPS_BY_DESTINATION,
    trainSectionPath,
    type Destination,
} from './workspaceRoutes';
import type { TabKey } from '../types';

export interface WorkspaceCommand {
    id: string;
    label: string;
    hint: string;
    /** The destination the command belongs to (its palette section). */
    section: Destination;
    path: string;
}

const RESULTS_LABEL: Record<(typeof RESULTS_TABS)[number], string> = {
    verdict: 'Verdict',
    failures: 'Failures',
    compare: 'Compare runs',
    try: 'Try it (playground)',
};

/** Results Advanced sections and their palette labels (ids in ``advancedSections.ts``). */
const RESULTS_ADVANCED: Array<[string, (typeof RESULTS_TABS)[number], string]> = [
    ['verdict-scores', 'verdict', 'Scores (every result of the run)'],
    ['verdict-details', 'verdict', 'Verdict details (rule report, scorecard, runtime, safety)'],
    ['score-run-advanced', 'verdict', 'Score this run: advanced options (metric, tokens, temperature, model path, judges)'],
    ['verdict-test-sets', 'verdict', 'Test sets (named eval sets, versions, prompt formats)'],
    ['failures-tools', 'failures', 'Failure tools (probe pack, rules and slices, planner, drift, radar…)'],
    ['how-scored-settings', 'verdict', "How it's scored: scoring settings (tool-call scoring, plan rubric, remediation checks, null as absent, SQL execution, automatic base-vs-fine-tuned check)"],
];

const SHIP_LABEL: Record<(typeof SHIP_STAGES)[number], string> = {
    package: 'Package',
    publish: 'Publish',
    deploy: 'Deploy',
    monitor: 'Monitor (observability)',
};

export const TRAIN_SECTION_LABEL: Record<string, string> = {
    model: 'Model and hyperparameters',
    lora: 'LoRA and advanced training',
    seeds: 'Seeds',
    checks: 'Checks',
    cloud: 'Cloud burst',
    presets: 'Training presets',
    plan: 'Plan and gaps',
    models: 'Base model registry',
    task: 'Task type',
    'pipeline-presets': 'Pipeline presets',
    adapters: 'Adapter Studio',
    tokenization: 'Tokenization',
    'train-advanced': 'Train: more options, advanced (checks, cloud burst, presets, plan, registry, task, adapters, tokenization)',
};

const DATA_SECTION_LABEL: Record<string, string> = {
    annotation: 'Annotation jobs',
    versions: 'Dataset versions',
    'review-queue': 'Review queue',
    health: 'Data health and leakage',
};

/** Data Advanced sections (UI simplification pass, phase 3; ids in ``advancedSections.ts``). */
const DATA_ADVANCED: Array<[string, TabKey, string]> = [
    ['own-files-options', 'data', 'Own train / val / test files: options (one picker per file, overlapping rows, chat template, column mapping)'],
    ['split-advanced', 'dataprep', 'Split: advanced (adapter, field mapping, presets, chat template, stratify / disjoint, effective config)'],
];

export function workspaceCommands(projectId: number): WorkspaceCommand[] {
    const commands: WorkspaceCommand[] = [];
    for (const destination of DESTINATION_ORDER) {
        const path = projectPath(projectId, destination);
        commands.push({ id: path, label: DESTINATION_LABEL[destination], hint: path, section: destination, path });
    }
    // The first step is the destination itself (same URL), so it is skipped.
    for (const tab of STEPS_BY_DESTINATION.data.slice(1)) {
        const path = dataStepPath(projectId, tab);
        commands.push({ id: path, label: `Data · ${STEP_LABEL[tab]}`, hint: path, section: 'data', path });
    }
    for (const [id, label] of Object.entries(DATA_SECTION_LABEL)) {
        const tab = id === 'annotation' || id === 'review-queue' ? 'goldset' : 'dataprep';
        const path = dataStepPath(projectId, tab, id);
        commands.push({ id: path, label, hint: path, section: 'data', path });
    }
    for (const [id, tab, label] of DATA_ADVANCED) {
        const path = dataStepPath(projectId, tab, id);
        commands.push({ id: path, label, hint: `Data · ${STEP_LABEL[tab]}`, section: 'data', path });
    }
    for (const [id, label] of Object.entries(TRAIN_SECTION_LABEL)) {
        const path = trainSectionPath(projectId, id);
        commands.push({ id: path, label, hint: path, section: 'train', path });
    }
    for (const tab of RESULTS_TABS) {
        const path = resultsPath(projectId, tab);
        commands.push({ id: path, label: RESULTS_LABEL[tab], hint: path, section: 'results', path });
    }
    // Results Advanced sections (UI simplification pass, phase 1): each opens via ``?open=<id>``.
    for (const [id, tab, label] of RESULTS_ADVANCED) {
        const path = resultsPath(projectId, tab, `open=${id}`);
        commands.push({ id: path, label, hint: `Results · ${RESULTS_LABEL[tab]}`, section: 'results', path });
    }
    for (const stage of SHIP_STAGES) {
        const path = shipPath(projectId, stage);
        commands.push({ id: path, label: SHIP_LABEL[stage], hint: path, section: 'ship', path });
    }
    commands.push({ id: `${projectPath(projectId, 'ship')}?open=compress`, label: 'Compress first', hint: 'Ship · Package', section: 'ship', path: `${projectPath(projectId, 'ship')}?open=compress&stage=package` });
    commands.push({ id: `${projectPath(projectId, 'ship')}?open=manifest`, label: 'Pipeline as Code', hint: 'Ship · Package', section: 'ship', path: `${projectPath(projectId, 'ship')}?open=manifest&stage=package` });
    for (const section of SETTINGS_SECTIONS) {
        const path = settingsPath(projectId, section.id);
        commands.push({ id: path, label: section.label, hint: path, section: 'settings', path });
    }
    return commands;
}
