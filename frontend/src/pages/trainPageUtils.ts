/** Pure helpers for the Train page (section ids, labels). */

export const TRAIN_SECTION_IDS = ['model', 'lora', 'seeds', 'checks', 'cloud', 'presets', 'plan', 'models', 'task', 'pipeline-presets', 'adapters', 'tokenization', 'remediation', 'structured-plan', 'tool-scoring'] as const;
export type TrainSectionId = (typeof TRAIN_SECTION_IDS)[number];

export function isTrainSectionId(value: string | null | undefined): value is TrainSectionId {
    return !!value && (TRAIN_SECTION_IDS as readonly string[]).includes(value);
}

/** "Span extraction (span-extraction)" from the project's recipe snapshot. */
export function taskShapeLabelFor(project: { selected_recipe?: Record<string, unknown> | null }): string {
    const snapshot = project.selected_recipe;
    if (!snapshot || typeof snapshot !== 'object') return 'Not set';
    const id = typeof snapshot.recipe_id === 'string' ? snapshot.recipe_id : null;
    const name = typeof snapshot.name === 'string' ? snapshot.name : null;
    if (name && id) return `${name} (${id})`;
    return name ?? id ?? 'Not set';
}
