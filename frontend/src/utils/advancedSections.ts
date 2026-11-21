/**
 * Registry of the ids pages give their ``AdvancedSection`` (UI simplification
 * pass). Every id registered here must be reachable from the ⌘K palette: a
 * ``workspaceCommands`` entry whose path carries ``?open=<id>``. The contract
 * test (``advancedSections.test.ts``) enforces it; a page that adopts
 * ``AdvancedSection`` adds its id here and the matching command.
 */

import type { WorkspaceCommand } from './workspaceCommands';

export const ADVANCED_SECTION_IDS: readonly string[] = [
    // Results › Verdict (phase 1): per-result score panels; rule report, lift in full, scorecard, runtime, safety.
    'verdict-scores',
    'verdict-details',
    // Results › Verdict (phase 2): the "Score this run" form's Advanced fields; the named test sets.
    'score-run-advanced',
    'verdict-test-sets',
    // Results › Failures (phase 1): probe pack, rule set, planner, active learning, reroute, drift, radar, …
    'failures-tools',
    // Data (phase 3): the own-files import card's options (Sources and Split); the split form's long tail.
    'own-files-options',
    'split-advanced',
];

/** Ids with no palette command whose path opens them. */
export function unreachableAdvancedIds(ids: readonly string[], commands: readonly WorkspaceCommand[]): string[] {
    const opened = new Set(
        commands
            .map((command) => new URL(command.path, 'http://local').searchParams.get('open'))
            .filter((value): value is string => Boolean(value)),
    );
    return ids.filter((id) => !opened.has(id));
}
