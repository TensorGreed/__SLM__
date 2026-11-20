import { describe, expect, it } from 'vitest';

import { ADVANCED_SECTION_IDS, unreachableAdvancedIds } from './advancedSections';
import { workspaceCommands } from './workspaceCommands';

describe('Advanced sections are reachable from the palette', () => {
    const commands = workspaceCommands(7);

    it('every registered section id has a command that opens it', () => {
        expect(unreachableAdvancedIds(ADVANCED_SECTION_IDS, commands)).toEqual([]);
    });

    it('registers the Results sections (phase 1) and their palette paths open the right tab', () => {
        expect(ADVANCED_SECTION_IDS).toEqual(expect.arrayContaining(['verdict-scores', 'verdict-details', 'failures-tools']));
        const paths = commands.map((c) => c.path);
        expect(paths).toContain('/project/7/results?open=verdict-scores&tab=verdict');
        expect(paths).toContain('/project/7/results?open=verdict-details&tab=verdict');
        expect(paths).toContain('/project/7/results?open=failures-tools&tab=failures');
    });

    it('the check catches an id without a command and accepts one with a command', () => {
        expect(unreachableAdvancedIds(['compress', 'no-such-section'], commands)).toEqual(['no-such-section']);
    });
});
