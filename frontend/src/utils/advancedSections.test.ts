import { describe, expect, it } from 'vitest';

import { ADVANCED_SECTION_IDS, unreachableAdvancedIds } from './advancedSections';
import { workspaceCommands } from './workspaceCommands';

describe('Advanced sections are reachable from the palette', () => {
    const commands = workspaceCommands(7);

    it('every registered section id has a command that opens it', () => {
        // Empty until pages adopt AdvancedSection; real from then on.
        expect(unreachableAdvancedIds(ADVANCED_SECTION_IDS, commands)).toEqual([]);
    });

    it('the check catches an id without a command and accepts one with a command', () => {
        expect(unreachableAdvancedIds(['compress', 'no-such-section'], commands)).toEqual(['no-such-section']);
    });
});
