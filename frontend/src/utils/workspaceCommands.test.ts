import { describe, expect, it } from 'vitest';

import { workspaceCommands } from './workspaceCommands';

describe('workspaceCommands', () => {
    const commands = workspaceCommands(7);

    it('starts with the six destinations in order', () => {
        expect(commands.slice(0, 6).map((c) => c.label)).toEqual(['Overview', 'Data', 'Train', 'Results', 'Ship', 'Settings']);
        expect(commands[0].path).toBe('/project/7');
    });

    it('covers every step, section, workflow and stage with a unique path', () => {
        const paths = commands.map((c) => c.path);
        expect(new Set(paths).size).toBe(paths.length);
        expect(paths).toContain('/project/7/data?step=goldset');
        expect(paths).toContain('/project/7/data?open=annotation&step=goldset');
        expect(paths).toContain('/project/7/train?open=plan');
        expect(paths).toContain('/project/7/results?tab=failures');
        expect(paths).toContain('/project/7/ship?stage=monitor');
        expect(paths).toContain('/project/7/settings?open=domain-packs');
    });

    it('files each command under its destination', () => {
        const byLabel = Object.fromEntries(commands.map((c) => [c.label, c.section]));
        expect(byLabel['Annotation jobs']).toBe('data');
        expect(byLabel['Base model registry']).toBe('train');
        expect(byLabel['Try it (playground)']).toBe('results');
        expect(byLabel['Pipeline as Code']).toBe('ship');
        expect(byLabel['Extension Studio']).toBe('settings');
    });
});
