/**
 * Bridge for the Cmd-K command palette: a custom DOM event lets any
 * part of the app open the palette without prop-drilling. Kept in a
 * separate file from the component itself so Vite's React fast-refresh
 * stays happy (it requires component files to export only components).
 */

export const OPEN_COMMAND_PALETTE_EVENT = 'brewslm:open-command-palette';

/** Imperatively open the global Cmd-K palette from anywhere. */
export function openCommandPalette(): void {
    window.dispatchEvent(new CustomEvent(OPEN_COMMAND_PALETTE_EVENT));
}
