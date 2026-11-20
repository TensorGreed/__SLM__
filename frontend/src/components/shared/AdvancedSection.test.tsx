import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import AdvancedSection from './AdvancedSection';
import { workspaceCommands } from '../../utils/workspaceCommands';

const KEY = 'brewslm.advanced.train';

afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
});

describe('AdvancedSection', () => {
    it('is collapsed by default and mounts children only while open', async () => {
        const user = userEvent.setup();
        render(<AdvancedSection pageKey="train" id="adv" count={3}><div data-testid="inner" /></AdvancedSection>);
        const toggle = screen.getByTestId('advanced-toggle');
        expect(toggle).toHaveTextContent('Advanced (3)');
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByTestId('inner')).toBeNull();
        await user.click(toggle);
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('is keyboard operable', async () => {
        const user = userEvent.setup();
        render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        await user.tab();
        expect(screen.getByTestId('advanced-toggle')).toHaveFocus();
        await user.keyboard('{Enter}');
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        await user.keyboard(' ');
        expect(screen.queryByTestId('inner')).toBeNull();
    });

    it('remembers open and closed per page', async () => {
        const user = userEvent.setup();
        const first = render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        await user.click(screen.getByTestId('advanced-toggle'));
        expect(window.localStorage.getItem(KEY)).toBe('open');
        first.unmount();

        const second = render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        await user.click(screen.getByTestId('advanced-toggle'));
        expect(window.localStorage.getItem(KEY)).toBe('closed');
        second.unmount();

        // A different page is unaffected.
        render(<AdvancedSection pageKey="ship" id="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.queryByTestId('inner')).toBeNull();
    });

    it('tolerates storage that throws', async () => {
        const user = userEvent.setup();
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.queryByTestId('inner')).toBeNull();
        await user.click(screen.getByTestId('advanced-toggle'));
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('opens when ?open= targets it, without overwriting the remembered state', () => {
        render(<AdvancedSection pageKey="train" id="adv" openSection="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        expect(window.localStorage.getItem(KEY)).toBeNull();
    });

    it('re-opens on a new request after the user closed it', async () => {
        const user = userEvent.setup();
        const { rerender } = render(<AdvancedSection pageKey="train" id="adv" openSection="adv"><div data-testid="inner" /></AdvancedSection>);
        await user.click(screen.getByTestId('advanced-toggle'));
        expect(screen.queryByTestId('inner')).toBeNull();
        rerender(<AdvancedSection pageKey="train" id="adv" openSection="adv" openNonce={1}><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('is reachable from a palette command: its ?open= value opens the section', () => {
        // The palette navigates to the command's path; the page passes the
        // ?open= value down. Later phases hide content by giving it an id
        // that some command carries.
        const command = workspaceCommands(7).find((c) => c.label === 'Compress first')!;
        const openValue = new URL(command.path, 'http://x').searchParams.get('open');
        expect(openValue).toBe('compress');
        render(<AdvancedSection pageKey="ship" id="compress" openSection={openValue}><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });
});
