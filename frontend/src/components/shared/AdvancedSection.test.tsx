import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import AdvancedSection from './AdvancedSection';

const KEY = 'brewslm.advanced.train.adv';
const TOGGLE = 'advanced-adv-toggle';

afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    window.localStorage.clear();
});

describe('AdvancedSection', () => {
    it('is collapsed by default and mounts children only while open', async () => {
        const user = userEvent.setup();
        render(<AdvancedSection pageKey="train" id="adv" count={3}><div data-testid="inner" /></AdvancedSection>);
        const toggle = screen.getByTestId(TOGGLE);
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
        expect(screen.getByTestId(TOGGLE)).toHaveFocus();
        await user.keyboard('{Enter}');
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        await user.keyboard(' ');
        expect(screen.queryByTestId('inner')).toBeNull();
    });

    it('remembers open and closed per page and section', async () => {
        const user = userEvent.setup();
        const first = render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        await user.click(screen.getByTestId(TOGGLE));
        expect(window.localStorage.getItem(KEY)).toBe('open');
        first.unmount();

        const second = render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        await user.click(screen.getByTestId(TOGGLE));
        expect(window.localStorage.getItem(KEY)).toBe('closed');
        second.unmount();

        // Another page, and another section on the same page, are unaffected.
        await user.click((render(<AdvancedSection pageKey="train" id="adv"><div /></AdvancedSection>), screen.getByTestId(TOGGLE)));
        expect(window.localStorage.getItem(KEY)).toBe('open');
        render(<AdvancedSection pageKey="ship" id="adv" testId="ship-adv"><div data-testid="ship-inner" /></AdvancedSection>);
        render(<AdvancedSection pageKey="train" id="other" testId="other-adv"><div data-testid="other-inner" /></AdvancedSection>);
        expect(screen.queryByTestId('ship-inner')).toBeNull();
        expect(screen.queryByTestId('other-inner')).toBeNull();
    });

    it('tolerates storage that throws', async () => {
        const user = userEvent.setup();
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        render(<AdvancedSection pageKey="train" id="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.queryByTestId('inner')).toBeNull();
        await user.click(screen.getByTestId(TOGGLE));
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('opens when ?open= targets it, without overwriting the remembered state', () => {
        render(<AdvancedSection pageKey="train" id="adv" openSection="adv"><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
        expect(window.localStorage.getItem(KEY)).toBeNull();
    });

    it('scrolls into view on an open request', () => {
        vi.useFakeTimers();
        const scroll = vi.fn();
        Element.prototype.scrollIntoView = scroll;
        render(<AdvancedSection pageKey="train" id="adv" openSection="adv"><div /></AdvancedSection>);
        act(() => { vi.advanceTimersByTime(60); });
        expect(scroll).toHaveBeenCalledTimes(1);
    });

    it('re-opens on a new request after the user closed it', async () => {
        const user = userEvent.setup();
        const { rerender } = render(<AdvancedSection pageKey="train" id="adv" openSection="adv"><div data-testid="inner" /></AdvancedSection>);
        await user.click(screen.getByTestId(TOGGLE));
        expect(screen.queryByTestId('inner')).toBeNull();
        rerender(<AdvancedSection pageKey="train" id="adv" openSection="adv" openNonce={1}><div data-testid="inner" /></AdvancedSection>);
        expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('keepMounted keeps hidden fields mounted with their value', async () => {
        const user = userEvent.setup();
        render(
            <AdvancedSection pageKey="train" id="adv" keepMounted>
                <input data-testid="field" defaultValue="0.2" />
            </AdvancedSection>,
        );
        const field = screen.getByTestId('field') as HTMLInputElement;
        expect(screen.getByTestId('advanced-adv-body')).toHaveAttribute('hidden');
        expect(field.value).toBe('0.2');
        await user.click(screen.getByTestId(TOGGLE));
        expect(screen.getByTestId('advanced-adv-body')).not.toHaveAttribute('hidden');
        expect(field.value).toBe('0.2');
    });

    it('without keepMounted, parent-held state still feeds the request while collapsed', () => {
        let sent = '';
        function Form() {
            const [rate, setRate] = useState('0.2');
            return (
                <form onSubmit={(e) => { e.preventDefault(); sent = rate; }}>
                    <AdvancedSection pageKey="train" id="adv">
                        <input data-testid="field" value={rate} onChange={(e) => setRate(e.target.value)} />
                    </AdvancedSection>
                    <button type="submit">Go</button>
                </form>
            );
        }
        render(<Form />);
        expect(screen.queryByTestId('field')).toBeNull();
        act(() => { screen.getByText('Go').click(); });
        expect(sent).toBe('0.2');
    });
});
