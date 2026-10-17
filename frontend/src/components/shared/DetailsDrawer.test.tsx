import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import DetailsDrawer from './DetailsDrawer';

const sections = [
    { id: 'a', title: 'Section A', render: vi.fn(() => <div data-testid="body-a" />) },
    { id: 'b', title: 'Section B', hint: 'b hint', render: vi.fn(() => <div data-testid="body-b" />) },
];

describe('DetailsDrawer', () => {
    it('starts closed and renders a section body only while open', async () => {
        const user = userEvent.setup();
        render(<DetailsDrawer sections={sections} testId="dd" />);
        expect(screen.queryByTestId('dd-section-a')).toBeNull();
        await user.click(screen.getByTestId('dd-toggle'));
        expect(screen.getByTestId('dd-section-b')).toHaveTextContent('b hint');
        expect(screen.queryByTestId('body-a')).toBeNull();
        await user.click(screen.getByTestId('dd-section-a'));
        expect(screen.getByTestId('body-a')).toBeInTheDocument();
        expect(screen.queryByTestId('body-b')).toBeNull();
        await user.click(screen.getByTestId('dd-section-a'));
        expect(screen.queryByTestId('body-a')).toBeNull();
    });

    it('opens the requested section on mount and on a new request', () => {
        const { rerender } = render(<DetailsDrawer sections={sections} testId="dd" openSection="b" />);
        expect(screen.getByTestId('body-b')).toBeInTheDocument();
        rerender(<DetailsDrawer sections={sections} testId="dd" openSection="a" openNonce={1} />);
        expect(screen.getByTestId('body-a')).toBeInTheDocument();
        expect(screen.getByTestId('body-b')).toBeInTheDocument();
    });

    it('renders nothing without sections', () => {
        const { container } = render(<DetailsDrawer sections={[]} />);
        expect(container.firstChild).toBeNull();
    });
});
