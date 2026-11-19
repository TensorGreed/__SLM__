import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import RowContextChip from './RowContextChip';

describe('RowContextChip', () => {
    it('renders nothing without context', () => {
        const { container } = render(<RowContextChip context={null} />);
        expect(container).toBeEmptyDOMElement();
        const r2 = render(<RowContextChip context={{}} />);
        expect(r2.container).toBeEmptyDOMElement();
    });

    it('shows system chip and expands the text (eval shape)', async () => {
        render(<RowContextChip context={{ system: 'Be terse.', system_source: 'row_context.system', digest: 'abc' }} />);
        expect(screen.getByText('Own system message')).toBeInTheDocument();
        expect(screen.queryByTestId('row-context-chip-body')).toBeNull();
        await userEvent.click(screen.getByTestId('row-context-chip-toggle'));
        expect(screen.getByTestId('row-context-chip-system')).toHaveTextContent('Be terse.');
    });

    it('shows own tool list with signatures and not-allowed flag', async () => {
        render(<RowContextChip context={{
            tools: [{ name: 'a', signature: 'a(x: int)', allowed: true }, { name: 'b', signature: 'b()', allowed: false }],
            tools_count: 7, tools_source: 'row', digest: 'd',
        }} />);
        expect(screen.getByText('Own tool list (7)')).toBeInTheDocument();
        await userEvent.click(screen.getByTestId('row-context-chip-toggle'));
        expect(screen.getByText('a(x: int)')).toBeInTheDocument();
        expect(screen.getAllByText(/not allowed by project/)).toHaveLength(1);
    });

    it('computes counts from a raw row context', async () => {
        render(<RowContextChip context={{ tools: [{ name: 'search', description: 'Find things' }, { name: 'open' }] }} />);
        expect(screen.getByText('Own tool list (2)')).toBeInTheDocument();
        await userEvent.click(screen.getByTestId('row-context-chip-toggle'));
        expect(screen.getByText('search')).toBeInTheDocument();
        expect(screen.getByText(/Find things/)).toBeInTheDocument();
    });

    it('labels project catalog for a row without its own list', () => {
        render(<RowContextChip context={{ system: 's', digest: 'd' }} toolListSource="project" />);
        expect(screen.getByText('Project catalog')).toBeInTheDocument();
        expect(screen.queryByText(/Own tool list/)).toBeNull();
    });

    it('lists unused kinds', () => {
        render(<RowContextChip context={{ labels: ['a'], output_schema: { type: 'object' } }} />);
        expect(screen.getByText('not used by this task: output_schema, labels')).toBeInTheDocument();
        render(<RowContextChip context={{ system: 's', unused_kinds: ['labels'], digest: 'd' }} testId="b" />);
        expect(screen.getByText('not used by this task: labels')).toBeInTheDocument();
    });

    it('marks raw tools unused when the project is not tool-calling', () => {
        render(<RowContextChip context={{ tools: [{ name: 'x' }] }} toolsUsed={false} />);
        expect(screen.getByText('not used by this task: tools')).toBeInTheDocument();
    });
});
