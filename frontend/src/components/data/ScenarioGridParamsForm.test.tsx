import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import ScenarioGridParamsForm, { parseMatrix } from './ScenarioGridParamsForm';

const MD = `| region | tier | team |
|---|---|---|
| north | gold | core |
| south | silver | edge |
| north |  | core |
`;

describe('parseMatrix', () => {
    it('reads each markdown column as a de-duplicated dimension', () => {
        const m = parseMatrix(MD);
        expect(m.error).toBeNull();
        expect(m.dimensions).toEqual([
            { name: 'region', values: ['north', 'south'] },
            { name: 'tier', values: ['gold', 'silver'] },
            { name: 'team', values: ['core', 'edge'] },
        ]);
        expect(m.cells).toBe(8);
    });

    it('reads CSV with quotes', () => {
        const m = parseMatrix('region,tier\n"north, far",gold\nsouth,silver\n');
        expect(m.dimensions[0].values).toEqual(['north, far', 'south']);
        expect(m.cells).toBe(4);
    });

    it('reports a dimension with no values and an empty file', () => {
        expect(parseMatrix('a,b\n1,\n').error).toContain('"b" has no values');
        expect(parseMatrix('  \n').error).toContain('no header');
    });

    it('rejects more than 10 dimensions', () => {
        const header = Array.from({ length: 11 }, (_, i) => `d${i}`).join(',');
        const row = Array.from({ length: 11 }, () => 'x').join(',');
        expect(parseMatrix(`${header}\n${row}`).error).toContain('At most 10');
    });
});

describe('ScenarioGridParamsForm', () => {
    it('previews dimensions and cells, and emits the params', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        render(<ScenarioGridParamsForm onChange={onChange} />);
        expect(onChange).toHaveBeenLastCalledWith(null);
        await user.upload(screen.getByTestId('sg-file'), new File([MD], 'matrix.md', { type: 'text/markdown' }));
        expect(await screen.findByTestId('sg-cells')).toHaveTextContent('3 dimensions, 8 cells');
        expect(screen.getByTestId('sg-dim-region')).toHaveTextContent('north, south');
        await waitFor(() => expect(onChange).toHaveBeenLastCalledWith({
            matrix_text: MD, matrix_format: 'auto', max_cells: 20, scenarios_per_cell: 1, seed: 0,
        }));
        await user.type(screen.getByTestId('sg-extra'), 'be brief');
        expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ extra_instructions: 'be brief' }));
    });

    it('notes sampling when the grid exceeds max_cells and clamps knobs', async () => {
        const user = userEvent.setup();
        render(<ScenarioGridParamsForm onChange={vi.fn()} />);
        await user.upload(screen.getByTestId('sg-file'), new File([MD], 'matrix.md'));
        const cells = screen.getByTestId('sg-max-cells');
        fireEvent.change(cells, { target: { value: '5' } });
        expect(await screen.findByTestId('sg-cells')).toHaveTextContent('5 will be sampled');
        const per = screen.getByTestId('sg-per-cell');
        fireEvent.change(per, { target: { value: '9' } });
        expect(per).toHaveValue(5);
    });

    it('shows a parse error and emits null', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        render(<ScenarioGridParamsForm onChange={onChange} />);
        await user.upload(screen.getByTestId('sg-file'), new File(['a,b\n1,\n'], 'm.csv'));
        expect(await screen.findByTestId('sg-parse-error')).toHaveTextContent('"b" has no values');
        expect(onChange).toHaveBeenLastCalledWith(null);
    });
});
