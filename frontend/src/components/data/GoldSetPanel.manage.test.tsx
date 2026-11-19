/**
 * GoldSetPanel answer-key management: row provenance labels, select + delete rows, the file
 * manager mounted with the set label, locked sets. Runs for two task shapes (the code path is the
 * same for every recipe).
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./LlmGoldGeneratePanel', () => ({ default: () => <div /> }));
vi.mock('../coach/CoachStrip', () => ({ default: () => null }));

import GoldSetPanel from './GoldSetPanel';

const SHAPES: Record<string, { recipe: string; rows: Record<string, unknown>[] }> = {
    'qa-sft': { recipe: 'qa-sft', rows: [{ id: 1, question: 'Q1', answer: 'A1' }, { id: 3, question: 'Q3', answer: 'A3', source_row_id: 'ext-7' }] },
    classification: { recipe: 'classification', rows: [{ id: 1, text: 'great', label: 'pos' }, { id: 3, text: 'bad', label: 'neg' }] },
};

function install(recipe: string, rows: Record<string, unknown>[], locked = false) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/gold/entries')) {
            return { data: { entries: rows, total: rows.length, dataset: { id: 2, name: 'Gold Dev Set', dataset_type: 'gold_dev', record_count: rows.length, is_locked: locked } } };
        }
        if (/\/projects\/\d+(\?|$)/.test(url)) return { data: { selected_recipe: { recipe_id: recipe } } };
        return { data: {} };
    });
}

function mount() {
    return render(
        <MemoryRouter initialEntries={['/route']}>
            <GoldSetPanel projectId={1} openSection="entries" />
        </MemoryRouter>,
    );
}

describe.each(Object.entries(SHAPES))('GoldSetPanel management (%s)', (_name, shape) => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.post.mockReset(); apiMock.delete.mockReset();
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('labels each row with its id, type and file id, and the set with its dataset', async () => {
        install(shape.recipe, shape.rows);
        mount();
        expect(await screen.findByTestId('gold-set-label')).toHaveTextContent('Gold Dev Set');
        expect(screen.getByTestId('gold-set-label')).toHaveTextContent('2 rows');
        expect(await screen.findByTestId('gold-entry-row-1')).toHaveTextContent('row 3');
        if (shape.recipe === 'qa-sft') expect(screen.getByTestId('gold-entry-row-1')).toHaveTextContent('file id ext-7');
        expect(screen.getByTestId('gold-entry-row-0')).toHaveTextContent('gold_dev');
    });

    it('deletes only the selected rows by id', async () => {
        install(shape.recipe, shape.rows);
        apiMock.post.mockResolvedValue({ data: { deleted: 1, remaining: 1, missing: [], dataset: {} } });
        const user = userEvent.setup();
        mount();
        await user.click(await screen.findByTestId('gold-entry-select-1'));
        expect(screen.getByTestId('gold-delete-selected')).toHaveTextContent('Delete selected (1)');
        await user.click(screen.getByTestId('gold-delete-selected'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/1/gold/gold_dev/rows/delete', { row_ids: [3] }));
        // the list is re-fetched and the selection cleared
        await waitFor(() => expect(screen.getByTestId('gold-delete-selected')).toHaveTextContent('Delete selected (0)'));
    });

    it('select all shown ticks every row', async () => {
        install(shape.recipe, shape.rows);
        const user = userEvent.setup();
        mount();
        await user.click(await screen.findByTestId('gold-select-all'));
        expect(screen.getByTestId('gold-delete-selected')).toHaveTextContent('Delete selected (2)');
    });

    it('locked set: no selection bar, row boxes disabled', async () => {
        install(shape.recipe, shape.rows, true);
        mount();
        expect(await screen.findByTestId('gold-files-locked')).toBeInTheDocument();
        expect(screen.queryByTestId('gold-entries-select-bar')).toBeNull();
        expect(screen.getByTestId('gold-entry-select-0')).toBeDisabled();
    });
});
