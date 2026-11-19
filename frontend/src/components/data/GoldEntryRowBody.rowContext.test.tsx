/** GoldEntryRowBody row-context chip: a row's tool list is "used" only when the task reads tool lists. */
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import GoldEntryRowBody from './GoldEntryRowBody';
import GoldSetPanel from './GoldSetPanel';
import { taskReadsRowContext } from '../../types/rowContext';

// A tool row whose gold answer is "no tool call" has no tool_name, but its own tool list is still read.
const row = {
    question: 'Say hi',
    answer: 'Hi',
    row_context: { tools: [{ name: 'get_item' }, { name: 'list_items' }] },
};

describe('GoldEntryRowBody row-context tool label', () => {
    it('labels a tool list as used when the task reads tool lists, even without tool_name', () => {
        render(<GoldEntryRowBody recipeId="qa-sft" row={row} testidPrefix="r" readsTools />);
        expect(screen.getByText('Own tool list (2)')).toBeInTheDocument();
        expect(screen.queryByText(/not used by this task/)).toBeNull();
    });

    it('labels it not used when the task does not read tool lists, even with tool_name', () => {
        render(<GoldEntryRowBody recipeId="qa-sft" row={{ ...row, tool_name: 'get_item' } as never} testidPrefix="r" />);
        expect(screen.getByText('not used by this task: tools')).toBeInTheDocument();
        expect(screen.queryByText(/Own tool list/)).toBeNull();
    });

    it('derives "reads tools" from the backend row_context_reads list', () => {
        expect(taskReadsRowContext({ row_context_reads: ['system', 'tools'] }, 'tools')).toBe(true);
        expect(taskReadsRowContext({ row_context_reads: ['system'] }, 'tools')).toBe(false);
        expect(taskReadsRowContext(null, 'tools')).toBe(false);
    });
});

// The panel takes the label from the GET /gold/entries response (shape as the backend returns it).
const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./LlmGoldGeneratePanel', () => ({ default: () => <div /> }));
vi.mock('../coach/CoachStrip', () => ({ default: () => null }));

function installEntries(profile: string, reads: string[]) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/gold/entries')) {
            return { data: {
                entries: [{ id: 1, question: 'Say hi', answer: 'Hi', row_context: { tools: [{ name: 'get_item' }] } }],
                total: 1,
                dataset: { id: 2, name: 'Gold Dev Set', dataset_type: 'gold_dev', record_count: 1, is_locked: false },
                effective_task_profile: profile,
                row_context_reads: reads,
            } };
        }
        // A project response has no task profile of its own (it never carried one).
        if (/\/projects\/\d+(\?|$)/.test(url)) return { data: { id: 1, selected_recipe: { recipe_id: 'generic-sft' } } };
        return { data: {} };
    });
}

describe('GoldSetPanel row-context tool label', () => {
    beforeEach(() => { apiMock.get.mockReset(); });

    it('tool-calling project: the row tool list is used', async () => {
        installEntries('tool_calling', ['system', 'tools']);
        render(<MemoryRouter><GoldSetPanel projectId={1} openSection="entries" /></MemoryRouter>);
        expect(await screen.findByText('Own tool list (1)')).toBeInTheDocument();
        expect(screen.queryByText(/not used by this task/)).toBeNull();
    });

    it('classification project: the row tool list is not used', async () => {
        installEntries('classification', ['system']);
        render(<MemoryRouter><GoldSetPanel projectId={1} openSection="entries" /></MemoryRouter>);
        expect(await screen.findByText('not used by this task: tools')).toBeInTheDocument();
    });
});
