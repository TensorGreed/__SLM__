/** GoldEntryRowBody row-context chip: a row's tool list is "used" only when the task reads tool lists. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import GoldEntryRowBody from './GoldEntryRowBody';
import { projectReadsRowTools } from '../../types/rowContext';

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

    it('derives "reads tools" from the task profile', () => {
        expect(projectReadsRowTools({ dataset_adapter_preset: { task_profile: 'tool_calling' } })).toBe(true);
        expect(projectReadsRowTools({ selected_recipe: { task_profile: 'tool_calling' } })).toBe(true);
        expect(projectReadsRowTools({ dataset_adapter_preset: { task_profile: 'qa' },
                                      selected_recipe: { task_profile: 'tool_calling' } })).toBe(false);
        expect(projectReadsRowTools(null)).toBe(false);
    });
});
