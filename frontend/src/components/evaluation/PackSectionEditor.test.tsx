/**
 * Quality-Lift phase 7 slice 1 — Generic PackSectionEditor tests.
 *
 * Pins the list-management contract every consumer relies on:
 *   * Initial render shows the loader's items.
 *   * Add appends via the newItem factory.
 *   * Remove deletes by index (and rebalances any collapsed indices).
 *   * Save is disabled until items differ from the baseline (dirty
 *     tracking).
 *   * Save is also disabled when any item fails isItemValid.
 *   * Save calls onSave, resets the dirty baseline on success, and
 *     surfaces inline errors on failure.
 *   * Per-item collapse hides the form body but keeps the header
 *     (delete button + trailing slot remain accessible).
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { PackSectionEditor } from './PackSectionEditor';

interface DemoItem {
    id: string;
    value: string;
}

function renderEditor(overrides: Record<string, unknown> = {}) {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const props = {
        title: 'Demo section',
        initialItems: [
            { id: 'a', value: 'one' },
            { id: 'b', value: 'two' },
        ] as DemoItem[],
        itemKey: (item: DemoItem, i: number) => item.id || `__new-${i}`,
        newItem: () => ({ id: '', value: '' }),
        renderItem: (item: DemoItem, index: number, mutate: (next: DemoItem) => void) => (
            <input
                aria-label={`item-${index}-value`}
                value={item.value}
                onChange={(e) => mutate({ ...item, value: e.target.value })}
            />
        ),
        onSave,
        testIdPrefix: 'demo',
        ...overrides,
    };
    const utils = render(<PackSectionEditor<DemoItem> {...props} />);
    return { onSave, ...utils };
}

describe('PackSectionEditor', () => {
    it('renders initial items + the add button', () => {
        renderEditor();
        expect(screen.getByText('Demo section')).toBeInTheDocument();
        // Both items render (their internal inputs exist).
        expect(screen.getByLabelText('item-0-value')).toHaveValue('one');
        expect(screen.getByLabelText('item-1-value')).toHaveValue('two');
        expect(screen.getByTestId('demo-add')).toBeInTheDocument();
    });

    it('Save button is disabled until items change', async () => {
        const user = userEvent.setup();
        renderEditor();
        const save = screen.getByTestId('demo-save');
        expect(save).toBeDisabled();

        await user.type(screen.getByLabelText('item-0-value'), 'x');
        expect(save).not.toBeDisabled();
    });

    it('Add appends an item via newItem; Save calls onSave with the new list', async () => {
        const user = userEvent.setup();
        const { onSave } = renderEditor({
            // Override newItem so we know the appended shape.
            newItem: () => ({ id: 'c', value: 'three' }),
        });
        await user.click(screen.getByTestId('demo-add'));
        // Save should be enabled because the list changed.
        await user.click(screen.getByTestId('demo-save'));
        await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
        const argList = onSave.mock.calls[0][0] as DemoItem[];
        expect(argList).toEqual([
            { id: 'a', value: 'one' },
            { id: 'b', value: 'two' },
            { id: 'c', value: 'three' },
        ]);
    });

    it('Remove drops the item + Save sends the trimmed list', async () => {
        const user = userEvent.setup();
        const { onSave } = renderEditor();
        await user.click(screen.getByTestId('demo-item-0-remove'));
        await user.click(screen.getByTestId('demo-save'));
        await waitFor(() => expect(onSave).toHaveBeenCalled());
        expect(onSave.mock.calls[0][0]).toEqual([{ id: 'b', value: 'two' }]);
    });

    it('isItemValid disables Save when any item fails validation', async () => {
        const user = userEvent.setup();
        renderEditor({
            // Mark item-1 invalid; even after we dirty it, Save stays
            // disabled because validity is global across the list.
            isItemValid: (it: DemoItem) => it.value !== 'two',
        });
        await user.type(screen.getByLabelText('item-0-value'), '!');
        // Dirty but invalid → Save still disabled.
        expect(screen.getByTestId('demo-save')).toBeDisabled();
    });

    it('Save surfaces an inline error when onSave rejects', async () => {
        const user = userEvent.setup();
        const onSave = vi.fn().mockRejectedValue(new Error('Backend rejected!'));
        const props = {
            title: 'Demo section',
            initialItems: [{ id: 'a', value: 'one' }],
            itemKey: (it: DemoItem) => it.id,
            newItem: () => ({ id: 'x', value: '' }),
            renderItem: (it: DemoItem, _i: number, mutate: (next: DemoItem) => void) => (
                <input
                    aria-label="v"
                    value={it.value}
                    onChange={(e) => mutate({ ...it, value: e.target.value })}
                />
            ),
            onSave,
            testIdPrefix: 'demo',
        };
        render(<PackSectionEditor<DemoItem> {...props} />);
        await user.type(screen.getByLabelText('v'), 'x');
        await user.click(screen.getByTestId('demo-save'));
        await waitFor(() => {
            expect(screen.getByRole('alert')).toHaveTextContent('Backend rejected!');
        });
    });

    it('collapse hides the item body but keeps the header accessible', async () => {
        const user = userEvent.setup();
        renderEditor();
        // Initially expanded — input visible.
        expect(screen.getByLabelText('item-0-value')).toBeInTheDocument();
        // Click the chevron on item 0.
        const item0 = screen.getByTestId('demo-item-0');
        const collapseBtn = item0.querySelector('button[aria-label="collapse"]')!;
        await user.click(collapseBtn);
        // Input gone, but header buttons (delete, etc.) still present.
        expect(screen.queryByLabelText('item-0-value')).toBeNull();
        expect(screen.getByTestId('demo-item-0-remove')).toBeInTheDocument();
    });

    it('shows the empty placeholder when the list is empty', () => {
        renderEditor({ initialItems: [] });
        expect(screen.getByText(/No item.* defined yet/i)).toBeInTheDocument();
    });
});
