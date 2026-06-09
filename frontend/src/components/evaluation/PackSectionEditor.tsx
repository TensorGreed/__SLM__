/**
 * Quality-Lift phase 7 slice 1 — Generic pack section editor.
 *
 * Thin shell that owns the list-management concerns shared by every
 * pack section we plan to ship:
 *   * Phase 7 slice 1: SlicesSection (this slice's concrete consumer)
 *   * Phase 7 slice 2: BehavioralTestsSection (planned)
 *   * Phase 7 slice 3: MultiSeedConfigSection (planned)
 *
 * The component is deliberately unaware of the item schema — every
 * section provides its own ``renderItem`` callback that receives a
 * single item + a mutator so the per-section form can update it.
 *
 * What the editor owns:
 *   * Item list state (the source of truth between load + save).
 *   * Add / remove buttons.
 *   * Dirty tracking (the Save button is only enabled when items
 *     differ from the last loaded snapshot).
 *   * Save lifecycle (calls onSave, surfaces success / error, resets
 *     the dirty baseline on success).
 *   * Optional collapse-per-item affordance so a long list of slices
 *     or tests doesn't fill the screen.
 *
 * What the section consumer owns:
 *   * The form for each item (renderItem).
 *   * Item creation (newItem callback — called when "+ Add" is
 *     pressed; the editor appends the result).
 *   * Item validity check (isItemValid — used to disable Save when
 *     any item is malformed). The backend validator runs on save too,
 *     but a UI-side check catches the common cases before the round-
 *     trip.
 *   * Stable item keying (itemKey — required for React reconciliation
 *     since item ids can be empty during typing).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Plus, Trash2 } from 'lucide-react';
import './PackSectionEditor.css';

export interface PackSectionEditorProps<T> {
    title: string;
    description?: string;
    /** Initial items from the loader. Re-fetched value resets the
     *  editor's internal state + dirty baseline. */
    initialItems: T[];
    /** Stable React key per item. Item ids may be empty mid-typing
     *  so the consumer typically uses an index-derived synthetic key
     *  or stamps an internal ``__local_id``. */
    itemKey: (item: T, index: number) => string;
    /** Build a fresh item when the user clicks "+ Add". */
    newItem: () => T;
    /** Render the per-item form. The mutator replaces the item at
     *  ``index`` with the new value; the consumer assembles the
     *  updated shape and calls it. */
    renderItem: (item: T, index: number, mutate: (next: T) => void) => React.ReactNode;
    /** Optional: returns ``true`` when the item passes UI-level
     *  validation. Used to disable the Save button when any item is
     *  malformed. When omitted, all items count as valid. */
    isItemValid?: (item: T) => boolean;
    /** Save the current item list. Returns / resolves on success;
     *  rejects with a string message on failure (surfaced as the
     *  inline error). */
    onSave: (items: T[]) => Promise<void>;
    /** Label for the Add button. Defaults to "Add item". */
    addLabel?: string;
    /** Per-item label used in the delete confirmation. Defaults to
     *  "item". */
    itemLabel?: string;
    /** Optional: render after the item header (right-aligned). Used
     *  by SlicesSection for the "Gate this slice" button. */
    renderItemHeaderTrailing?: (item: T, index: number) => React.ReactNode;
    /** Hash the item to detect dirtiness. When omitted, JSON.stringify
     *  is used (fine for closed-grammar items; sections with file
     *  blobs / huge payloads can plug in a cheaper hash). */
    hashItem?: (item: T) => string;
    /** When true, items render with a collapse caret. Default true. */
    collapsible?: boolean;
    /** When set, the editor renders a custom data-testid prefix so
     *  consuming tests can scope their queries. */
    testIdPrefix?: string;
}


function defaultHash<T>(item: T): string {
    try {
        return JSON.stringify(item);
    } catch {
        return String(item);
    }
}


export function PackSectionEditor<T>({
    title,
    description,
    initialItems,
    itemKey,
    newItem,
    renderItem,
    isItemValid,
    onSave,
    addLabel = 'Add item',
    itemLabel = 'item',
    renderItemHeaderTrailing,
    hashItem,
    collapsible = true,
    testIdPrefix,
}: PackSectionEditorProps<T>) {
    const [items, setItems] = useState<T[]>(initialItems);
    const [baselineHashes, setBaselineHashes] = useState<string[]>(() =>
        initialItems.map((it) => (hashItem ?? defaultHash)(it)),
    );
    const [collapsed, setCollapsed] = useState<Set<number>>(new Set());
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);
    const [savedAt, setSavedAt] = useState<number | null>(null);

    // Re-sync internal state when the loader hands us a fresh list
    // (e.g. project switch, manual reload).
    useEffect(() => {
        setItems(initialItems);
        setBaselineHashes(initialItems.map((it) => (hashItem ?? defaultHash)(it)));
        setCollapsed(new Set());
        setSaveError(null);
        setSavedAt(null);
    }, [initialItems, hashItem]);

    const currentHashes = useMemo(
        () => items.map((it) => (hashItem ?? defaultHash)(it)),
        [items, hashItem],
    );
    const isDirty = useMemo(
        () =>
            currentHashes.length !== baselineHashes.length ||
            currentHashes.some((h, i) => h !== baselineHashes[i]),
        [currentHashes, baselineHashes],
    );
    const allValid = useMemo(
        () => (isItemValid ? items.every(isItemValid) : true),
        [items, isItemValid],
    );

    const mutateItem = useCallback(
        (index: number, next: T) => {
            setItems((prev) => prev.map((it, i) => (i === index ? next : it)));
        },
        [],
    );

    const handleAdd = useCallback(() => {
        setItems((prev) => [...prev, newItem()]);
    }, [newItem]);

    const handleRemove = useCallback(
        (index: number) => {
            // No confirm() — the user can re-add cheaply, and the Save
            // button stays disabled until they commit. Matches the
            // "rejected rows are selectable + bulk-droppable" rule in
            // spirit: removing here is just a UI mutation, not a
            // destructive action on persisted state.
            setItems((prev) => prev.filter((_, i) => i !== index));
            setCollapsed((prev) => {
                const next = new Set<number>();
                for (const ci of prev) {
                    if (ci < index) {
                        next.add(ci);
                    } else if (ci > index) {
                        next.add(ci - 1);
                    }
                }
                return next;
            });
        },
        [],
    );

    const toggleCollapsed = useCallback((index: number) => {
        setCollapsed((prev) => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    }, []);

    const handleSave = useCallback(async () => {
        setSaving(true);
        setSaveError(null);
        try {
            await onSave(items);
            setBaselineHashes(items.map((it) => (hashItem ?? defaultHash)(it)));
            setSavedAt(Date.now());
        } catch (err: unknown) {
            const message =
                err instanceof Error
                    ? err.message
                    : typeof err === 'string'
                    ? err
                    : 'Save failed.';
            setSaveError(message);
        } finally {
            setSaving(false);
        }
    }, [onSave, items, hashItem]);

    const sectionTestId = testIdPrefix ? `${testIdPrefix}-section` : undefined;

    return (
        <section className="pack-section-editor" data-testid={sectionTestId}>
            <header className="pack-section-editor__header">
                <div>
                    <h3>{title}</h3>
                    {description && (
                        <p className="pack-section-editor__description">{description}</p>
                    )}
                </div>
                <button
                    type="button"
                    className="btn btn-secondary pack-section-editor__add"
                    onClick={handleAdd}
                    data-testid={testIdPrefix ? `${testIdPrefix}-add` : undefined}
                >
                    <Plus size={14} aria-hidden="true" /> {addLabel}
                </button>
            </header>

            {items.length === 0 ? (
                <p className="pack-section-editor__empty">
                    No {itemLabel}s defined yet.
                </p>
            ) : (
                <ul className="pack-section-editor__list">
                    {items.map((item, index) => {
                        const key = itemKey(item, index);
                        const isCollapsed = collapsible && collapsed.has(index);
                        const itemValid = isItemValid ? isItemValid(item) : true;
                        return (
                            <li
                                key={key}
                                className={`pack-section-editor__item ${
                                    itemValid ? '' : 'pack-section-editor__item--invalid'
                                }`}
                                data-testid={testIdPrefix ? `${testIdPrefix}-item-${index}` : undefined}
                            >
                                <div className="pack-section-editor__item-head">
                                    {collapsible && (
                                        <button
                                            type="button"
                                            className="pack-section-editor__collapse"
                                            onClick={() => toggleCollapsed(index)}
                                            aria-label={isCollapsed ? 'expand' : 'collapse'}
                                            aria-expanded={!isCollapsed}
                                        >
                                            {isCollapsed ? (
                                                <ChevronRight size={14} aria-hidden="true" />
                                            ) : (
                                                <ChevronDown size={14} aria-hidden="true" />
                                            )}
                                        </button>
                                    )}
                                    <span className="pack-section-editor__item-index">
                                        #{index + 1}
                                    </span>
                                    {renderItemHeaderTrailing && (
                                        <span className="pack-section-editor__item-trailing">
                                            {renderItemHeaderTrailing(item, index)}
                                        </span>
                                    )}
                                    <button
                                        type="button"
                                        className="btn btn-ghost pack-section-editor__remove"
                                        onClick={() => handleRemove(index)}
                                        aria-label={`Remove ${itemLabel}`}
                                        data-testid={testIdPrefix ? `${testIdPrefix}-item-${index}-remove` : undefined}
                                    >
                                        <Trash2 size={13} aria-hidden="true" />
                                    </button>
                                </div>
                                {!isCollapsed && (
                                    <div className="pack-section-editor__item-body">
                                        {renderItem(item, index, (next) => mutateItem(index, next))}
                                    </div>
                                )}
                            </li>
                        );
                    })}
                </ul>
            )}

            <footer className="pack-section-editor__footer">
                {saveError && (
                    <span className="pack-section-editor__error" role="alert">
                        {saveError}
                    </span>
                )}
                {!saveError && savedAt && !isDirty && (
                    <span className="pack-section-editor__saved-hint">Saved.</span>
                )}
                <button
                    type="button"
                    className="btn btn-primary"
                    disabled={!isDirty || saving || !allValid}
                    onClick={() => void handleSave()}
                    data-testid={testIdPrefix ? `${testIdPrefix}-save` : undefined}
                >
                    {saving ? 'Saving…' : 'Save'}
                </button>
            </footer>
        </section>
    );
}

export default PackSectionEditor;
