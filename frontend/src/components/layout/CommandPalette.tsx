/**
 * CommandPalette — global Cmd-K / Ctrl-K palette for fast navigation.
 * Lightweight modal: open with the keyboard shortcut (or any caller
 * passing ``open``), type to filter, ↑/↓ to move, Enter to navigate,
 * Esc to dismiss.
 *
 * No external dep — just a substring match. The action set is the
 * workspace command registry (``utils/workspaceCommands``), generated from
 * the IA — nothing is hand-maintained here (Epic U phase U6).
 */

import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Database,
    FlaskConical,
    Home,
    Rocket,
    Search,
    Settings2,
    Sparkles,
} from 'lucide-react';

import { workspaceCommands } from '../../utils/workspaceCommands';

import { DESTINATION_LABEL, type Destination } from '../../utils/workspaceRoutes';

import { OPEN_COMMAND_PALETTE_EVENT } from './commandPaletteBridge';
import './CommandPalette.css';

interface CommandItem {
    id: string;
    label: string;
    hint?: string;
    section: string;
    icon: ReactNode;
    onSelect: () => void;
}

interface Props {
    projectId: number | null;
}

const KEY_OPEN = 'k';

const DESTINATION_ICON: Record<Destination, ReactNode> = {
    overview: <Home size={14} />,
    data: <Database size={14} />,
    train: <FlaskConical size={14} />,
    results: <Sparkles size={14} />,
    ship: <Rocket size={14} />,
    settings: <Settings2 size={14} />,
};

function buildItems(
    projectId: number | null,
    navigate: ReturnType<typeof useNavigate>,
): CommandItem[] {
    const items: CommandItem[] = [];

    items.push({
        id: 'projects',
        label: 'Back to projects',
        hint: '/',
        section: 'Navigation',
        icon: <Home size={14} />,
        onSelect: () => navigate('/'),
    });

    if (projectId == null) return items;

    // The registry is generated from the IA (``utils/workspaceCommands``).
    for (const command of workspaceCommands(projectId)) {
        items.push({
            id: command.id,
            label: command.label,
            hint: command.hint,
            section: DESTINATION_LABEL[command.section],
            icon: DESTINATION_ICON[command.section],
            onSelect: () => navigate(command.path),
        });
    }

    return items;
}

function filterItems(items: CommandItem[], query: string): CommandItem[] {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((item) => {
        const haystack = `${item.label} ${item.section} ${item.hint ?? ''}`.toLowerCase();
        return haystack.includes(q);
    });
}

function groupBySection(items: CommandItem[]): Array<[string, CommandItem[]]> {
    const order: string[] = [];
    const map = new Map<string, CommandItem[]>();
    for (const item of items) {
        if (!map.has(item.section)) {
            order.push(item.section);
            map.set(item.section, []);
        }
        map.get(item.section)!.push(item);
    }
    return order.map((section) => [section, map.get(section)!]);
}

export default function CommandPalette({ projectId }: Props) {
    const navigate = useNavigate();
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [activeIndex, setActiveIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const listRef = useRef<HTMLDivElement | null>(null);

    const items = useMemo(
        () => buildItems(projectId, navigate),
        [projectId, navigate],
    );
    const filtered = useMemo(() => filterItems(items, query), [items, query]);
    const grouped = useMemo(() => groupBySection(filtered), [filtered]);
    // Derive active index inline so it auto-clamps when the filter shrinks
    // the list — keeps state management out of an effect.
    const effectiveActiveIndex =
        filtered.length === 0 ? 0 : Math.min(activeIndex, filtered.length - 1);

    // Global Cmd-K / Ctrl-K opener.
    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === KEY_OPEN) {
                event.preventDefault();
                setOpen((prev) => !prev);
            } else if (event.key === 'Escape') {
                setOpen(false);
            }
        };
        const onOpenEvent = () => setOpen(true);
        window.addEventListener('keydown', onKey);
        window.addEventListener(OPEN_COMMAND_PALETTE_EVENT, onOpenEvent);
        return () => {
            window.removeEventListener('keydown', onKey);
            window.removeEventListener(OPEN_COMMAND_PALETTE_EVENT, onOpenEvent);
        };
    }, []);

    // Focus the input every time we open. Pure side effect — no state
    // updates here (state reset happens in close()).
    useEffect(() => {
        if (open) {
            window.requestAnimationFrame(() => {
                inputRef.current?.focus();
            });
        }
    }, [open]);

    const close = useCallback(() => {
        setOpen(false);
        setQuery('');
        setActiveIndex(0);
    }, []);

    const handleSelect = useCallback(
        (index: number) => {
            const item = filtered[index];
            if (!item) return;
            close();
            item.onSelect();
        },
        [close, filtered],
    );

    const handleKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLInputElement>) => {
            if (event.key === 'ArrowDown') {
                event.preventDefault();
                setActiveIndex((prev) => {
                    if (filtered.length === 0) return 0;
                    const clamped = Math.min(prev, filtered.length - 1);
                    return (clamped + 1) % filtered.length;
                });
            } else if (event.key === 'ArrowUp') {
                event.preventDefault();
                setActiveIndex((prev) => {
                    if (filtered.length === 0) return 0;
                    const clamped = Math.min(prev, filtered.length - 1);
                    return (clamped - 1 + filtered.length) % filtered.length;
                });
            } else if (event.key === 'Enter') {
                event.preventDefault();
                handleSelect(effectiveActiveIndex);
            } else if (event.key === 'Escape') {
                event.preventDefault();
                close();
            }
        },
        [close, effectiveActiveIndex, filtered.length, handleSelect],
    );

    if (!open) return null;

    let runningIndex = -1;
    return (
        <div className="cmdk-backdrop" role="presentation" onClick={close}>
            <div
                className="cmdk-panel"
                role="dialog"
                aria-label="Command palette"
                onClick={(event) => event.stopPropagation()}
            >
                <div className="cmdk-input-row">
                    <Search size={14} className="cmdk-input-icon" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Search pages and actions…"
                        className="cmdk-input"
                        aria-label="Command query"
                        title="Command query"
                    />
                    <span className="cmdk-input-hint">esc to close</span>
                </div>
                <div className="cmdk-list" ref={listRef} role="listbox" aria-label="Command results">
                    {filtered.length === 0 ? (
                        <div className="cmdk-empty">No matches.</div>
                    ) : (
                        grouped.map(([section, sectionItems]) => (
                            <div key={section} className="cmdk-group" role="group" aria-label={section}>
                                <div className="cmdk-group-label">{section}</div>
                                {sectionItems.map((item) => {
                                    runningIndex += 1;
                                    const index = runningIndex;
                                    const active = index === effectiveActiveIndex;
                                    const ariaProps = active
                                        ? { 'aria-selected': 'true' as const }
                                        : { 'aria-selected': 'false' as const };
                                    return (
                                        <div
                                            key={item.id}
                                            className={`cmdk-item ${active ? 'is-active' : ''}`}
                                            onMouseEnter={() => setActiveIndex(index)}
                                            onClick={() => handleSelect(index)}
                                            role="option"
                                            {...ariaProps}
                                        >
                                            <span className="cmdk-item-icon">{item.icon}</span>
                                            <span className="cmdk-item-label">{item.label}</span>
                                            {item.hint && (
                                                <span className="cmdk-item-hint">{item.hint}</span>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
