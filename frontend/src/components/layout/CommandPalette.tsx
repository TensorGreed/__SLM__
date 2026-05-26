/**
 * CommandPalette — global Cmd-K / Ctrl-K palette for fast navigation.
 * Lightweight modal: open with the keyboard shortcut (or any caller
 * passing ``open``), type to filter, ↑/↓ to move, Enter to navigate,
 * Esc to dismiss.
 *
 * No external dep — just a substring match. The action set is scoped
 * to the current project workspace.
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
    Activity,
    BookOpen,
    Bot,
    Boxes,
    ClipboardList,
    FileCode,
    FolderTree,
    Home,
    Layers,
    PenSquare,
    Puzzle,
    Rocket,
    Search,
    Settings2,
    Sparkles,
    Workflow,
} from 'lucide-react';

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
    beginnerMode?: boolean;
}

const KEY_OPEN = 'k';

function buildItems(
    projectId: number | null,
    beginnerMode: boolean,
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
    const base = `/project/${projectId}`;

    const pipelineItems: Array<[string, string, ReactNode]> = [
        ['Pipeline Runs', `${base}/pipeline/data`, <FolderTree size={14} />],
        ['Data Studio', `${base}/data-studio`, <ClipboardList size={14} />],
        ['Annotation', `${base}/annotate`, <PenSquare size={14} />],
    ];

    for (const [label, path, icon] of pipelineItems) {
        items.push({
            id: path,
            label,
            hint: path,
            section: 'Data Pipeline',
            icon,
            onSelect: () => navigate(path),
        });
    }

    const trainingItems: Array<[string, string, ReactNode]> = [
        ['Training Configurations', `${base}/training-config`, <Settings2 size={14} />],
        ['Base Model Registry', `${base}/models`, <Boxes size={14} />],
        ['Autopilot Planner', `${base}/autopilot`, <ClipboardList size={14} />],
        ['Playground', `${base}/playground`, <Bot size={14} />],
        ['Deployments', `${base}/deployments`, <Rocket size={14} />],
        ['Observability', `${base}/observability`, <Activity size={14} />],
        ['Guided Setup', `${base}/wizard`, <Sparkles size={14} />],
    ];

    for (const [label, path, icon] of trainingItems) {
        items.push({
            id: path,
            label,
            hint: path,
            section: 'Training',
            icon,
            onSelect: () => navigate(path),
        });
    }

    if (!beginnerMode) {
        items.push({
            id: `${base}/adapter-studio`,
            label: 'Adapter Studio',
            hint: `${base}/adapter-studio`,
            section: 'Training',
            icon: <Boxes size={14} />,
            onSelect: () => navigate(`${base}/adapter-studio`),
        });
        items.push({
            id: `${base}/extensions`,
            label: 'Extension Studio',
            hint: `${base}/extensions`,
            section: 'Training',
            icon: <Puzzle size={14} />,
            onSelect: () => navigate(`${base}/extensions`),
        });
        items.push(
            {
                id: `${base}/workflow`,
                label: 'Workflow Builder',
                hint: `${base}/workflow`,
                section: 'Automation',
                icon: <Workflow size={14} />,
                onSelect: () => navigate(`${base}/workflow`),
            },
            {
                id: `${base}/recipes`,
                label: 'Pipeline recipes',
                hint: `${base}/recipes`,
                section: 'Automation',
                icon: <BookOpen size={14} />,
                onSelect: () => navigate(`${base}/recipes`),
            },
            {
                id: `${base}/manifest`,
                label: 'Pipeline as Code',
                hint: `${base}/manifest`,
                section: 'Automation',
                icon: <FileCode size={14} />,
                onSelect: () => navigate(`${base}/manifest`),
            },
            {
                id: `${base}/domain/packs`,
                label: 'Domain Packs',
                hint: `${base}/domain/packs`,
                section: 'Domain',
                icon: <Boxes size={14} />,
                onSelect: () => navigate(`${base}/domain/packs`),
            },
            {
                id: `${base}/domain/profiles`,
                label: 'Domain Profiles',
                hint: `${base}/domain/profiles`,
                section: 'Domain',
                icon: <Layers size={14} />,
                onSelect: () => navigate(`${base}/domain/profiles`),
            },
        );
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

export default function CommandPalette({ projectId, beginnerMode = false }: Props) {
    const navigate = useNavigate();
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [activeIndex, setActiveIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const listRef = useRef<HTMLDivElement | null>(null);

    const items = useMemo(
        () => buildItems(projectId, beginnerMode, navigate),
        [projectId, beginnerMode, navigate],
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
