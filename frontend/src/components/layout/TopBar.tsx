import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Bell, CircleHelp } from 'lucide-react';

import api from '../../api/client';
import './TopBar.css';

interface TopBarProps {
    title: string;
    subtitle?: string;
    actions?: ReactNode;
    withSidebar?: boolean;
}

interface MeResponse {
    principal?: {
        user_id?: number;
        username?: string;
        role?: string;
    } | null;
}

interface RuntimeSettingField {
    key: string;
    label: string;
    description: string;
    category: string;
    type: 'string' | 'bool' | 'int';
    requires_restart: boolean;
    options: string[];
    multiline: boolean;
    source: 'env' | 'override' | string;
    value: string | boolean | number;
}

interface RuntimeSettingsResponse {
    path: string;
    fields: RuntimeSettingField[];
}

export default function TopBar({ title, subtitle, actions, withSidebar = false }: TopBarProps) {
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [menuOpen, setMenuOpen] = useState(false);
    const [profileOpen, setProfileOpen] = useState(false);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [currentUser, setCurrentUser] = useState<{ userId: number | null; username: string; role: string } | null>(null);
    const [settingsPayload, setSettingsPayload] = useState<RuntimeSettingsResponse | null>(null);
    const [draft, setDraft] = useState<Record<string, string | boolean | number>>({});
    const [settingsError, setSettingsError] = useState('');
    const [settingsInfo, setSettingsInfo] = useState('');
    const [loadingSettings, setLoadingSettings] = useState(false);
    const [savingSettings, setSavingSettings] = useState(false);

    useEffect(() => {
        let mounted = true;
        api.get<MeResponse>('/auth/me')
            .then((res) => {
                if (!mounted) return;
                const principal = res.data?.principal;
                setCurrentUser({
                    userId: typeof principal?.user_id === 'number' ? principal.user_id : null,
                    username: String(principal?.username || 'user'),
                    role: String(principal?.role || 'viewer'),
                });
            })
            .catch(() => {
                if (!mounted) return;
                setCurrentUser(null);
            });
        return () => {
            mounted = false;
        };
    }, []);

    useEffect(() => {
        const onDocClick = (event: MouseEvent) => {
            if (!menuRef.current) return;
            if (menuRef.current.contains(event.target as Node)) return;
            setMenuOpen(false);
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, []);

    const groupedSettings = useMemo(() => {
        const groups: Record<string, RuntimeSettingField[]> = {};
        for (const field of settingsPayload?.fields || []) {
            if (!groups[field.category]) groups[field.category] = [];
            groups[field.category].push(field);
        }
        return groups;
    }, [settingsPayload]);

    const loadRuntimeSettings = async () => {
        setLoadingSettings(true);
        setSettingsError('');
        setSettingsInfo('');
        try {
            const res = await api.get<RuntimeSettingsResponse>('/settings/runtime');
            setSettingsPayload(res.data);
            const nextDraft: Record<string, string | boolean | number> = {};
            (res.data?.fields || []).forEach((field) => {
                nextDraft[field.key] = field.value;
            });
            setDraft(nextDraft);
        } catch (error: any) {
            setSettingsPayload(null);
            setSettingsError(error?.response?.data?.detail || 'Failed to load runtime settings');
        } finally {
            setLoadingSettings(false);
        }
    };

    const openSettings = () => {
        setSettingsOpen(true);
        setMenuOpen(false);
        void loadRuntimeSettings();
    };

    const openProfile = () => {
        setProfileOpen(true);
        setMenuOpen(false);
    };

    const handleLogout = () => {
        localStorage.removeItem('slm_token');
        window.location.href = '/login';
    };

    const saveSettings = async () => {
        if (!settingsPayload) return;
        const updates: Record<string, string | boolean | number> = {};
        for (const field of settingsPayload.fields) {
            if (!(field.key in draft)) continue;
            if (draft[field.key] !== field.value) {
                updates[field.key] = draft[field.key];
            }
        }
        if (Object.keys(updates).length === 0) {
            setSettingsInfo('No changes to save.');
            return;
        }

        setSavingSettings(true);
        setSettingsError('');
        setSettingsInfo('');
        try {
            const res = await api.put('/settings/runtime', { updates });
            const restartKeys = Array.isArray(res.data?.requires_restart_keys)
                ? res.data.requires_restart_keys.filter(Boolean)
                : [];
            if (restartKeys.length > 0) {
                setSettingsInfo(`Saved. Restart recommended for: ${restartKeys.join(', ')}`);
            } else {
                setSettingsInfo('Saved.');
            }
            await loadRuntimeSettings();
        } catch (error: any) {
            setSettingsError(error?.response?.data?.detail || 'Failed to save settings');
        } finally {
            setSavingSettings(false);
        }
    };

    return (
        <>
            <header className={`topbar ${withSidebar ? 'topbar--with-sidebar' : ''}`}>
                <div className="topbar-left">
                    <div className="topbar-kicker">Workspace</div>
                    <h1 className="topbar-title">{title}</h1>
                    {subtitle && <span className="topbar-subtitle">{subtitle}</span>}
                </div>
                <div className="topbar-actions">
                    {actions}
                    <button className="topbar-icon-btn" title="Help">
                        <CircleHelp size={16} />
                    </button>
                    <button className="topbar-icon-btn" title="Notifications">
                        <Bell size={16} />
                    </button>
                    <div className="topbar-user" ref={menuRef}>
                        <button
                            className="topbar-user-btn"
                            onClick={() => setMenuOpen((prev) => !prev)}
                        >
                            <span className="topbar-user-name">{currentUser?.username || 'User'}</span>
                            <span className="topbar-user-caret">▾</span>
                        </button>
                        {menuOpen && (
                            <div className="topbar-user-menu">
                                <div className="topbar-user-menu-head">
                                    <strong>{currentUser?.username || 'Unknown'}</strong>
                                    <span>{currentUser?.role || 'viewer'}</span>
                                </div>
                                <button className="topbar-user-menu-item" onClick={openProfile}>
                                    Profile
                                </button>
                                <button className="topbar-user-menu-item" onClick={openSettings}>
                                    Settings
                                </button>
                                <button className="topbar-user-menu-item" onClick={handleLogout}>
                                    Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {settingsOpen && (
                <div className="topbar-settings-backdrop" onClick={() => setSettingsOpen(false)}>
                    <div className="topbar-settings-modal" onClick={(event) => event.stopPropagation()}>
                        <div className="topbar-settings-head">
                            <div>
                                <h3>System Settings</h3>
                                <p>Manage runtime-configurable values normally set in `.env`/config.</p>
                            </div>
                            <button className="btn btn-ghost" onClick={() => setSettingsOpen(false)}>Close</button>
                        </div>

                        {loadingSettings && <div className="topbar-settings-note">Loading settings...</div>}
                        {settingsError && <div className="topbar-settings-error">{settingsError}</div>}
                        {settingsInfo && <div className="topbar-settings-info">{settingsInfo}</div>}

                        {!loadingSettings && settingsPayload && (
                            <div className="topbar-settings-body">
                                {Object.entries(groupedSettings).map(([category, fields]) => (
                                    <section key={category} className="topbar-settings-group">
                                        <h4>{category}</h4>
                                        {fields.map((field) => (
                                            <label key={field.key} className="topbar-settings-field">
                                                <div className="topbar-settings-field-head">
                                                    <span>{field.label}</span>
                                                    <div className="topbar-settings-field-badges">
                                                        <span className="badge badge-info">{field.source}</span>
                                                        {field.requires_restart && <span className="badge badge-warning">restart</span>}
                                                    </div>
                                                </div>
                                                <small>{field.description}</small>
                                                {field.type === 'bool' ? (
                                                    <input
                                                        type="checkbox"
                                                        checked={Boolean(draft[field.key])}
                                                        onChange={(event) => {
                                                            setDraft((prev) => ({ ...prev, [field.key]: event.target.checked }));
                                                        }}
                                                    />
                                                ) : field.options && field.options.length > 0 ? (
                                                    <select
                                                        className="input"
                                                        value={String(draft[field.key] ?? '')}
                                                        onChange={(event) => {
                                                            setDraft((prev) => ({ ...prev, [field.key]: event.target.value }));
                                                        }}
                                                    >
                                                        {field.options.map((opt) => (
                                                            <option key={opt} value={opt}>
                                                                {opt}
                                                            </option>
                                                        ))}
                                                    </select>
                                                ) : field.multiline ? (
                                                    <textarea
                                                        className="input topbar-settings-textarea"
                                                        value={String(draft[field.key] ?? '')}
                                                        onChange={(event) => {
                                                            setDraft((prev) => ({ ...prev, [field.key]: event.target.value }));
                                                        }}
                                                    />
                                                ) : (
                                                    <input
                                                        className="input"
                                                        type={field.type === 'int' ? 'number' : 'text'}
                                                        value={String(draft[field.key] ?? '')}
                                                        onChange={(event) => {
                                                            const nextValue =
                                                                field.type === 'int'
                                                                    ? Number(event.target.value || 0)
                                                                    : event.target.value;
                                                            setDraft((prev) => ({ ...prev, [field.key]: nextValue }));
                                                        }}
                                                    />
                                                )}
                                            </label>
                                        ))}
                                    </section>
                                ))}
                            </div>
                        )}

                        <div className="topbar-settings-actions">
                            <button className="btn btn-secondary" onClick={() => void loadRuntimeSettings()} disabled={loadingSettings || savingSettings}>
                                Refresh
                            </button>
                            <button className="btn btn-primary" onClick={() => void saveSettings()} disabled={loadingSettings || savingSettings}>
                                {savingSettings ? 'Saving...' : 'Save Settings'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {profileOpen && (
                <div className="topbar-settings-backdrop" onClick={() => setProfileOpen(false)}>
                    <div className="topbar-profile-modal" onClick={(event) => event.stopPropagation()}>
                        <div className="topbar-settings-head">
                            <div>
                                <h3>Profile</h3>
                                <p>Current authenticated user details.</p>
                            </div>
                            <button className="btn btn-ghost" onClick={() => setProfileOpen(false)}>Close</button>
                        </div>
                        <div className="topbar-profile-body">
                            <div className="topbar-profile-row">
                                <span>Username</span>
                                <strong>{currentUser?.username || 'Unknown'}</strong>
                            </div>
                            <div className="topbar-profile-row">
                                <span>Role</span>
                                <strong>{currentUser?.role || 'viewer'}</strong>
                            </div>
                            <div className="topbar-profile-row">
                                <span>User ID</span>
                                <strong>{currentUser?.userId ?? 'n/a'}</strong>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
