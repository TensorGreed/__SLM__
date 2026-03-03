import './TopBar.css';

interface TopBarProps {
    title: string;
    subtitle?: string;
    actions?: React.ReactNode;
}

export default function TopBar({ title, subtitle, actions }: TopBarProps) {
    return (
        <header className="topbar">
            <div className="topbar-left">
                <h1 className="topbar-title">{title}</h1>
                {subtitle && <span className="topbar-subtitle">{subtitle}</span>}
            </div>
            {actions && <div className="topbar-actions">{actions}</div>}
        </header>
    );
}
