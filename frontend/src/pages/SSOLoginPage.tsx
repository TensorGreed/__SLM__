import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Activity } from 'lucide-react';
import api from '../api/client';

export default function SSOLoginPage() {
    const [searchParams] = useSearchParams();
    const [authConfig, setAuthConfig] = useState<{ auth_enabled: boolean; sso_enabled: boolean } | null>(null);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const token = searchParams.get('token');
        if (token) {
            localStorage.setItem('slm_token', token);
            window.location.href = '/';
        }
    }, [searchParams]);

    useEffect(() => {
        api.get('/auth/config')
            .then(res => setAuthConfig(res.data))
            .catch(err => console.error('Failed to fetch auth config:', err));
    }, []);

    const handleSSOLogin = () => {
        window.location.href = '/api/auth/sso/login';
    };

    const handleLocalLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const res = await api.post('/auth/local/login', { username, password });
            localStorage.setItem('slm_token', res.data.token);
            window.location.href = '/';
        } catch (err: any) {
            console.error('Local login failed:', err);
            setError(err.response?.data?.detail || 'Invalid credentials');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '100vh',
            background: 'var(--bg-primary)'
        }}>
            <div className="card" style={{ maxWidth: 400, width: '100%', textAlign: 'center', padding: 'var(--space-2xl)' }}>
                <div style={{
                    width: 48,
                    height: 48,
                    borderRadius: 12,
                    background: 'linear-gradient(135deg, #a855f7 0%, #ec4899 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto var(--space-xl)',
                    boxShadow: '0 4px 20px rgba(168, 85, 247, 0.4)'
                }}>
                    <Activity color="white" size={24} />
                </div>
                <h1 style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, marginBottom: 'var(--space-sm)' }}>
                    SLM Studio
                </h1>
                <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-2xl)' }}>
                    Enterprise authentication required
                </p>

                {authConfig?.sso_enabled ? (
                    <button
                        onClick={handleSSOLogin}
                        className="btn btn-primary"
                        style={{ width: '100%', padding: 'var(--space-md)' }}
                    >
                        Sign in with SSO
                    </button>
                ) : (
                    <form onSubmit={handleLocalLogin} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                        <div style={{ textAlign: 'left' }}>
                            <label style={{ display: 'block', marginBottom: 'var(--space-xs)', color: 'var(--text-secondary)' }}>Username</label>
                            <input
                                required
                                type="text"
                                className="input"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter your username"
                                style={{ width: '100%', boxSizing: 'border-box' }}
                            />
                        </div>
                        <div style={{ textAlign: 'left' }}>
                            <label style={{ display: 'block', marginBottom: 'var(--space-xs)', color: 'var(--text-secondary)' }}>Password</label>
                            <input
                                required
                                type="password"
                                className="input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="API Key or Password"
                                style={{ width: '100%', boxSizing: 'border-box' }}
                            />
                        </div>
                        {error && (
                            <div style={{ color: 'var(--danger)', fontSize: 'var(--font-size-sm)' }}>
                                {error}
                            </div>
                        )}
                        <button
                            type="submit"
                            className="btn btn-primary"
                            disabled={loading || !username || !password}
                            style={{ width: '100%', padding: 'var(--space-md)', marginTop: 'var(--space-sm)' }}
                        >
                            {loading ? 'Signing in...' : 'Sign in'}
                        </button>
                    </form>
                )}
            </div>
        </div>
    );
}
