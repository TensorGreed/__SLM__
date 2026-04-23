import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import api from '../api/client';
import './SSOLoginPage.css';

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
        } catch (err) {
            console.error('Local login failed:', err);
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Invalid credentials');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="card auth-card">
                <div className="auth-logo">
                    <span className="auth-logo-mark">BS</span>
                </div>
                <h1 className="auth-title">
                    BrewSLM
                </h1>
                <p className="auth-subtitle">
                    Secure sign-in to your BrewSLM workspace
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
                    <form onSubmit={handleLocalLogin} className="auth-form">
                        <div className="auth-field">
                            <label>Username</label>
                            <input
                                required
                                type="text"
                                className="input"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter your username"
                            />
                        </div>
                        <div className="auth-field">
                            <label>Password</label>
                            <input
                                required
                                type="password"
                                className="input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="API Key or Password"
                            />
                        </div>
                        {error && (
                            <div className="auth-error">
                                {error}
                            </div>
                        )}
                        <button
                            type="submit"
                            className="btn btn-primary"
                            disabled={loading || !username || !password}
                            style={{ width: '100%', padding: 'var(--space-md)' }}
                        >
                            {loading ? 'Signing in...' : 'Sign in'}
                        </button>
                    </form>
                )}
            </div>
        </div>
    );
}
