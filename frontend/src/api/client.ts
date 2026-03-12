import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

api.interceptors.request.use((config) => {
    const token = localStorage.getItem('slm_token');
    if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

let redirectingToLogin = false;

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        const status = error.response?.status;
        const requestUrl = String(error.config?.url || '');

        if (status === 401) {
            localStorage.removeItem('slm_token');

            const isLocalLoginRequest = requestUrl.includes('/auth/local/login');
            const isSsoLoginRequest = requestUrl.includes('/auth/sso/login');
            const isAlreadyOnLoginPage = typeof window !== 'undefined' && window.location.pathname === '/login';

            if (!isLocalLoginRequest && !isSsoLoginRequest && !isAlreadyOnLoginPage && !redirectingToLogin) {
                redirectingToLogin = true;
                window.location.href = '/login';
            }
        }

        const message = error.response?.data?.detail || error.message || 'An error occurred';
        console.error('[API Error]', message);
        return Promise.reject(error);
    }
);

export default api;
