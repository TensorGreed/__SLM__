/**
 * Axios HTTP client configured with /api base URL and auth-token interceptor that redirects on 401.
 */

import axios from 'axios';

import { noticeResponse } from './jobWake';

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
    // A FormData body must go out as multipart. The JSON default above would make axios
    // serialise it to JSON (dropping every File), and an explicit 'multipart/form-data'
    // without a boundary is just as broken, so clear the header and let the browser set
    // 'multipart/form-data; boundary=…' itself. Interceptors run before transformRequest.
    if (typeof FormData !== 'undefined' && config.data instanceof FormData && config.headers) {
        config.headers.delete('Content-Type');
    }
    return config;
});

let redirectingToLogin = false;

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => {
        // A call that started a Job wakes the jobs store's poll (it stops itself when idle), whichever page made it.
        noticeResponse(response);
        return response;
    },
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
