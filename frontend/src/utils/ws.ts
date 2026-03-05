export function buildWsUrl(path: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const token = localStorage.getItem('slm_token');
    const separator = path.includes('?') ? '&' : '?';
    const authQuery = token ? `${separator}token=${encodeURIComponent(token)}` : '';
    return `${protocol}://${window.location.host}${path}${authQuery}`;
}
