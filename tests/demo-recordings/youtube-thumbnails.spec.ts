import { test, expect } from '@playwright/test';
import * as path from 'path';

// Renders each <section class="thumb" data-thumb="..."> in
// docs-demo/youtube/thumbnails-source.html as a separate 1280x720
// PNG under docs-demo/youtube/thumbnails/. Not a recording — pure
// image extraction. Run once whenever the thumbnail HTML changes.

const SOURCE = path.resolve(
    __dirname,
    '../../docs-demo/youtube/thumbnails-source.html',
);
const OUT_DIR = path.resolve(__dirname, '../../docs-demo/youtube/thumbnails');

const SLIDES = [
    'playlist-cover',
    'v01',
    'v02',
    'v03',
    'v04',
    'v05',
    'v06',
    'v09',
    'v10',
    'v11',
    'v12',
    'v14',
];

test('Render YouTube thumbnails', async ({ page }) => {
    test.setTimeout(2 * 60_000);
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto(`file://${SOURCE}`);
    await page.waitForLoadState('domcontentloaded');

    for (const slug of SLIDES) {
        const el = page.locator(`section.thumb[data-thumb="${slug}"]`);
        await expect(el).toBeVisible();
        // Wait a beat for fonts to settle
        await page.waitForTimeout(150);
        const out = path.join(OUT_DIR, `${slug}.png`);
        await el.screenshot({ path: out });
        console.log(`[thumb] ${slug} → ${out}`);
    }
});
