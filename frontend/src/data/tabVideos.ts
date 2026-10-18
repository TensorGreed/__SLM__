/**
 * Static catalog mapping each workspace step to a walkthrough video +
 * chapter timestamp. Sourced from `docs-demo/youtube/videos.md` and the
 * narration scripts under `docs-demo/scripts/narration/`.
 *
 * Epic U phase U7 (2026-10-18): video 03 was re-recorded on the
 * six-destination workspace (Overview → Data's five steps → Train → Results
 * → Ship); its chapters below are the new take's. `youtubeId` is `null`
 * until the new take is uploaded — the embed modal then renders a "watch on
 * @TensorGreed" card — so the old take's chapter offsets are never
 * deep-linked. Filling in `youtubeId` is the only change needed when it
 * goes live (see `docs-demo/videos/03-support-faq-pipeline/PUBLISH.md`).
 */

import type { TabKey } from '../types';

export interface VideoChapter {
    /** Display label, e.g. "Data · Sources". */
    label: string;
    /** Offset within the video, in seconds. */
    timeSeconds: number;
}

export interface VideoMeta {
    /** Stable short id used as catalog key — "v03", … */
    id: string;
    /**
     * The 11-char YouTube video id (e.g. "dQw4w9WgXcQ").
     * `null` means the video isn't published yet; the modal degrades
     * to a channel-link card.
     */
    youtubeId: string | null;
    /** Human title shown above the player. */
    title: string;
    /** YouTube chapters in display order. */
    chapters: VideoChapter[];
}

/**
 * Inline mm:ss helper. Keeps the catalog readable.
 */
function t(timecode: string): number {
    const [mm, ss] = timecode.split(':').map((part) => parseInt(part, 10));
    return mm * 60 + ss;
}

export const VIDEOS: Record<string, VideoMeta> = {
    v03: {
        id: 'v03',
        youtubeId: null,
        title: 'Fine-Tune an SLM on Support Tickets — Workspace Walkthrough',
        chapters: [
            { label: 'Overview', timeSeconds: t('00:00') },
            { label: 'Data · Sources', timeSeconds: t('00:18') },
            { label: 'Data · Prepare', timeSeconds: t('00:32') },
            { label: 'Data · Labels', timeSeconds: t('00:43') },
            { label: 'Data · Generate', timeSeconds: t('00:59') },
            { label: 'Data · Split', timeSeconds: t('01:16') },
            { label: 'Train', timeSeconds: t('01:33') },
            { label: 'Train · More options', timeSeconds: t('01:50') },
            { label: 'Results · Verdict', timeSeconds: t('02:07') },
            { label: 'Results · Failures', timeSeconds: t('02:19') },
            { label: 'Results · Compare', timeSeconds: t('02:29') },
            { label: 'Results · Try it', timeSeconds: t('02:41') },
            { label: 'Ship · Package', timeSeconds: t('02:50') },
            { label: 'Ship · Publish', timeSeconds: t('03:02') },
            { label: 'Ship · Deploy', timeSeconds: t('03:10') },
            { label: 'Ship · Monitor', timeSeconds: t('03:22') },
            { label: 'Wrap', timeSeconds: t('03:34') },
        ],
    },
};

/**
 * Channel URL used when a video's `youtubeId` isn't set yet.
 */
export const CHANNEL_URL = 'https://www.youtube.com/@TensorGreed';

/**
 * Maps a step to the video + chapter that explains it.
 * `chapterIndex` is the index into the video's `chapters` array.
 */
export const TAB_VIDEO_MAP: Record<TabKey, { videoId: string; chapterIndex: number }> = {
    data: { videoId: 'v03', chapterIndex: 1 },          // Data · Sources
    cleaning: { videoId: 'v03', chapterIndex: 2 },      // Data · Prepare
    goldset: { videoId: 'v03', chapterIndex: 3 },       // Data · Labels
    synthetic: { videoId: 'v03', chapterIndex: 4 },     // Data · Generate
    dataprep: { videoId: 'v03', chapterIndex: 5 },      // Data · Split
    training: { videoId: 'v03', chapterIndex: 6 },      // Train
    tokenization: { videoId: 'v03', chapterIndex: 7 },  // Train · More options
    eval: { videoId: 'v03', chapterIndex: 8 },          // Results · Verdict
    compression: { videoId: 'v03', chapterIndex: 12 },  // Ship · Package
    export: { videoId: 'v03', chapterIndex: 12 },       // Ship · Package
};

export interface TabVideo {
    video: VideoMeta;
    chapter: VideoChapter;
}

/**
 * Resolves the tab → video + chapter pair from the catalog.
 * Returns `null` only if the catalog is misconfigured for a tab — the
 * caller should treat that as "no walkthrough for this tab" rather
 * than throwing in render.
 */
export function getTabVideo(tabKey: TabKey): TabVideo | null {
    const entry = TAB_VIDEO_MAP[tabKey];
    if (!entry) {
        return null;
    }
    const video = VIDEOS[entry.videoId];
    if (!video) {
        return null;
    }
    const chapter = video.chapters[entry.chapterIndex];
    if (!chapter) {
        return null;
    }
    return { video, chapter };
}

/**
 * Builds the YouTube embed URL with a `start` query for chapter
 * deep-linking. Caller is responsible for handling the
 * `youtubeId === null` case before calling this.
 */
export function buildEmbedUrl(youtubeId: string, startSeconds: number): string {
    const url = new URL(`https://www.youtube.com/embed/${youtubeId}`);
    if (startSeconds > 0) {
        url.searchParams.set('start', String(startSeconds));
    }
    // rel=0 keeps "More videos" suggestions scoped to the same channel,
    // not random YouTube. modestbranding is deprecated but harmless.
    url.searchParams.set('rel', '0');
    return url.toString();
}

/**
 * Watch-on-YouTube fallback URL (full site, with timestamp), used as
 * the "open on YouTube" external link from inside the modal.
 */
export function buildWatchUrl(youtubeId: string, startSeconds: number): string {
    const url = new URL(`https://www.youtube.com/watch`);
    url.searchParams.set('v', youtubeId);
    if (startSeconds > 0) {
        url.searchParams.set('t', `${startSeconds}s`);
    }
    return url.toString();
}

/**
 * Formats a number of seconds back into a human-readable mm:ss.
 */
export function formatTimecode(totalSeconds: number): string {
    const mm = Math.floor(totalSeconds / 60);
    const ss = totalSeconds % 60;
    return `${mm}:${String(ss).padStart(2, '0')}`;
}
