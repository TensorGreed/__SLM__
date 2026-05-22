/**
 * Static catalog mapping each pipeline tab to a walkthrough video +
 * chapter timestamp. Sourced from `docs-demo/youtube/videos.md`.
 *
 * `youtubeId` is `null` for videos that haven't been published yet —
 * the embed modal renders a "watch on @TensorGreed channel" fallback
 * card in that case so the affordance still lands even before upload.
 * Filling in `youtubeId` here is the only change needed when the
 * videos go live.
 */

import type { TabKey } from '../types';

export interface VideoChapter {
    /** Display label, e.g. "Data tab". */
    label: string;
    /** Offset within the video, in seconds. */
    timeSeconds: number;
}

export interface VideoMeta {
    /** Stable short id used as catalog key — "v03", "v07", … */
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
        title: 'Fine-Tune an SLM on Support Tickets — Full Pipeline Walkthrough',
        chapters: [
            { label: 'Intro', timeSeconds: t('00:00') },
            { label: 'Data tab', timeSeconds: t('00:15') },
            { label: 'Cleaning', timeSeconds: t('00:34') },
            { label: 'Gold Set', timeSeconds: t('00:55') },
            { label: 'Synthetic', timeSeconds: t('01:12') },
            { label: 'Dataset Prep', timeSeconds: t('01:29') },
            { label: 'Tokenization', timeSeconds: t('01:48') },
            { label: 'Training Config', timeSeconds: t('02:04') },
            { label: 'Evaluation & wrap', timeSeconds: t('02:27') },
        ],
    },
    v07: {
        id: 'v07',
        youtubeId: null,
        title: 'Train a 135M LoRA in 12 Seconds — Real Celery, Real LoRA, Local GPU',
        chapters: [
            { label: 'Intro', timeSeconds: t('00:00') },
            { label: 'Training Config recap', timeSeconds: t('00:17') },
            { label: 'Kickoff (API)', timeSeconds: t('00:34') },
            { label: 'Watching status: running', timeSeconds: t('00:46') },
            { label: 'Completed with metrics', timeSeconds: t('00:59') },
            { label: 'Wrap', timeSeconds: t('01:16') },
        ],
    },
    v08: {
        id: 'v08',
        youtubeId: null,
        title: 'Evaluate a Fine-Tuned SLM Against a 200-Row Gold Set',
        chapters: [
            { label: 'Intro', timeSeconds: t('00:00') },
            { label: 'Eval setup', timeSeconds: t('00:11') },
            { label: 'Kickoff (POST /run-heldout)', timeSeconds: t('00:29') },
            { label: 'Running eval on gold_dev', timeSeconds: t('00:38') },
            { label: 'Results: Auto-Gate & predictions', timeSeconds: t('00:52') },
            { label: 'Wrap', timeSeconds: t('01:08') },
        ],
    },
    v09: {
        id: 'v09',
        youtubeId: null,
        title: 'Compress an SLM to a 105 MB GGUF — Merge LoRA + llama.cpp Quantize',
        chapters: [
            { label: 'Intro', timeSeconds: t('00:00') },
            { label: 'Compression form', timeSeconds: t('00:15') },
            { label: 'Merge LoRA + quantize', timeSeconds: t('00:31') },
            { label: 'GGUF on disk (105 MB)', timeSeconds: t('00:49') },
            { label: 'Export tab', timeSeconds: t('01:03') },
            { label: 'Export registered', timeSeconds: t('01:18') },
            { label: 'Wrap', timeSeconds: t('01:31') },
        ],
    },
};

/**
 * Channel URL used when a video's `youtubeId` isn't set yet.
 */
export const CHANNEL_URL = 'https://www.youtube.com/@TensorGreed';

/**
 * Maps a pipeline tab to the video + chapter that explains it.
 * `chapterIndex` is the index into the video's `chapters` array.
 */
export const TAB_VIDEO_MAP: Record<TabKey, { videoId: string; chapterIndex: number }> = {
    data: { videoId: 'v03', chapterIndex: 1 },          // 00:15 Data tab
    cleaning: { videoId: 'v03', chapterIndex: 2 },      // 00:34 Cleaning
    goldset: { videoId: 'v03', chapterIndex: 3 },       // 00:55 Gold Set
    synthetic: { videoId: 'v03', chapterIndex: 4 },     // 01:12 Synthetic
    dataprep: { videoId: 'v03', chapterIndex: 5 },      // 01:29 Dataset Prep
    tokenization: { videoId: 'v03', chapterIndex: 6 },  // 01:48 Tokenization
    training: { videoId: 'v07', chapterIndex: 1 },      // 00:17 Training Config recap
    eval: { videoId: 'v08', chapterIndex: 1 },          // 00:11 Eval setup
    compression: { videoId: 'v09', chapterIndex: 1 },   // 00:15 Compression form
    export: { videoId: 'v09', chapterIndex: 4 },        // 01:03 Export tab
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
