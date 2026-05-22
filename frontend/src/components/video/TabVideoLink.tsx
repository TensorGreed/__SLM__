/**
 * TabVideoLink — small "▶ Watch the 2-minute walkthrough" affordance
 * rendered above each pipeline-tab body. Click → opens a modal with
 * the matching video deep-linked to the relevant chapter.
 *
 * Renders nothing if the tab has no catalogued video (so wiring this
 * into a future tab is safe without a code change in the page shell).
 */

import { useState } from 'react';

import type { TabKey } from '../../types';
import { getTabVideo, formatTimecode } from '../../data/tabVideos';
import YouTubeEmbedModal from './YouTubeEmbedModal';

import './TabVideoLink.css';

interface Props {
    tabKey: TabKey;
}

export default function TabVideoLink({ tabKey }: Props) {
    const [open, setOpen] = useState(false);
    const entry = getTabVideo(tabKey);

    if (!entry) {
        return null;
    }

    const { video, chapter } = entry;

    return (
        <>
            <button
                type="button"
                className="tab-video-link"
                onClick={() => setOpen(true)}
                aria-label={`Watch the 2-minute walkthrough for ${chapter.label}`}
            >
                <span className="tab-video-link__icon" aria-hidden="true">▶</span>
                <span className="tab-video-link__label">
                    Watch the walkthrough — <strong>{chapter.label}</strong>
                </span>
                <span className="tab-video-link__time" aria-hidden="true">
                    {formatTimecode(chapter.timeSeconds)}
                </span>
            </button>
            {open && (
                <YouTubeEmbedModal
                    video={video}
                    initialChapterIndex={video.chapters.indexOf(chapter)}
                    onClose={() => setOpen(false)}
                />
            )}
        </>
    );
}
