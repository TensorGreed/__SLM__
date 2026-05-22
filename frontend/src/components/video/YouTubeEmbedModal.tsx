/**
 * YouTubeEmbedModal — in-page modal for the inline-video-embed
 * feature. Plays the video deep-linked to a chapter, lists every
 * chapter on the side for navigation within the same video, and
 * provides an "open on YouTube" external link.
 *
 * When `video.youtubeId` is `null` (video not yet published), the
 * embed iframe is replaced with a small placeholder card pointing
 * to the channel.
 */

import { useEffect, useState } from 'react';

import type { VideoMeta } from '../../data/tabVideos';
import {
    CHANNEL_URL,
    buildEmbedUrl,
    buildWatchUrl,
    formatTimecode,
} from '../../data/tabVideos';

import './YouTubeEmbedModal.css';

interface Props {
    video: VideoMeta;
    initialChapterIndex: number;
    onClose: () => void;
}

export default function YouTubeEmbedModal({
    video,
    initialChapterIndex,
    onClose,
}: Props) {
    const safeInitial = Math.max(
        0,
        Math.min(initialChapterIndex, video.chapters.length - 1),
    );
    const [activeChapterIndex, setActiveChapterIndex] = useState(safeInitial);
    const chapter = video.chapters[activeChapterIndex] ?? video.chapters[0];

    useEffect(() => {
        const handleKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [onClose]);

    const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    const isPublished = video.youtubeId !== null;
    // When the user jumps chapters, swap the iframe `key` so it
    // remounts with the new ?start= and actually seeks. Pure src
    // changes don't always trigger YouTube to seek.
    const iframeKey = `${video.id}-${activeChapterIndex}`;

    return (
        <div
            className="modal-overlay video-embed-overlay"
            role="dialog"
            aria-modal="true"
            aria-labelledby="video-embed-title"
            onClick={handleBackdropClick}
        >
            <div className="modal video-embed-modal">
                <header className="video-embed-modal__head">
                    <h3 id="video-embed-title" className="video-embed-modal__title">
                        {video.title}
                    </h3>
                    <button
                        type="button"
                        className="video-embed-modal__close"
                        onClick={onClose}
                        aria-label="Close walkthrough"
                    >
                        ✕
                    </button>
                </header>

                <div className="video-embed-modal__body">
                    <div className="video-embed-modal__player">
                        {isPublished ? (
                            <iframe
                                key={iframeKey}
                                className="video-embed-modal__iframe"
                                src={buildEmbedUrl(
                                    video.youtubeId as string,
                                    chapter.timeSeconds,
                                )}
                                title={video.title}
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                allowFullScreen
                                referrerPolicy="strict-origin-when-cross-origin"
                            />
                        ) : (
                            <div className="video-embed-modal__unpublished">
                                <p className="video-embed-modal__unpublished-headline">
                                    This walkthrough isn't on YouTube yet.
                                </p>
                                <p className="video-embed-modal__unpublished-sub">
                                    The video and its chapter list ship with the
                                    repo; the recording lands on the channel as
                                    the series rolls out.
                                </p>
                                <a
                                    href={CHANNEL_URL}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="btn btn-secondary video-embed-modal__channel-link"
                                >
                                    Visit @TensorGreed channel →
                                </a>
                            </div>
                        )}
                    </div>

                    <aside
                        className="video-embed-modal__chapters"
                        aria-label="Chapters"
                    >
                        <div className="video-embed-modal__chapters-head">
                            Chapters
                        </div>
                        <ul className="video-embed-modal__chapter-list">
                            {video.chapters.map((c, idx) => {
                                const active = idx === activeChapterIndex;
                                return (
                                    <li key={idx}>
                                        <button
                                            type="button"
                                            className={
                                                'video-embed-modal__chapter'
                                                + (active ? ' is-active' : '')
                                            }
                                            onClick={() => setActiveChapterIndex(idx)}
                                            aria-current={active ? 'true' : undefined}
                                        >
                                            <span className="video-embed-modal__chapter-time">
                                                {formatTimecode(c.timeSeconds)}
                                            </span>
                                            <span className="video-embed-modal__chapter-label">
                                                {c.label}
                                            </span>
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    </aside>
                </div>

                <footer className="video-embed-modal__footer">
                    {isPublished ? (
                        <a
                            href={buildWatchUrl(
                                video.youtubeId as string,
                                chapter.timeSeconds,
                            )}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="video-embed-modal__external"
                        >
                            Open on YouTube ↗
                        </a>
                    ) : (
                        <span className="video-embed-modal__external video-embed-modal__external--disabled">
                            Not published yet
                        </span>
                    )}
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onClose}
                    >
                        Close
                    </button>
                </footer>
            </div>
        </div>
    );
}
