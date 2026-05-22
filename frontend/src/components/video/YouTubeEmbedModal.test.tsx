import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import YouTubeEmbedModal from './YouTubeEmbedModal';
import type { VideoMeta } from '../../data/tabVideos';

const PUBLISHED: VideoMeta = {
    id: 'vTest',
    youtubeId: 'abcd1234EFG',
    title: 'Published Walkthrough',
    chapters: [
        { label: 'Intro', timeSeconds: 0 },
        { label: 'Middle', timeSeconds: 90 },
        { label: 'Outro', timeSeconds: 200 },
    ],
};

const UNPUBLISHED: VideoMeta = {
    id: 'vUnpub',
    youtubeId: null,
    title: 'Coming Soon Walkthrough',
    chapters: [
        { label: 'Intro', timeSeconds: 0 },
        { label: 'Action', timeSeconds: 45 },
    ],
};

describe('YouTubeEmbedModal', () => {
    it('renders an iframe deep-linked to the initial chapter timestamp', () => {
        const { container } = render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={1}
                onClose={vi.fn()}
            />,
        );
        const iframe = container.querySelector('iframe');
        expect(iframe).not.toBeNull();
        expect(iframe?.getAttribute('src')).toContain(
            'youtube.com/embed/abcd1234EFG',
        );
        expect(iframe?.getAttribute('src')).toContain('start=90');
    });

    it('updates the iframe src when the user picks a different chapter', async () => {
        const { container } = render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={0}
                onClose={vi.fn()}
            />,
        );
        // Iframe at Intro (0s) has no start param.
        let iframe = container.querySelector('iframe');
        expect(iframe?.getAttribute('src')).not.toContain('start=');

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /3:20 Outro/i }),
        );

        iframe = container.querySelector('iframe');
        expect(iframe?.getAttribute('src')).toContain('start=200');
    });

    it('marks the active chapter via aria-current', async () => {
        render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={0}
                onClose={vi.fn()}
            />,
        );
        const introButton = screen.getByRole('button', { name: /0:00 Intro/i });
        expect(introButton).toHaveAttribute('aria-current', 'true');

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /1:30 Middle/i }),
        );

        expect(
            screen.getByRole('button', { name: /1:30 Middle/i }),
        ).toHaveAttribute('aria-current', 'true');
        expect(introButton).not.toHaveAttribute('aria-current', 'true');
    });

    it('shows a channel fallback card when the video is unpublished', () => {
        render(
            <YouTubeEmbedModal
                video={UNPUBLISHED}
                initialChapterIndex={0}
                onClose={vi.fn()}
            />,
        );
        // No iframe in the unpublished state.
        expect(document.querySelector('iframe')).toBeNull();
        expect(
            screen.getByText(/walkthrough isn't on YouTube yet/i),
        ).toBeInTheDocument();
        // Channel link is present and points to @TensorGreed.
        const channelLink = screen.getByRole('link', {
            name: /TensorGreed channel/i,
        });
        expect(channelLink).toHaveAttribute(
            'href',
            'https://www.youtube.com/@TensorGreed',
        );
    });

    it('shows a disabled "Not published yet" footer when unpublished', () => {
        render(
            <YouTubeEmbedModal
                video={UNPUBLISHED}
                initialChapterIndex={0}
                onClose={vi.fn()}
            />,
        );
        expect(screen.getByText(/Not published yet/i)).toBeInTheDocument();
        expect(
            screen.queryByRole('link', { name: /Open on YouTube/i }),
        ).not.toBeInTheDocument();
    });

    it('renders the "Open on YouTube" external link with chapter timestamp when published', () => {
        render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={2}
                onClose={vi.fn()}
            />,
        );
        const link = screen.getByRole('link', { name: /Open on YouTube/i });
        const href = link.getAttribute('href') ?? '';
        expect(href).toContain('youtube.com/watch');
        expect(href).toContain('v=abcd1234EFG');
        expect(href).toContain('t=200s');
        expect(link).toHaveAttribute('target', '_blank');
    });

    it('closes when the close button is clicked', async () => {
        const onClose = vi.fn();
        render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={0}
                onClose={onClose}
            />,
        );
        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Close walkthrough/i }),
        );
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('closes on Escape', async () => {
        const onClose = vi.fn();
        render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={0}
                onClose={onClose}
            />,
        );
        const user = userEvent.setup();
        await user.keyboard('{Escape}');
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('closes on backdrop click but not on content click', async () => {
        const onClose = vi.fn();
        render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={0}
                onClose={onClose}
            />,
        );
        const user = userEvent.setup();

        // Click inside the modal body — should NOT close.
        const dialog = screen.getByRole('dialog');
        const title = within(dialog).getByText('Published Walkthrough');
        await user.click(title);
        expect(onClose).not.toHaveBeenCalled();

        // Click on the backdrop (the overlay element itself) — closes.
        await user.click(dialog);
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('clamps an out-of-range initialChapterIndex to a valid chapter', () => {
        const { container } = render(
            <YouTubeEmbedModal
                video={PUBLISHED}
                initialChapterIndex={99}
                onClose={vi.fn()}
            />,
        );
        // Should clamp to the last chapter (index 2 → 200s).
        const iframe = container.querySelector('iframe');
        expect(iframe?.getAttribute('src')).toContain('start=200');
    });
});
