# YouTube Publishing Kit — BrewSLM Series

Everything needed to upload the 11-video BrewSLM series to YouTube
under **@TensorGreed**. Author: Anurag Jain. Public, indexed for
ML/AI-engineer search intent.

## What's in here

```
docs-demo/youtube/
├── README.md                 ← this file
├── playlist.md               ← playlist title / description / tags / privacy review
├── videos.md                 ← all 11 videos (title / description / timestamps / tags)
├── thumbnails-source.html    ← the deck used to render thumbnails
└── thumbnails/
    ├── playlist-cover.png    ← 1280×720, use as playlist banner image
    ├── v01.png               ← per-video thumbnails
    ├── v02.png
    ├── v03.png
    ├── v04.png
    ├── v05.png
    ├── v06.png
    ├── v09.png
    ├── v10.png
    ├── v11.png
    ├── v12.png
    └── v14.png
```

## Upload order (copy-paste-ready)

1. **Create the playlist first** using
   [`playlist.md`](playlist.md) — title, description, tags. Upload
   `thumbnails/playlist-cover.png` as the playlist's display image.
2. **Upload videos in this order**:

   | # | Local file | YouTube title to paste | Thumbnail |
   |---|---|---|---|
   | 1 | `docs-demo/recordings/raw/01-slm-101-narrated.mp4` | Small Language Model 101: When to Use SLMs Over LLMs | `thumbnails/v01.png` |
   | 2 | `02-brewslm-quickstart-narrated.mp4` | BrewSLM in 2 Minutes — Local Fine-Tuning Workspace Quickstart | `thumbnails/v02.png` |
   | 3 | `03-support-faq-pipeline-narrated.mp4` | Fine-Tune an SLM on Support Tickets — Full Pipeline Walkthrough | `thumbnails/v03.png` |
   | 4 | `04-pii-detector-pipeline-narrated.mp4` | Build a PII Detector with a Small Language Model — End-to-End | `thumbnails/v04.png` |
   | 5 | `05-sentiment-classifier-pipeline-narrated.mp4` | Sentiment Classifier with a Small Language Model — Mobile CPU Target | `thumbnails/v05.png` |
   | 6 | `06-byo-custom-samples-narrated.mp4` | Bring Your Own Data: Train an SLM on Your Own CSV in 90 Seconds | `thumbnails/v06.png` |
   | 7 | `09-training-run-narrated.mp4` | Train a 135M LoRA in 12 Seconds — Real Celery, Real LoRA, Local GPU | `thumbnails/v09.png` |
   | 8 | `10-evaluation-narrated.mp4` | Evaluate a Fine-Tuned SLM Against a 200-Row Gold Set | `thumbnails/v10.png` |
   | 9 | `11-compression-export-narrated.mp4` | Compress an SLM to a 105 MB GGUF — Merge LoRA + llama.cpp Quantize | `thumbnails/v11.png` |
   | 10 | `12-final-model-usage-narrated.mp4` | Serve a Fine-Tuned SLM with Ollama — Closing the Loop | `thumbnails/v12.png` |
   | 11 | `14-architecture-narrated.mp4` | BrewSLM Architecture: 5 Processes, 1 Machine, Local-First AI | `thumbnails/v14.png` |

3. For each video: paste the matching description block from
   [`videos.md`](videos.md), paste the tag list, upload the
   thumbnail, set audience = "Not made for kids", set language =
   English, set category = "Science & Technology", set license =
   "Standard YouTube License".

4. Add each video to the playlist as you publish them.

## Why these titles + descriptions

- **Front-loaded keywords** — every title starts with the search
  term most likely to land it ("Fine-Tune", "Train", "PII
  Detector", "Sentiment Classifier", etc.) rather than burying the
  hook behind the series name.
- **Specific numbers in titles where they're real** — "135M LoRA",
  "12 Seconds", "105 MB GGUF", "200-Row Gold Set". Clickability +
  honesty.
- **Hashtag rule**: YouTube shows the first 3 hashtags from the
  description above the title. Each video's description ends with
  a curated 3-5 hashtag set — only the first 3 will surface
  publicly; the rest are flavor.
- **Chapter markers**: computed directly from
  `tts/audio/v*-durations.json`. Drop-in compatible with YouTube's
  auto-chapter feature — the timestamps render as clickable
  segments on the progress bar.

## Why the thumbnails look the way they do

- **Dark CRT/terminal aesthetic** matches the platform's existing
  visual language (the Lab Journal feature uses the same palette).
- **High contrast** — phosphor green on near-black, white headline
  text. Reads cleanly at 320×180 (YouTube's smallest grid size on
  mobile) without losing the headline.
- **Episode number, top-left** — large enough to scan in a playlist
  grid. Helps viewers find the next video in sequence.
- **`@TensorGreed` mark, top-right** — branding without dominating
  the layout. Same position on every thumbnail.
- **Hook line in green caps** — the one-line angle. Different per
  episode (`SPAN-LEVEL ENTITY EXTRACTION · 10 PII TYPES`,
  `REAL CELERY WORKER · REAL LORA · LOCAL GPU`).
- **Headline (2-3 lines, mixed case)** — the title hook, with one
  word or phrase in accent green for emphasis.
- **Status badge bottom-left** — keeps the "what kind of video is
  this" answer visible at a glance.

## Regenerating thumbnails

If you change a headline:

1. Edit [`thumbnails-source.html`](thumbnails-source.html).
2. From repo root: `npx playwright test youtube-thumbnails.spec.ts
   --project chromium`. Takes <5 seconds. Writes new PNGs to
   `thumbnails/`.

The spec lives at
[`tests/demo-recordings/youtube-thumbnails.spec.ts`](../../tests/demo-recordings/youtube-thumbnails.spec.ts).
No backend required — the source HTML is fully self-contained.

## Privacy review before clicking "Public"

See [`playlist.md`](playlist.md#privacy-review-before-publishing-public)
for the full review. Short version: the file paths shown in some
frames (`/home/anuragj/Desktop/GitHub/__SLM__/`) leak the
machine's home directory + username. Low impact, but you might
want to mentally clear that before publishing. Everything else
(API keys, git author, etc.) is either masked, not in any frame,
or unsensitive.

## Optional follow-ups (not in this commit)

These are deliberate non-goals for this kit; mentioned in case
you want to add them later:

- **YouTube end-screen overlays**: 5–20 second end cards pointing
  at "next video in series" + "subscribe". Requires re-encoding
  each MP4 with an extra 20s of silence at the end. Worth doing if
  retention metrics show drop-off at the end of each video.
- **Closed captions / subtitles**: YouTube auto-generates these
  from the audio, but auto-generation will mispronounce
  `BrewSLM`, `GGUF`, `LoRA`, `Ollama`. You can upload corrected
  `.srt` files generated from the narration scripts in
  `docs-demo/scripts/narration/*.md`. ~30 min of work per video.
- **Short-form clips (YouTube Shorts)**: the kickoff sections of
  V07 (training) and V09 (compression) are <30s and could be
  cropped 9:16 as standalone Shorts. Different aesthetic; would
  need a separate thumbnail template.
- **Cards (mid-video)**: link to specific timestamps in other
  videos. Useful when the series is fully published; awkward
  before then.
