"""Kokoro-82M direct TTS adapter.

Drop-in replacement for ``tts.orpheus_direct.synthesize``: same
signature (``text, voice, out_path, gap_ms`` → duration in seconds),
same output format (24kHz mono WAV). The lesson-builder pipeline imports
``synthesize`` + ``_split_sentences`` from here just like it used to
from ``orpheus_direct`` — no callers need to change beyond the import.

Why Kokoro:

- 82M parameters — small enough to run locally on the dev box without
  a separate model-server process, no network dependency, no per-token
  billing. The previous Orpheus path needed the same setup; Kokoro just
  produces noticeably better narration at lower latency.
- Single-narrator long-form is exactly the regime Kokoro is best at.
  We deliberately pass on Dia 2-2B (multi-speaker dialogue model) —
  the dialogue strength is wasted on lesson narration and the 2B model
  is heavier to deploy.

Voices: Kokoro ships with named voice presets like ``af_heart`` (warm
American female default), ``af_sky``, ``am_michael``, ``bf_emma`` etc.
Pass the voice name through ``voice``; we forward it to the pipeline.

Sentence chunking: re-exported as ``_split_sentences`` because the
lesson-build canonical TTS contract uses it. Kokoro tolerates much
longer chunks than Orpheus did (no runaway-token regime), but we keep
the sentence-by-sentence pattern for predictable timing per beat in
the video build.
"""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import numpy as np


# Kokoro's KPipeline lazy-init — building it is heavy (loads the
# phonemizer + model weights) so we cache a single instance for the
# process. Re-using the pipeline across sentences also avoids the
# first-call warmup tax dominating the per-sentence wall-clock.
_PIPELINE = None


# Kokoro's default sample rate. Hard-coded because we match the
# 24kHz output the previous Orpheus pipeline produced — keeps the
# downstream ffmpeg concat / caption-burn steps unchanged.
SAMPLE_RATE = 24000


# Default voice. ``af_heart`` is the official recommended default per
# the Kokoro model card — warm narrator voice, well-suited to the
# Academy lessons. Override via ``voice=`` per call when a video wants
# a different timbre.
DEFAULT_VOICE = "af_heart"


def _get_pipeline():
    """Lazy-init the Kokoro pipeline. Kept module-level so the first
    sentence pays the load cost and every subsequent sentence is
    free of warmup overhead.

    Pinned to CPU: this dev box's GB10 reports sm_121 (Grace Blackwell)
    and PyTorch's bundled nvrtc doesn't yet ship a compiler target for
    that architecture, so any JIT'd CUDA kernel inside the Kokoro
    pipeline fails with ``nvrtc: invalid value for --gpu-architecture``.
    Kokoro is 82M parameters — CPU is fast enough for the batch
    lesson-rendering use case (each sentence renders in well under a
    second on a modern multi-core box).
    """
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "kokoro package is not installed. Run `pip install kokoro "
                "soundfile` (espeak-ng must also be available on PATH for "
                "the g2p step)."
            ) from exc
        import torch
        # ``lang_code='a'`` = American English. The Kokoro model is
        # multilingual but we run American narration; switching is a
        # parameter not a model swap.
        _PIPELINE = KPipeline(lang_code="a", device="cpu")
        # Defense in depth: even if a sub-component tries to move
        # tensors to CUDA, force-disable CUDA in this process so the
        # JIT path can't fire on sm_121.
        if hasattr(torch.cuda, "is_available"):
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    return _PIPELINE


# Same regex contract the orpheus adapter uses — sentence-break on
# punctuation-then-capital, clause-break on commas / semicolons /
# em-dashes for long-sentence fallback. Keeping the patterns identical
# means downstream caption-segment timing doesn't shift with the swap.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=["A-Z])')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;—])\s+')


def _split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Break narration into sentence-sized chunks. Kokoro doesn't have
    the truncation/runaway regime Orpheus did, but per-sentence pieces
    keep caption timing predictable per beat and let a single bad
    pronunciation be regenerated without re-rendering the whole scene."""
    chunks: list[str] = []
    for part in _SENT_SPLIT_RE.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            chunks.append(part)
            continue
        buf = ""
        for clause in _CLAUSE_SPLIT_RE.split(part):
            if len(buf) + len(clause) + 1 <= max_chars:
                buf = f"{buf} {clause}".strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = clause
        if buf:
            chunks.append(buf)
    return chunks


def _synthesize_chunk(text: str, voice: str) -> np.ndarray:
    """One sentence → int16 PCM @ 24kHz mono.

    Kokoro returns float32 audio in [-1, 1]; we convert to int16 to
    match the WAV format the downstream ffmpeg pipeline expects. The
    pipeline yields one generator step per chunk; we drain it and
    concatenate.
    """
    pipeline = _get_pipeline()
    pieces: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        if audio is None:
            continue
        arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.size == 0:
            continue
        pieces.append(arr.astype(np.float32))

    if not pieces:
        return np.zeros(0, dtype=np.int16)

    full = np.concatenate(pieces)
    # Float32 [-1,1] → int16 [-32768, 32767], clamping defensively in
    # case Kokoro ever returns a slight overflow.
    return np.clip(full * 32767.0, -32768, 32767).astype(np.int16)


def synthesize(text: str, voice: str, out_path: Path, gap_ms: int = 130) -> float:
    """Synthesize ``text`` with ``voice`` and write a 24kHz mono WAV.

    Drop-in replacement for ``tts.orpheus_direct.synthesize``. Renders
    one sentence at a time and concatenates with a short inter-sentence
    gap so the video build's beat-by-beat caption timing stays the same
    after the TTS swap. Returns the duration in seconds.
    """
    sentences = _split_sentences(text)
    gap = np.zeros(int(SAMPLE_RATE * gap_ms / 1000), dtype=np.int16)
    pieces: list[np.ndarray] = []
    for i, sentence in enumerate(sentences):
        audio = _synthesize_chunk(sentence, voice or DEFAULT_VOICE)
        pieces.append(audio)
        if i < len(sentences) - 1:
            pieces.append(gap)
    full = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(full.tobytes())
    return full.size / SAMPLE_RATE


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Kokoro TTS — render a WAV.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dur = synthesize(args.text, args.voice, args.out)
    print(f"wrote {args.out} ({dur:.2f}s)")


if __name__ == "__main__":
    _cli()
