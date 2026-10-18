"""Shared narration synthesis for the per-video generator scripts.

Calls ``tts/orpheus_direct.synthesize``'s building blocks directly (Ollama →
SNAC) instead of the Orpheus-FastAPI bridge on :5005, which is not kept
running. The retrying per-sentence ``synthesize_checked`` is the approach
from ``docs-demo/videos/20-legal-rag-vs-sft/build.py``: Orpheus occasionally
returns an empty or truncated decode for one sentence (and loops on long
ones); every chunk is re-asked until its audio is plausible for its length,
very short sentences are folded into a neighbour, and a sentence that stays
silent on its own is voiced joined to the previous one.

``check_sections`` is the listen-check stand-in the lane brief asks for:
every WAV must be > 1 s and < 60 s, the total in the 2–4 minute band, and a
section whose duration is implausible for its word count (~2.6 words/s is
normal) is regenerated once before failing.
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tts import orpheus_direct as od  # noqa: E402

VOICE = "leo"  # one narrator across the docs-demo series
AUDIO_DIR = Path(__file__).parent / "audio"

# Orpheus returns an empty decode for very short sentences no matter how
# often it is asked; fold those into the neighbouring sentence.
MIN_CHUNK_CHARS = 30
# Orpheus truncates at its token cap on long chunks; keep them clause-sized.
MAX_CHUNK_CHARS = 170
# Anything shorter than this for its text is a dropped / truncated take.
MIN_SECONDS_PER_CHAR = 0.035

# Plausibility band for a whole section, in seconds per word. Normal speech
# from this voice is ~2.6 words/s (0.38 s/word); the band is wide enough for
# pauses at sentence ends and narrow enough to catch a truncated or looping
# section.
MIN_SECONDS_PER_WORD = 0.25
MAX_SECONDS_PER_WORD = 0.62
SECTION_MIN_SECONDS = 1.0
SECTION_MAX_SECONDS = 60.0
TOTAL_MIN_SECONDS = 120.0
TOTAL_MAX_SECONDS = 240.0


def _merge_short(chunks: list[str]) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and (len(chunk) < MIN_CHUNK_CHARS or len(merged[-1]) < MIN_CHUNK_CHARS):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def _seconds(audio: np.ndarray) -> float:
    return audio.size / 24000.0


def synthesize_checked(text: str, voice: str, out_path: Path, gap_ms: int = 130) -> float:
    """Synthesize ``text`` sentence by sentence into a 24 kHz mono WAV,
    retrying any chunk whose audio is implausibly short. Returns seconds."""
    sentences = _merge_short(od._split_sentences(text, max_chars=MAX_CHUNK_CHARS))
    gap = np.zeros(int(24000 * gap_ms / 1000), dtype=np.int16)
    pieces: list[np.ndarray] = []
    prev_text = ""
    carry = ""
    for i, sentence in enumerate(sentences):
        if carry:
            sentence = f"{carry} {sentence}"
            carry = ""
        floor = len(sentence) * MIN_SECONDS_PER_CHAR
        audio = od._synthesize_chunk(sentence, voice)
        attempt = 1
        while _seconds(audio) < floor and attempt < 4:
            print(f"    [tts] retry {attempt} ({_seconds(audio):.1f}s < {floor:.1f}s): {sentence[:60]}…", flush=True)
            again = od._synthesize_chunk(sentence, voice)
            if again.size > audio.size:
                audio = again  # keep the longest take
            attempt += 1
        if _seconds(audio) < 0.8 * floor:
            if pieces and prev_text:
                # Some sentences come back silent in isolation; voiced joined
                # to the previous one they go through. Replace the previous take.
                joined = f"{prev_text} {sentence}"
                print(f"    [tts] joining with previous: {sentence[:60]}…", flush=True)
                audio = od._synthesize_chunk(joined, voice)
                if _seconds(audio) < 0.8 * len(joined) * MIN_SECONDS_PER_CHAR:
                    audio = od._synthesize_chunk(joined, voice)
                if _seconds(audio) >= 0.8 * len(joined) * MIN_SECONDS_PER_CHAR:
                    while pieces and pieces[-1] is gap:
                        pieces.pop()
                    pieces.pop()  # the previous sentence's take
                    sentence = joined
            elif i < len(sentences) - 1:
                # Nothing before it: voice it together with the next one.
                print(f"    [tts] carrying forward: {sentence[:60]}…", flush=True)
                carry = sentence
                continue
            if _seconds(audio) < 0.8 * len(sentence) * MIN_SECONDS_PER_CHAR:
                # Last resort: some punctuation patterns silence the model.
                plain = sentence.replace(":", ",").replace("?", ".").replace(";", ",").replace("—", ",")
                print(f"    [tts] plain punctuation: {plain[:60]}…", flush=True)
                audio = od._synthesize_chunk(plain, voice)
            if _seconds(audio) < 0.8 * len(sentence) * MIN_SECONDS_PER_CHAR:
                raise RuntimeError(f"TTS kept dropping a sentence: {sentence[:80]}")
        pieces.append(audio)
        prev_text = sentence
        if i < len(sentences) - 1:
            pieces.append(gap)
    full = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(full.tobytes())
    return _seconds(full)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def word_count(text: str) -> int:
    return len([w for w in text.replace("—", " ").split() if any(c.isalnum() for c in w)])


def plausible(text: str, seconds: float) -> bool:
    words = max(1, word_count(text))
    per_word = seconds / words
    return (
        SECTION_MIN_SECONDS < seconds < SECTION_MAX_SECONDS
        and MIN_SECONDS_PER_WORD <= per_word <= MAX_SECONDS_PER_WORD
    )


def generate_video(
    video: str,
    sections: list[tuple[str, str, str]],
    voice: str = VOICE,
    only: set[str] | None = None,
) -> dict[str, float]:
    """Synthesize every section of ``video`` (``v02`` …) into
    ``tts/audio/<video>-section-<slug>.wav``, write ``<video>-durations.json``
    (keyed by the section key the Playwright spec reads) and
    ``concat-<video>.txt`` (the ffmpeg concat list the mux step reads).

    ``only`` limits synthesis to those slugs; the other sections' durations
    are read back from the existing WAVs so the JSON stays complete.
    A section whose duration is implausible for its word count is regenerated
    once; if it is still implausible the run fails and names it."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    total_wall = 0.0
    for slug, key, text in sections:
        out = AUDIO_DIR / f"{video}-section-{slug}.wav"
        if only is not None and slug not in only and out.exists():
            durations[key] = wav_duration_seconds(out)
            print(f"[{slug}] kept {durations[key]:.2f}s")
            continue
        words = word_count(text)
        dur = 0.0
        for attempt in (1, 2):
            print(f"[{slug}] {words} words, attempt {attempt} …", flush=True)
            t0 = time.time()
            dur = synthesize_checked(text, voice, out)
            total_wall += time.time() - t0
            print(f"[{slug}] {dur:.2f}s audio ({dur / max(1, words):.2f} s/word)", flush=True)
            if plausible(text, dur):
                break
            print(f"[{slug}] implausible for {words} words — regenerating", flush=True)
        if not plausible(text, dur):
            raise RuntimeError(
                f"{video} section {slug}: {dur:.1f}s for {words} words is outside the "
                f"{MIN_SECONDS_PER_WORD}–{MAX_SECONDS_PER_WORD} s/word band after two takes"
            )
        durations[key] = dur

    total = sum(v for k, v in durations.items() if not k.startswith("__"))
    durations["__total_audio__"] = total
    durations["__total_inference_wall__"] = total_wall
    (AUDIO_DIR / f"{video}-durations.json").write_text(json.dumps(durations, indent=2))
    (AUDIO_DIR / f"concat-{video}.txt").write_text(
        "".join(f"file '{(AUDIO_DIR / f'{video}-section-{slug}.wav').resolve()}'\n" for slug, _, _ in sections)
    )
    print()
    print(f"Total audio: {total:.1f}s ({total / 60:.2f} min)")
    print(f"Total inference: {total_wall:.1f}s")
    if not TOTAL_MIN_SECONDS <= total <= TOTAL_MAX_SECONDS:
        raise RuntimeError(f"{video}: total {total:.1f}s is outside the 2–4 minute band")
    return durations


def parse_only(argv: list[str]) -> set[str] | None:
    """``--only slug [slug …]`` regenerates just those sections."""
    if "--only" in argv:
        idx = argv.index("--only")
        return set(argv[idx + 1 :])
    return None
