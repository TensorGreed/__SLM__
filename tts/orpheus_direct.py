"""Direct Orpheus → WAV pipeline that sidesteps the FastAPI bridge.

The bridge in ``tts/Orpheus-FastAPI/`` has a stream-parser bug that
makes the SNAC decoder reject all tokens when the upstream is Ollama
(the splitter doubles each token with a trailing ``'>'``, which throws
off the 7-position frame indexing).

This script calls Ollama once (no streaming), parses every
``<custom_token_NNNN>`` linearly, converts each to a SNAC code id with
the frame-position offset, packs them into 7-tuples, and decodes via
the same SNAC checkpoint the bridge uses.

Usage:
  python tts/orpheus_direct.py \\
      --text "Five second test." \\
      --voice tara \\
      --out /tmp/test.wav
"""

from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

import numpy as np
import requests
import torch
from snac import SNAC


_OLLAMA_URL = "http://127.0.0.1:11434/v1/completions"
_MODEL = "legraphista/Orpheus:3b-ft-q4_k_m"

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

# Default voice. The Orpheus-FT checkpoint ships eight English voices;
# "leo" is the male voice used across the existing docs-demo set.
DEFAULT_VOICE = "leo"


def _request_orpheus(
    text: str, voice: str, max_tokens: int = 4096, temperature: float = 0.6
) -> str:
    """Single non-streamed call to Ollama. Returns the raw completion.

    Called per-sentence (see ``_split_sentences``), so the 4096-token
    cap is comfortably above a single sentence's audio length (~10s ≈
    1800 tokens) while staying far from the runaway/loop regime that a
    whole-paragraph prompt invites."""
    prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
    resp = requests.post(
        _OLLAMA_URL,
        json={
            "model": _MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def _tokens_to_codes(raw_text: str) -> list[int]:
    """Convert ``<custom_token_NNNN>`` sequence into SNAC code ids.

    Orpheus emits a small structural prefix (tokens 1, 4, 5) followed
    by 7-token audio frames; each in-frame position carries an offset
    of ``pos * 4096 + 10``. We track an audio-only index so the prefix
    tokens don't throw off the frame alignment.
    """
    codes: list[int] = []
    audio_idx = 0
    for m in _TOKEN_RE.finditer(raw_text):
        n = int(m.group(1))
        # Skip the structural boundary tokens (1 / 4 / 5).
        if n < 10:
            continue
        position = audio_idx % 7
        code = n - 10 - position * 4096
        if 0 <= code < 4096:
            codes.append(code)
        else:
            codes.append(-1)
        audio_idx += 1
    return codes


# Force CPU — the SNAC decoder uses snake activations whose nvrtc
# kernel doesn't compile against this box's CUDA sm_121 GPU
# (PyTorch only supports up to sm_120). CPU decode of 30s of audio
# takes ~3-5s on this machine, fast enough for narration work.
_SNAC_DEVICE = "cpu"
_SNAC_MODEL: SNAC | None = None


def _snac() -> SNAC:
    """Lazily load + cache the SNAC checkpoint. Per-sentence synthesis
    decodes many times, so re-instantiating on every call would dominate
    wall time."""
    global _SNAC_MODEL
    if _SNAC_MODEL is None:
        _SNAC_MODEL = (
            SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_SNAC_DEVICE)
        )
    return _SNAC_MODEL


def _decode_to_audio(codes: list[int]) -> np.ndarray:
    """Run SNAC decode over the 7-token-frame stream. Returns int16
    PCM samples at 24 kHz mono."""
    # Drop the initial structural marker run (snac tokens always come
    # in frame-aligned 7-tuples, but the model emits a small header).
    # Find the first valid frame start by skipping leading sentinels.
    while codes and codes[0] < 0:
        codes.pop(0)
    n_frames = len(codes) // 7
    if n_frames == 0:
        return np.zeros(0, dtype=np.int16)

    snac_device = _SNAC_DEVICE
    model = _snac()

    audio_chunks: list[np.ndarray] = []
    # Chunk frames so a long generation doesn't blow VRAM.
    chunk = 64
    for start in range(0, n_frames, chunk):
        sub = codes[start * 7:(start + chunk) * 7]
        sub_frames = len(sub) // 7
        if sub_frames == 0:
            continue
        # Reject windows that contain any rejection sentinel.
        if any(c < 0 for c in sub):
            continue

        codes_0 = torch.zeros(sub_frames, dtype=torch.int32, device=snac_device)
        codes_1 = torch.zeros(sub_frames * 2, dtype=torch.int32, device=snac_device)
        codes_2 = torch.zeros(sub_frames * 4, dtype=torch.int32, device=snac_device)
        for j in range(sub_frames):
            i = j * 7
            codes_0[j] = sub[i]
            codes_1[j * 2] = sub[i + 1]
            codes_1[j * 2 + 1] = sub[i + 4]
            codes_2[j * 4] = sub[i + 2]
            codes_2[j * 4 + 1] = sub[i + 3]
            codes_2[j * 4 + 2] = sub[i + 5]
            codes_2[j * 4 + 3] = sub[i + 6]
        with torch.inference_mode():
            audio_hat = model.decode([
                codes_0.unsqueeze(0),
                codes_1.unsqueeze(0),
                codes_2.unsqueeze(0),
            ])
        # SNAC outputs (batch, channels, samples); take the central
        # slice the bridge uses to avoid edge artifacts.
        audio = audio_hat[:, :, 2048:-2048].squeeze().cpu().numpy()
        if audio.ndim == 0 or audio.size == 0:
            audio = audio_hat.squeeze().cpu().numpy()
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        audio_chunks.append(audio_int16)

    if not audio_chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(audio_chunks)


# Split on sentence-ending punctuation followed by whitespace and a
# capital letter or opening quote. The capital-letter lookahead keeps
# decimals like "Qwen 2.5" intact (the '.' is followed by a digit).
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=["A-Z])')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;—])\s+')


def _split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Break narration into sentence-sized chunks. Over-long sentences
    fall back to clause splitting on commas / dashes so no single chunk
    risks the truncation or runaway regime of a long generation."""
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
    """One sentence → int16 PCM. Retries once on an empty decode or a
    runaway generation (audio implausibly long for the text length, the
    tell-tale of the model looping)."""
    def _gen(temperature: float) -> np.ndarray:
        raw = _request_orpheus(text, voice, temperature=temperature)
        return _decode_to_audio(_tokens_to_codes(raw))

    audio = _gen(0.6)
    runaway_cap = len(text) * 0.11 + 3.0  # seconds
    too_long = audio.size / 24000.0 > runaway_cap
    if audio.size == 0 or too_long:
        # Lower temperature on retry: steadier, less prone to looping.
        retry = _gen(0.3)
        if retry.size and (
            audio.size == 0 or retry.size / 24000.0 <= runaway_cap
        ):
            audio = retry
    return audio


def synthesize(text: str, voice: str, out_path: Path, gap_ms: int = 130) -> float:
    """Synthesize ``text`` with ``voice`` and write a 24kHz mono WAV.

    Narration is synthesized one sentence at a time and concatenated
    with a short inter-sentence gap. Whole-paragraph generations on this
    Orpheus checkpoint either truncate at the token cap or loop; chunking
    keeps every generation short enough to be reliable while preserving a
    single consistent speaker voice. Returns the duration in seconds."""
    sentences = _split_sentences(text)
    gap = np.zeros(int(24000 * gap_ms / 1000), dtype=np.int16)
    pieces: list[np.ndarray] = []
    for i, sentence in enumerate(sentences):
        audio = _synthesize_chunk(sentence, voice)
        pieces.append(audio)
        if i < len(sentences) - 1:
            pieces.append(gap)
    full = (
        np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(full.tobytes())
    return full.size / 24000.0


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dur = synthesize(args.text, args.voice, args.out)
    print(f"wrote {args.out} ({dur:.2f}s)")


if __name__ == "__main__":
    _cli()
