"""Tiny, download-free fast tokenizer for trainer masking tests.

One token per character (plus whole special tokens), built in-process with
``tokenizers`` so offsets are exact and no model is fetched. ``CHATML``
mirrors the Qwen2.5-style template that injects ``<|im_end|>\\n`` after
every message and has no ``{% generation %}`` markers.
"""

from __future__ import annotations

import string

SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]

CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# A template that inserts a fixed default system message when none is given
# (rendered prefixes still line up with the full render).
CHATML_DEFAULT_SYSTEM = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are helpful.<|im_end|>\n' }}"
    "{% endif %}" + CHATML
)


def build_tiny_tokenizer(chat_template: str | None = CHATML):
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import PreTrainedTokenizerFast

    chars = sorted(set(string.printable))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
    vocab["<unk>"] = len(vocab)
    core = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        pad_token="<|pad|>",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = chat_template
    return tok
