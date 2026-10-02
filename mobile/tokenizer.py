"""
CLIP BPE tokenizer (context 77) for MobileCLIP2 text prompts — no torch / open_clip.

Port of OpenAI CLIP's simple_tokenizer (MIT licence), as used by open_clip for
MobileCLIP2. Only stdlib + numpy (+ ``regex`` for Unicode classes if present).
Mobile apps port this file directly; the vocabulary is
``mobile/assets/bpe_simple_vocab_16e6.txt.gz``.
"""

import gzip
import html
import os
from functools import lru_cache

import numpy as np

try:
    import regex as re_u
    _PATTERN = re_u.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re_u.IGNORECASE)
except ImportError:  # stdlib fallback: same classes for ASCII/Latin text
    import re as re_u
    _PATTERN = re_u.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+""",
        re_u.IGNORECASE)

import re

DEFAULT_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    return {(a, b) for a, b in zip(word[:-1], word[1:])}


def _clean(text: str) -> str:
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip().lower()


class CLIPTokenizer:
    def __init__(self, vocab_path: str = DEFAULT_VOCAB, context_length: int = 77):
        self.context_length = context_length
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(vocab_path).read().decode("utf-8").split("\n")
        merges = [tuple(m.split()) for m in merges[1:49152 - 256 - 2 + 1]]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        vocab.extend("".join(m) for m in merges)
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str):
        ids = []
        for token in re_u.findall(_PATTERN, _clean(text)):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(token).split(" "))
        return ids

    def __call__(self, text: str) -> np.ndarray:
        """Return int64 token ids [1, context_length] (SOT ... EOT, zero padded)."""
        ids = [self.sot] + self.encode(text) + [self.eot]
        if len(ids) > self.context_length:
            ids = ids[:self.context_length]
            ids[-1] = self.eot
        out = np.zeros((1, self.context_length), dtype=np.int64)
        out[0, :len(ids)] = ids
        return out
