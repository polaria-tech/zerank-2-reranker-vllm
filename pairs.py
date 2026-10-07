"""(query, documents) pairs shared by reference.py and parity.py.

Deliberately varied: relevant / off-topic / lexical-trap documents, multilingual, and groups
mixing very different document lengths (a few tokens -> several thousand) to catch
padding and pooling bugs.

The "long" set (8k -> ~39k tokens, close to the 40960-token context) is built from the Python
standard library's pydoc output: natural English, available on any machine, no data files.
"""

import importlib
import pydoc
import re
from pathlib import Path

_HERE = Path(__file__).parent
_LICENSE = (_HERE / "upstream" / "LICENSE").read_text()
_README = (_HERE / "upstream" / "README.md").read_text()

GROUPS: list[tuple[str, list[str]]] = [
    (
        "What is 2+2?",
        ["4", "The answer is definitely 1 million", "2+2 equals four.", "Paris is the capital of France."],
    ),
    (
        "How do I reset a forgotten Linux root password?",
        [
            "Reboot, edit the GRUB entry, append init=/bin/bash to the kernel line, remount / read-write and run passwd.",
            "Use `sudo passwd root` if you still have a sudo-capable account.",
            "Linux is a family of open-source Unix-like operating systems based on the Linux kernel.",
            "To reset a forgotten Windows password, use a password reset disk.",
            "Root vegetables such as carrots and beets store well in a cool cellar.",
        ],
    ),
    (
        "Quelle est la capitale de l'Australie ?",
        [
            "Canberra est la capitale de l'Australie depuis 1913.",
            "Sydney est la plus grande ville d'Australie, mais ce n'est pas la capitale.",
            "The capital of Australia is Canberra.",
            "La capitale de l'Autriche est Vienne.",
        ],
    ),
    (
        "side effects of ibuprofen",
        [
            "Common side effects of ibuprofen include heartburn, nausea, stomach pain and dizziness; long-term use raises the risk of ulcers and kidney problems.",
            "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to treat pain and fever.",
            "Acetaminophen overdose can cause liver damage.",
            "",
        ],
    ),
    (
        "python sort list of dicts by key",
        [
            "sorted(items, key=lambda d: d['age']) returns a new list ordered by the 'age' key; use operator.itemgetter('age') for speed.",
            "list.sort() sorts a list in place.",
            "Dictionaries in Python 3.7+ preserve insertion order.",
            "In JavaScript, use arr.sort((a, b) => a.age - b.age).",
        ],
    ),
    (
        "Under the Apache License, must I state changes when redistributing modified files?",
        [
            # Very different lengths within one group.
            "Yes.",
            "You must cause any modified files to carry prominent notices stating that You changed the files.",
            _LICENSE,
            _LICENSE + "\n\n" + _README + "\n\n" + _LICENSE,
            _README,
            "MIT license: permission is hereby granted, free of charge, to any person obtaining a copy.",
        ],
    ),
    (
        "zerank-2 score calibration",
        [
            _README,
            "The model outputs a raw logit; divide by 5 and apply a sigmoid to get a calibrated 0-1 relevance score.",
            "Calibration of thermometers requires an ice bath at 0 degrees Celsius.",
            "x " * 3000,
        ],
    ),
    (
        # Intermediate lengths (~60 -> ~900 tokens).
        "What patent rights does a contributor grant under the Apache License?",
        [_LICENSE[i : i + n] for i, n in ((0, 300), (4000, 900), (5500, 1600), (2000, 2600), (0, 4000))],
    ),
    (
        "日本の首都はどこですか？",
        ["日本の首都は東京です。", "大阪は日本で二番目に大きな都市圏です。", "The capital of Japan is Tokyo.", "北京是中国的首都。"],
    ),
]


def _doc(module: str) -> str:
    text = pydoc.render_doc(importlib.import_module(module), renderer=pydoc.plaintext)
    # pydoc prints object addresses ("<object at 0x7f...>"): strip them so the text is reproducible.
    return re.sub(r" at 0x[0-9a-f]+", "", text)


CHARS_PER_TOKEN = 4.3  # measured on pydoc text with the zerank-2 tokenizer


def long_text(target_tokens: int, relevant: str | None = None, where: str = "start") -> str:
    """~target_tokens of unrelated stdlib documentation, with `relevant` inserted at `where`."""
    filler = "\n\n".join(_doc(m) for m in (
        "logging", "pathlib", "http.client", "collections", "email.message", "unittest.case"))
    n = int(target_tokens * CHARS_PER_TOKEN) - (len(relevant) if relevant else 0)
    assert n <= len(filler), "not enough filler text"
    filler = filler[:max(n, 0)]
    if relevant is None:
        return filler
    if where == "start":
        return relevant + "\n\n" + filler
    if where == "end":
        return filler + "\n\n" + relevant
    half = len(filler) // 2
    return filler[:half] + "\n\n" + relevant + "\n\n" + filler[half:]


def long_groups() -> list[tuple[str, list[str]]]:
    argparse_doc = _doc("argparse")
    return [(
        "How do I add subcommands, each with their own arguments, to a command-line parser?",
        [
            argparse_doc,                                   # ~6k tokens, relevant
            long_text(16_000, argparse_doc, "start"),
            long_text(24_000, argparse_doc, "end"),
            long_text(32_000, argparse_doc, "middle"),
            long_text(32_000),                              # long and irrelevant
            long_text(39_000, argparse_doc, "end"),         # just under the 40960 context
            "Use parser.add_subparsers() and call add_parser() for each subcommand.",
        ],
    )]


def flat_pairs(which: str = "main") -> list[tuple[str, str]]:
    groups = {"main": GROUPS, "long": None}[which] or long_groups()
    return [(q, d) for q, docs in groups for d in docs]
