"""Knowledge-base loader: one chunk per `## [SECTION-ID] Title` section.

Every chunk keeps its section ID plus the document header metadata
(doc_type, status, effective_from ...), which retrieval and the agent use to
apply the precedence rules in KB-00.1.
"""
from __future__ import annotations
import glob
import os
import re
from dataclasses import dataclass

_HEADER = re.compile(r"^---\n(.*?)\n---\n", re.S)
_SECTION = re.compile(r"^## \[([^\]]+)\] (.+)$", re.M)

# KB-00.1: policy > agent guide > help > faq > marketing. Superseded docs are
# reference-only.
PRECEDENCE = {"policy": 1.0, "meta": 0.95, "agent_guide": 0.9, "help": 0.8,
              "faq": 0.6, "marketing": 0.4}


@dataclass(frozen=True)
class Chunk:
    section_id: str
    title: str
    text: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str
    language: str
    file: str

    @property
    def superseded(self) -> bool:
        return self.status != "current"

    @property
    def weight(self) -> float:
        w = PRECEDENCE.get(self.doc_type, 0.5)
        return w * (0.35 if self.superseded else 1.0)


def load_kb(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        raw = open(path, encoding="utf-8").read()
        m = _HEADER.match(raw)
        meta: dict[str, str] = {}
        if m:
            for line in m.group(1).splitlines():
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
        body = raw[m.end():] if m else raw
        secs = list(_SECTION.finditer(body))
        for i, s in enumerate(secs):
            end = secs[i + 1].start() if i + 1 < len(secs) else len(body)
            chunks.append(Chunk(
                section_id=s.group(1), title=s.group(2).strip(),
                text=body[s.end():end].strip(), doc_id=meta.get("doc_id", ""),
                doc_type=meta.get("doc_type", ""), status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", ""),
                language=meta.get("language", "en"), file=os.path.basename(path)))
    return chunks
