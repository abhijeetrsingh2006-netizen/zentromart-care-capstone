from __future__ import annotations

import asyncio
import csv
import datetime as dt
import io
import importlib.util
import json
import os
import re
import secrets
import threading
import time
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
KB = ROOT / "kb"
PRACTICE = ROOT / "practice"
REFERENCE_DATE = dt.date(2026, 10, 20)
PORT = int(os.environ.get("PORT", "8000"))
GNANI_KEY = os.environ.get("GNANI_API_KEY", "")
SESSION_TTL_SECONDS = 1800
_SESSIONS: dict[str, dict] = {}
_SESSION_LOCK = threading.Lock()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_digits(s: str) -> str:
    return s.translate(str.maketrans("०१२३४५६७८९", "0123456789"))


STOP_WORDS = {"the", "a", "an", "is", "are", "am", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "will", "would", "should", "i", "me", "my", "we", "our", "you", "your", "they", "it", "this", "that", "what", "when", "where", "how", "who", "which", "to", "of", "for", "in", "on", "at", "by", "with", "from", "and", "or", "but", "if", "as", "about", "after", "before", "have", "has", "had", "do", "does", "did", "now", "still", "once", "please", "मेरा", "मेरी", "मेरे", "क्या", "है", "हैं", "को", "में", "से", "कितने", "दिन", "तक", "कर", "सकती", "हूँ", "है", "अब", "मैंने", "मुझे", "मेरा"}
QUERY_EXPANSIONS = {
    "come": "reach arrive received timeline", "money": "refund", "back": "refund return",
    "cash": "cod payment cod_bank cod_wallet", "delivery": "cod express shipped",
    "collection": "reverse pickup pickup", "collect": "reverse pickup", "torn": "damaged defective",
    "opened": "open unopened seal condition", "open": "opened unopened seal", "tried": "used condition",
    "free": "waived fee", "refund": "refund timeline initiated reach",
    "cash": "cash-on-delivery cod payment cod_bank cod_wallet",
}


def tokenize(s: str) -> set[str]:
    words = {x.lower() for x in re.findall(r"[\w-]+", s, flags=re.UNICODE) if len(x) > 1}
    words |= {part for word in tuple(words) if "-" in word for part in word.split("-") if len(part) > 1}
    words -= STOP_WORDS
    return words


def query_tokens(s: str) -> set[str]:
    words = tokenize(s)
    expanded = set(words)
    for word in words:
        expanded.update(QUERY_EXPANSIONS.get(word, "").split())
    
    low = s.lower()
    if any(x in low for x in ("कितने दिन", "कितने दिनों", "kitne din", "kitne dino")):
        expanded.update({"return", "window", "days", "eligibility"})
    if any(x in low for x in ("वापस", "लौटा", "return kar", "wapas")):
        expanded.update({"return", "window", "eligibility", "policy"})
    if any(x in low for x in ("खोल", "खुला", "opened")):
        expanded.update({"opened", "seal", "condition"})
    if "paise" in low or "paisa" in low:
        expanded.update({"refund", "money", "reach"})
    return expanded


def load_sections() -> list[dict]:
    sections = []
    for path in sorted(KB.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        if raw.startswith("---"):
            head = raw.split("---", 2)[1]
            for line in head.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip('"')
        current_id, current_title, body = "", "", []
        for line in raw.splitlines():
            match = re.match(r"##\s+\[([^]]+)\]\s*(.*)", line)
            if match:
                if current_id:
                    sections.append({"id": current_id, "title": current_title,
                                     "text": "\n".join(body).strip(), "file": path.name,
                                     "meta": meta})
                current_id, current_title, body = match.group(1), match.group(2), []
            elif current_id:
                body.append(line)
        if current_id:
            sections.append({"id": current_id, "title": current_title,
                             "text": "\n".join(body).strip(), "file": path.name,
                             "meta": meta})
    return sections


SECTIONS = load_sections()
PRODUCTS = {p["sku"]: p for p in read_csv(DATA / "products.csv")}
QUESTIONS = read_csv(PRACTICE / "questions.csv")
ORDERS = {o["order_id"]: o for o in read_csv(DATA / "orders.csv")}
CUSTOMERS = {c["customer_id"]: c for c in read_csv(DATA / "customers.csv")}
ITEMS: dict[str, list[dict[str, str]]] = {}
for item in read_csv(DATA / "order_items.csv"):
    ITEMS.setdefault(item["order_id"], []).append(item)
HOLIDAYS = {row["date"] for row in read_csv(DATA / "holidays.csv")}


def section_score(query: str, section: dict) -> float:
    q = query_tokens(query)
    body = tokenize(section["title"] + " " + section["text"])
    if not q or not body:
        return 0
    overlap = len(q & body) / max(1, len(q))
    
    precision = len(q & body) / max(1, len(body)) ** 0.18
    overlap = 0.78 * overlap + 0.22 * precision
    meta = section["meta"]
    weights = {"policy": 1.5, "agent_guide": 1.3, "help": 1.15, "faq": 1.0, "marketing": 0.4, "meta": 1.0}
    boost = weights.get(meta.get("doc_type", ""), 1.0)
    if meta.get("status") == "superseded":
        boost = 0.12
    if meta.get("doc_type") == "marketing":
        boost *= 0.45
    if section["id"].startswith("KB-00.1"):
        boost = 1.0
    return overlap * boost


def retrieve(query: str, limit: int = 4) -> list[dict]:
    expanded_query = query
    low = query.lower()
    
    for product in PRODUCTS.values():
        name_tokens = tokenize(product["title"])
        title_overlap_needed = max(2, (len(name_tokens) * 3 + 4) // 5)
        if name_tokens and len(name_tokens & query_tokens(query)) >= title_overlap_needed:
            category_terms = {"beauty_sealed": "beauty unopened opened seal intact packaging condition",
                              "electronics": "electronics accessories return window damaged condition",
                              "fashion": "clothing footwear bags return window tags",
                              "home_kitchen": "home kitchen return window condition",
                              "books_media": "books stationery return window condition",
                              "grocery": "grocery fresh perishable not returnable damaged",
                              "innerwear": "innerwear lingerie swimwear not returnable",
                              "digital": "gift card digital code not returnable"}
            expanded_query += f" {product['return_category']} product category {category_terms.get(product['return_category'], '')}"
            break
    scored = [(section_score(expanded_query, s), s) for s in SECTIONS]
    scored.sort(key=lambda x: x[0], reverse=True)
    found = []
    for score, s in scored:
        if score <= 0.01:
            continue
        found.append({"id": s["id"], "title": s["title"], "file": s["file"],
                      "score": round(score, 3), "text": s["text"][:1100],
                      "status": s["meta"].get("status", "current"),
                      "doc_type": s["meta"].get("doc_type", ""),
                      "effective_from": s["meta"].get("effective_from", "")})
        if len(found) >= limit:
            break
    return found


def sections_by_ids(ids: list[str]) -> list[dict]:
    return [{"id": s["id"], "title": s["title"], "file": s["file"], "text": s["text"][:1100],
             "status": s["meta"].get("status", "current"), "doc_type": s["meta"].get("doc_type", ""),
             "effective_from": s["meta"].get("effective_from", "")}
            for wanted in ids for s in SECTIONS if s["id"] == wanted]


def make_ticket(summary: str, order_id: str = "", reason: str = "support_handover", intent: str = "support") -> dict:
    
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.tools import create_ticket
    return create_ticket(summary=summary, order_id=order_id or None, reason=reason, intent=intent)


def lookup(order_id: str, proof: str | None = None) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.tools import lookup_order
    return lookup_order(order_id, verification_value=proof)


# --------------------------------------------------------------------------
# Conversation engine: rule-based agent in ./support_agent (stdlib only).
# Exact facts (deadlines, fees, refund dates, escalation triggers) come from
# deterministic rules over the CSVs + current policy; retrieval supplies the
# cited sections and answers the long tail.  No practice-answer lookup.
# --------------------------------------------------------------------------
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from support_agent.agent import SupportAgent  # noqa: E402


class _ToolAdapter:
    """Routes the agent's tool calls through this module's lookup/make_ticket so
    tests can monkeypatch make_ticket and the offline mock tools stay the single
    source of truth."""

    def lookup_order(self, order_id, verification_value=None):
        return lookup(order_id, verification_value)

    def create_ticket(self, summary, order_id=None, reason="support_handover", intent="support"):
        return make_ticket(summary, order_id or "", reason, intent)


AGENT = SupportAgent(tools=_ToolAdapter())


def _evidence_sources(ids: list[str]) -> list[dict]:
    out = sections_by_ids([i for i in ids if not i.startswith("products.csv:")])
    for ev in ids:
        if ev.startswith("products.csv:") and ev.split(":", 1)[1] in PRODUCTS:
            p = PRODUCTS[ev.split(":", 1)[1]]
            out.append({"id": ev, "title": p["title"], "file": "products.csv", "text": p["description"], "status": "current"})
    return out


def response_for(query: str, state: dict, voice: bool = False) -> tuple[str, list[dict], dict]:
    """One conversation turn. `state` is the per-visitor session dict; `voice=True`
    means the text came from speech recognition (spoken digits are repaired)."""
    session = state.setdefault("_agent_session", AGENT.new_session())
    reply = AGENT.respond(session, query, channel="voice" if voice else "chat", asr=voice)
    tool = ("create_ticket" if any(c.startswith("create_ticket") for c in reply.tool_calls)
            else "lookup_order" if reply.tool_calls else "")
    meta = {"decision": reply.decision, "intent": reply.intent, "language": reply.language,
            "tool": tool, "tool_calls": reply.tool_calls,
            "verified": bool(session.order), "ticket_id": reply.ticket_id}
    state["verified_order"] = session.order_id if session.order else None
    return reply.text, _evidence_sources(reply.evidence), meta


def wav_to_pcm(audio: bytes) -> bytes:
    """Return raw PCM from a RIFF/WAV upload.  Walks the chunks instead of assuming a
    44-byte header, and rejects anything that is not 16 kHz mono 16-bit PCM (what the
    Gnani realtime STT frame spec expects)."""
    import struct
    if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("Voice input must be WAV audio.")
    pos, fmt = 12, None
    while pos + 8 <= len(audio):
        cid, size = audio[pos:pos + 4], struct.unpack("<I", audio[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", audio[body:body + 16])
        elif cid == b"data":
            if not fmt or fmt[0] != 1 or fmt[1] != 1 or fmt[2] != 16000 or fmt[5] != 16:
                raise ValueError("Voice input must be mono 16 kHz 16-bit PCM WAV.")
            return audio[body:body + size]
        pos = body + size + (size & 1)
    raise ValueError("WAV file has no audio data.")


STT_LANGS = {"en-IN": "en-IN", "hi-IN": "hi-IN", "hi-en": "hi-IN"}      # Hinglish is recognised as Hindi
TTS_VOICES = {"en": ("en-IN", "Kaveri"), "hi": ("hi-IN", "Nalini"), "hi-en": ("hi-IN", "Poorvi")}


def gnani_stt(audio: bytes, language: str) -> str:
    if not GNANI_KEY:
        raise RuntimeError("Set GNANI_API_KEY to enable Gnani speech recognition.")
    pcm = wav_to_pcm(audio)
    keywords = [k for k in os.environ.get("GNANI_STT_KEYWORDS", "").split(",") if k.strip()] or None

    async def transcribe():
        from gnani.stt import GnaniSTTStreamClient
        async with GnaniSTTStreamClient(api_key=GNANI_KEY, language_code=language, sample_rate=16000,
                                        keywords=keywords) as stream:
            events = await stream.stream_audio(io.BytesIO(pcm), realtime_pace=True)
        return " ".join(event.text.strip() for event in events if getattr(event, "text", "").strip())
    return asyncio.run(transcribe())


def gnani_tts(text: str, language: str = "auto", voice: str = "auto") -> bytes:
    """Speak `text` in the language it is written in.  `language` may be en / hi / hi-en
    (or en-IN / hi-IN); 'auto' detects it from the text itself."""
    if not GNANI_KEY:
        raise RuntimeError("Set GNANI_API_KEY to enable Gnani text-to-speech.")
    from support_agent import nlu
    from support_agent.speech_text import to_speakable
    key = {"en-IN": "en", "hi-IN": "hi"}.get(language, language)
    if key not in TTS_VOICES:
        key = nlu.detect_language(text)
        key = key if key in TTS_VOICES else "en"
    code, default_voice = TTS_VOICES[key]
    selected = voice if voice in ("Kaveri", "Nalini", "Poorvi") else default_voice
    spoken = to_speakable(text, "hi" if key == "hi" else "en")       # dates, IDs, rupees as words

    async def synthesize():
        from gnani.tts import GnaniTTSRealtimeClient
        async with GnaniTTSRealtimeClient(api_key=GNANI_KEY) as client:
            return await client.synthesize_and_collect(spoken, voice=selected, model="timbre-v2.5", language=code)
    return asyncio.run(synthesize())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, data: dict, status: int = 200, cookie: str | None = None):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie:
            self.send_header("Set-Cookie", f"zm_session={cookie}; HttpOnly; SameSite=Lax; Path=/")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            sdk_available = importlib.util.find_spec("gnani") is not None
            return self.send_json({"ok": True, "gnani_enabled": bool(GNANI_KEY and sdk_available),
                                   "api_key_configured": bool(GNANI_KEY), "sdk_available": sdk_available,
                                   "reference_date": "2026-10-20"})
        if path == "/api/search":
            from urllib.parse import parse_qs
            query = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            return self.send_json({"results": retrieve(query, 8)})
        asset = ROOT / ("index.html" if path in ("/", "") else path.lstrip("/"))
        if not asset.is_file() or ROOT not in asset.resolve().parents and asset.resolve() != ROOT:
            self.send_error(404)
            return
        content = asset.read_bytes()
        mime = "text/html; charset=utf-8" if asset.suffix == ".html" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        path = urlparse(self.path).path
        length = min(int(self.headers.get("Content-Length", "0")), 8_000_000)
        body = self.rfile.read(length)
        if path == "/api/reset":
            sid = None
            for part in self.headers.get("Cookie", "").split(";"):
                if part.strip().startswith("zm_session="):
                    sid = part.strip().split("=", 1)[1]
            with _SESSION_LOCK:
                if sid in _SESSIONS:
                    _SESSIONS.pop(sid, None)
            return self.send_json({"ok": True})
        if path == "/api/chat":
            try:
                payload = json.loads(body.decode("utf-8"))
                message = str(payload.get("message", ""))[:4000]
            except Exception:
                return self.send_json({"error": "Invalid request."}, 400)
            if not message.strip():
                return self.send_json({"error": "Enter a message first."}, 400)
            sid = None
            for part in self.headers.get("Cookie", "").split(";"):
                if part.strip().startswith("zm_session="):
                    sid = part.strip().split("=", 1)[1]
            fresh = not sid or not re.fullmatch(r"[a-f0-9]{32}", sid)
            if fresh:
                sid = secrets.token_hex(16)
            with _SESSION_LOCK:
                state = _SESSIONS.get(sid)
                if not state or time.monotonic() - state.get("_last_activity", 0) > SESSION_TTL_SECONDS:
                    state = {}
                    _SESSIONS[sid] = state
                    fresh = True
                state["_last_activity"] = time.monotonic()
                answer, sources, meta = response_for(message, state, voice=bool(payload.get("voice")))
            return self.send_json({"answer": answer, "sources": sources, "meta": meta,
                                   "reference_date": "20 October 2026"}, cookie=sid if fresh else None)
        if path == "/api/stt":
            try:
                language = self.headers.get("X-Language-Code", "en-IN")
                if language not in STT_LANGS:
                    return self.send_json({"error": "Choose English, Hindi, or Hinglish for voice input."}, 400)
                transcript = gnani_stt(body, STT_LANGS[language])
                return self.send_json({"transcript": transcript})
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 503)
        if path == "/api/tts":
            try:
                payload = json.loads(body.decode("utf-8"))
                audio = gnani_tts(str(payload.get("text", ""))[:1000], str(payload.get("language", "auto")),
                                  str(payload.get("voice", "Nalini")))
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(audio)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 503)
            return
        self.send_error(404)


if __name__ == "__main__":
    print(f"ZentroMart assistant running at http://127.0.0.1:{PORT}")
    print(f"Dataset reference date: {REFERENCE_DATE.isoformat()} | Gnani voice: {'enabled' if GNANI_KEY else 'not configured'}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
