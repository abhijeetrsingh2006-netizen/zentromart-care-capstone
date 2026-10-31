"""Hybrid retriever over the KB sections.

* BM25 over word tokens + a char n-gram TF-IDF channel (robust to Hinglish
  spelling variants), fused with reciprocal-rank fusion.
* Query expansion maps customer wording ("send it back", "paisa wapas",
  "nobody came to collect") onto the vocabulary the documents use.
* Precedence re-ranking (policy > guide > help > FAQ > marketing; superseded
  docs demoted unless the question is about the old rules).
* `coverage()` tells the agent whether the documents actually answer the
  question, so unanswerable ones get "I don't have that" instead of a guess.
"""
from __future__ import annotations
import math
import re
from collections import Counter

from .kb import Chunk

_TOKEN = re.compile(r"[\w\u0900-\u097F₹]+", re.U)   # \w alone splits Devanagari at vowel signs
STOP = set("""a an the is are was were be been am do does did can could should would will shall may i me my we our you your it its
this that these those of to in on at by for from with as and or but if so than then there here what which who whom how when where why
long much many get got have has had not no yes also just any some about into out up down over after before again more most very
please tell me know want like need let one kya hai hain ka ki ke ko se me mein pe par aur ya to toh bhi hi nahi kab kaise kitne kitna
kitni mera meri mere mujhe hum aap tum kar karo karna kare sakta sakti sakte hoon hu hota hoti hote tha thi the wala wali wale ne
क्या है हैं का की के को से में पर और या तो भी ही नहीं कब कैसे कितने कितना कितनी मेरा मेरी मेरे मुझे हम आप कर करो करना सकता सकती सकते हूँ हूं था थी थे
""".split())

# (regex over lower-cased query) -> extra English vocabulary appended to the query
_EXPANSIONS = [
    (r"send (it )?back|wapas|vapas|lauta|लौटा|वापस|return|exchange", "return returns window"),
    (r"nobody (came|turned)|no one came|didn.?t (come|collect|pick)|pickup (miss|nahi)|reverse pickup", "missed pickup reverse pickup rescheduled"),
    (r"refund|money back|paisa|paise|पैसे|रिफ़ंड|रिफंड|रिफन्ड", "refund refunds initiated business days"),
    (r"kitne din|how many days|how long|kab tak|कितने दिन|कब तक|deadline|window", "days window timeline"),
    (r"cancel|कैंसिल|radd|रद्द|stop (it|my)", "cancel cancellation shipped"),
    (r"damag|broken|torn|crack|defect|tuta|tut |टूटा|खराब|kharab|फटा|wrong item|missing|not working", "damaged defective wrong missing items report 48 hours photos"),
    (r"warranty|guarantee|वारंटी|assured|claim", "warranty assured brand claim"),
    (r"not (at )?home|nobody home|ghar pe nahi|घर पर नहीं|miss(ed)? (the )?(courier|delivery)|attempt", "delivery attempts courier reschedule rto"),
    (r"late|delay|देर|compensat|voucher|dedh", "late deliveries compensation express wallet credit"),
    (r"where is|kahan|कहाँ|track|status|arrived|ship hua|shipped|kab aay|कब आ", "tracking order statuses shipped delivered"),
    (r"coupon|promo|offer|code|कूपन", "coupon coupons offers one per order"),
    (r"cod|cash on delivery|कैश ऑन", "cash on delivery cod fee"),
    (r"upi|card|net ?banking|wallet|वॉलेट", "refund wallet upi card net banking business days"),
    (r"human|insaan|इंसान|real person|talk to (a )?(person|someone|agent)|agent se", "human colleague escalation handover"),
    (r"verify|verification|otp|phone number|last (four|4)", "verification phone pincode identity"),
    (r"pincode|pin code|deliver (to|in)|ship(ping)? (to|abroad)|abroad|outside india|dubai|international", "delivery pincode serviceable india"),
    (r"plus|membership", "zentromart plus membership benefits"),
    (r"earbud|speaker|smartwatch|charging|bluetooth|pairing|sync|power on|turn on", "troubleshooting steps"),
    (r"seal|opened|khol|खोल|used|unopened|lipstick|perfume|parfum|cream", "beauty unopened seal intact condition"),
    (r"fee|charge|shulk|शुल्क|kitna lag", "fee charge fees"),
    (r"never (placed|ordered)|didn.?t (place|order)|unauthori|fraud|hacked|not me|नहीं (किया|दिया)", "hand over human order payment account activity they did not make"),
    (r"phone number|address|details of|who ordered|someone else|another (customer|person)|neighbou?r", "never share another customer's details verification phone"),
    (r"swimsuit|briefs|lingerie|\bbra\b|innerwear|swimwear|underwear|bikini", "innerwear lingerie swimwear not returnable"),
]


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOP and len(t) > 1 or t.isdigit()]


def stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _ngrams(text: str) -> list[str]:
    out = []
    for w in tokens(text):
        w = f" {w} "
        for n in (3, 4, 5):
            out += [w[i:i + n] for i in range(len(w) - n + 1)]
    return out


def expand(query: str) -> str:
    q = query.lower()
    extra = [add for pat, add in _EXPANSIONS if re.search(pat, q)]
    return query + " " + " ".join(extra)


def _catalog_tokens() -> set[str]:
    import csv, os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    out: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.update(stem(t) for t in tokens(r["title"] + " " + r["brand"]))
    return out


_SKU = re.compile(r"\b[A-Za-z]{3}-?\d{4}\b")


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [c.title + " " + c.text for c in chunks]
        self.doc_tokens = [[stem(t) for t in tokens(d)] for d in self.docs]
        self.df = Counter(t for d in self.doc_tokens for t in set(d))
        self.N = len(chunks)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.N
        self.tf = [Counter(d) for d in self.doc_tokens]
        # char n-gram TF-IDF channel (stdlib only), robust to Hinglish spellings
        grams = [Counter(_ngrams(d)) for d in self.docs]
        self.cdf = Counter(g for c in grams for g in c)
        self.cmat = [self._cvec(c) for c in grams]
        # words that only occur in product names carry no policy meaning
        self.catalog = {t for t in _catalog_tokens() if t not in self.df}

    def _cvec(self, counts: Counter) -> dict:
        v = {g: (1 + math.log(c)) * (math.log((1 + self.N) / (1 + self.cdf.get(g, 0))) + 1) for g, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def _char_scores(self, q: str) -> list[float]:
        qv = self._cvec(Counter(_ngrams(q)))
        return [sum(w * d.get(g, 0.0) for g, w in qv.items()) for d in self.cmat]

    def clean(self, query: str) -> str:
        q = _SKU.sub(" ", query)
        return " ".join(t for t in tokens(q) if stem(t) not in self.catalog)

    # --- scoring channels
    def idf(self, t: str) -> float:
        return math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))

    def _bm25(self, qtoks: list[str], k1=1.4, b=0.75) -> list[float]:
        scores = []
        for i, tf in enumerate(self.tf):
            dl, s = len(self.doc_tokens[i]), 0.0
            for t in set(qtoks):
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 5, include_superseded: bool = False):
        q = expand(self.clean(query))
        qt = [stem(t) for t in tokens(q) if t not in STOP]
        bm = self._bm25(qt)
        ch = self._char_scores(q)

        def ranks(vals):
            order = sorted(range(len(vals)), key=lambda i: -vals[i])
            return {i: r for r, i in enumerate(order)}
        rb, rc = ranks(bm), ranks(list(ch))
        fused = []
        for i, c in enumerate(self.chunks):
            if c.superseded and not include_superseded:
                continue
            s = 1 / (60 + rb[i]) + 0.7 / (60 + rc[i])
            fused.append((s * c.weight, i))
        fused.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in fused[:k]]

    # --- "do the documents answer this?" gate
    def coverage(self, query: str, top: int = 3) -> float:
        """IDF-weighted share of the question's salient *Latin-script* words
        found in the top retrieved sections.  Words the corpus has never seen
        (midnight, student, GST ...) count as maximally salient, so an
        unanswerable question scores low even if it shares a generic word
        such as 'delivery'.  Product-name words are ignored.  Questions with
        no Latin content word fall back to whether the topic expansion fired."""
        cleaned = self.clean(query)
        qt = [t for t in content_tokens(cleaned) if re.search(r"[a-z0-9]", t)]
        if not qt:
            return 0.5 if expand(cleaned).strip() != cleaned.strip() else 0.0
        hits = self.search(query, k=top)
        vocab = set()
        for c, _ in hits:
            vocab.update(stem(t) for t in tokens(c.title + " " + c.text))
        max_idf = math.log(1 + self.N)
        tot = got = 0.0
        for t in set(qt):
            w = self.idf(t) if t in self.df else max_idf
            tot += w
            if t in vocab:
                got += w
        return got / tot if tot else 0.0

    def topic_recognised(self, query: str) -> bool:
        c = self.clean(query)
        return expand(c).strip() != c.strip()
