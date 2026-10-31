"""Light-weight, dependency-free NLU: language, entities, intents, ASR clean-up.

Everything is rule-based on purpose: the checks that matter here (verification
digits, order IDs, dates, payment methods, escalation triggers) must be exact
and explainable, and customers' Hinglish / Hindi / ASR-mangled text is handled
by lexicons rather than by a model that could silently paraphrase a number.
"""
from __future__ import annotations
import difflib
import re
from datetime import date, timedelta

from .business import PRODUCTS, TODAY

# ----------------------------------------------------------------- language
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_OTHER_INDIC = {
    "ta": r"[\u0B80-\u0BFF]", "te": r"[\u0C00-\u0C7F]", "kn": r"[\u0C80-\u0CFF]",
    "ml": r"[\u0D00-\u0D7F]", "bn": r"[\u0980-\u09FF]", "gu": r"[\u0A80-\u0AFF]",
    "pa": r"[\u0A00-\u0A7F]",
}
_HINGLISH = set("""mujhe mera meri mere hai hain kya kaise kab kitne kitna kitni din ka ki ke ko pe par mein nahi nahin karo kar karna
karein kijiye wala wali shuru hua hui aaya aayi aur theek achha accha sawaal sawal chahiye paisa paise rupaye rupaiye parso kal aaj bhi
sakta sakti sakte hoon tak agle ek milega milegi lagta laga kahan kaha batao bataiye aapka aap hum hume bhej bhejna wapas vapas ise isko
kuch koi abhi phir lekin magar toh jaana chahiye hoga hogi""".split()) - {"do", "to", "the"}


def detect_language(text: str, fallback: str = "en") -> str:
    if not text.strip():
        return fallback
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return fallback                                  # e.g. "4533"
    dev = len(_DEVANAGARI.findall(text))
    if dev / len(letters) > 0.3:
        return "hi"
    for code, pat in _OTHER_INDIC.items():
        if len(re.findall(pat, text)) / len(letters) > 0.3:
            return code
    toks = re.findall(r"[a-z]+", text.lower())
    if sum(t in _HINGLISH for t in toks) >= 1 and \
       (sum(t in _HINGLISH for t in toks) >= 2 or len(toks) <= 4):
        return "hi-en"
    return "en"


# ------------------------------------------------------------- ASR clean-up
_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
                "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_PURE_DIGIT_WORDS = {k for k in _DIGIT_WORDS if k not in ("to", "too", "for", "oh")}
_FILLERS = re.compile(r"^(?:(?:uh|um|so|hello|haan|hi|hey|okay|ok|well)\s+)+")
_SKU_PREFIXES = sorted({s.split("-")[0].lower() for s in PRODUCTS})
_SKU_SPOKEN = re.compile(
    r"\b(" + "|".join(_SKU_PREFIXES) + r")[-\s]*((?:(?:zero|oh|one|two|to|too|three|four|five|six|seven|eight|nine|\d)\s*){1,4})\b")


def normalize_asr(text: str) -> str:
    """Repair typical speech-to-text artefacts so the same NLU can be used for
    typed and spoken input: spoken digits ('zero zero one to' -> 0012), spelled
    acronyms ('u p i'), split words ('re fund'), leading fillers."""
    t = text.strip()
    t = re.sub(r"[?!.,;]+(\s|$)", r"\1", t)
    low = t.lower()
    low = _FILLERS.sub("", low)
    low = re.sub(r"\bu\s*p\s*i\b", "upi", low)
    low = re.sub(r"\bc\s*o\s*d\b", "cod", low)
    low = re.sub(r"\bre\s+fund", "refund", low)

    def sku(m):
        digs = "".join(_DIGIT_WORDS.get(w, w) for w in m.group(2).split())
        return f"{m.group(1)}-{digs.zfill(4)[:4]} " if digs else m.group(0)
    low = _SKU_SPOKEN.sub(sku, low)

    # runs of spoken digits that contain at least one unambiguous digit word
    words = low.split()
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and words[j] in _DIGIT_WORDS:
            j += 1
        run = words[i:j]
        if run and any(w in _PURE_DIGIT_WORDS for w in run) and \
           (len(run) >= 2):
            out.append("".join(_DIGIT_WORDS[w] for w in run))
            i = j
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


# ------------------------------------------------------------------ entities
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_MONTHS.update({"जनवरी": 1, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7,
                "अगस्त": 8, "सितंबर": 9, "अक्टूबर": 10, "नवंबर": 11, "दिसंबर": 12})


def ascii_digits(text: str) -> str:
    return text.translate(_DEV_DIGITS)


def extract_order_id(text: str) -> str | None:
    """8-digit order ID, possibly written in groups ('8978 2837', '8978-2837')."""
    t = ascii_digits(text)
    m = re.search(r"(?<!\d)(\d{4})[\s\-]?(\d{4})(?!\d)", t)
    return m.group(1) + m.group(2) if m else None


def extract_verification(text: str, order_id: str | None = None) -> str | None:
    """A standalone 6-digit pincode or 4-digit phone suffix (order ID removed)."""
    t = re.sub(r"\b[A-Za-z]{3}-?\d{4}\b", " ", ascii_digits(text))      # SKUs are not proofs
    if order_id:
        t = re.sub(r"(\d{4})[\s\-]?(\d{4})", lambda m: " " if m.group(1) + m.group(2) == order_id else m.group(0), t)
    t = re.sub(r"\b20\d\d\b(?=\s*(?:$|\.|,|को|ko))", " ", t) if re.search(
        r"(?i)jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", t) else t
    six = re.findall(r"(?<!\d)\d{6}(?!\d)", t)
    if six:
        return six[-1]
    four = re.findall(r"(?<!\d)\d{4}(?!\d)", t)
    if four:
        return four[-1]
    spaced = re.sub(r"[\s\-]", "", t)               # "4 5 3 3"
    m = re.fullmatch(r"\D*(\d{4}|\d{6})\D*", spaced)
    return m.group(1) if m else None


def extract_sku(text: str) -> str | None:
    m = re.search(r"\b([A-Za-z]{3})-?(\d{4})\b", text)
    if m:
        sku = f"{m.group(1).upper()}-{m.group(2)}"
        if sku in PRODUCTS:
            return sku
    return None


_TITLE_TOKENS = {}
for _sku, _p in PRODUCTS.items():
    core, _, variant = _p["title"].partition(",")
    _TITLE_TOKENS[_sku] = ([t for t in re.findall(r"[a-z0-9]+", core.lower())],
                           [t for t in re.findall(r"[a-z0-9]+", variant.lower())])


def match_product(text: str, candidates=None) -> tuple[str | None, bool]:
    """Resolve a product mention to a SKU. Returns (sku, ambiguous).  Matching is
    fuzzy per token so 'kurtha' / dropped brand words from speech still resolve."""
    sku = extract_sku(text)
    if sku:
        return sku, False
    if candidates is not None and len(candidates) == 1:
        only = next(iter(candidates))
        core = _TITLE_TOKENS[only][0]
        if sum(t in set(re.findall(r"[a-z0-9]+", text.lower())) for t in core) >= 2:
            return only, False
    q = re.findall(r"[a-z0-9]+", text.lower())
    qs = set(q)
    best, best_s, ties = None, 0.0, 0
    for sku, (core, variant) in _TITLE_TOKENS.items():
        if candidates is not None and sku not in candidates:
            continue
        if len(core) < 2:
            continue
        got = 0.0
        for t in core:
            if t in qs:
                got += 1
            elif len(t) > 4:
                m = difflib.get_close_matches(t, q, n=1, cutoff=0.85)
                got += 0.9 if m else 0
        s = got / len(core)
        s += 0.05 * sum(v in qs for v in variant)          # colour as tie-breaker
        if s > best_s + 1e-9:
            best, best_s, ties = sku, s, 0
        elif abs(s - best_s) < 1e-9:
            ties += 1
    if best and best_s >= 0.7:
        return best, ties > 0
    return None, False


def extract_dates(text: str) -> list[date]:
    t = ascii_digits(text.lower())
    found: list[date] = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", t):
        found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    names = "|".join(sorted((k for k in _MONTHS), key=len, reverse=True))
    for m in re.finditer(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({names})(?:\s*,?\s*(\d{{4}}))?", t):
        try:
            found.append(date(int(m.group(3) or TODAY.year), _MONTHS[m.group(2)], int(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(rf"({names})\s+(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?", t):
        try:
            found.append(date(int(m.group(3) or TODAY.year), _MONTHS[m.group(1)], int(m.group(2))))
        except ValueError:
            pass
    return found


def relative_date(text: str) -> date | None:
    t = text.lower()
    if re.search(r"day before yesterday|parso|parson|परसों", t):
        return TODAY - timedelta(days=2)
    if re.search(r"yesterday|\bkal\b|कल", t):
        return TODAY - timedelta(days=1)
    if re.search(r"\btoday\b|\baaj\b|आज", t):
        return TODAY
    return None


def extract_payment_method(text: str) -> str | None:
    t = text.lower()
    if re.search(r"cash on delivery|\bcod\b|कैश ऑन", t):
        return "cod"
    if re.search(r"net ?banking|नेट ?बैंकिंग", t):
        return "netbanking"
    if re.search(r"\bupi\b|यूपीआई", t):
        return "upi"
    if re.search(r"credit|debit|\bcard\b|कार्ड", t):
        return "card"
    if re.search(r"wallet|वॉलेट", t):
        return "wallet"
    return None


# ------------------------------------------------------------------- intents
def _any(pats, t):
    return any(re.search(p, t) for p in pats)


_RULES: list[tuple[str, list[str]]] = [
    ("safety", [r"swell|swollen|bulg(e|ing)|burning smell|burnt smell|smells? (like )?(burn|smoke)|smoke|spark|caught fire|on fire|exploded|overheat|too hot|injur|shock", r"चिंगारी|जलने की बू|धुआँ|धुआं|फूल (गई|गया|रही)"]),
    ("human", [r"human|real person|(talk|speak|connect|transfer)\w*( me)?( to| with)? (a |an )?(person|someone|agent|representative|executive|colleague|manager)|insaan|insan\b|customer care|manushya", r"इंसान|किसी व्यक्ति|कस्टमर केयर|एजेंट से|मानव"]),
    ("legal", [r"lawyer|legal action|consumer court|consumer forum|police|\bfir\b|\bcourt\b|sue\b|complaint (to|with) (the )?(authorit|consumer)", r"कानूनी|पुलिस|उपभोक्ता (अदालत|फोरम)|कोर्ट|वकील"]),
    ("fraud", [r"never (placed|ordered|made|did)|did ?n.?t (place|order|make|do)|not (placed|ordered|made) by me|unauthori[sz]ed|fraud|hacked|someone (else )?(used|ordered|placed|logged)|not me\b|don.?t recogni[sz]e|maine order nahi|nahi kiya tha", r"मैंने .{0,30}(नहीं|ना) (किया|दिया|लिया)|धोखाधड़ी|अनधिकृत"]),
    ("privacy", [r"(phone|mobile|number|address|email|details?|name) of (the |that |another |some )?(person|customer|someone|buyer|neighbou?r|guy|man|woman|who)|who (ordered|bought|placed)|neighbou?r.?s? (order|number|phone)|another customer|someone else.?s (order|number|address)"]),
    ("repeat_contact", [r"(third|3rd|fourth|4th|fifth|\d+(st|nd|rd|th)) time|teesri baar|chauthi baar|again and again|multiple times|several times|keep (calling|contacting)", r"तीसरी बार|चौथी बार|बार-बार|कई बार"]),
    ("cancel", [r"cancel|stop (it|an? |the |my |this )?(order|parcel|delivery|shipment)?|radd|रद्द|कैंसिल|कैन्सल|रोक"]),
    ("damaged", [r"damag|broken|torn|crack|defect|tuta|tuti|toota|टूटा|टूटी|खराब|kharab|फटा|फटी|wrong (item|product|size|colou?r)|missing|expired|dead on arrival|doesn.?t work|not working|galat (saman|item)|गलत (सामान|आइटम)|क्षतिग्रस्त"]),
    ("warranty", [r"warrant|guarantee|assured|वारंटी|गारंटी"]),
    ("late_compensation", [r"\blate\b|delay|देर|voucher|compensat|dedh|a day late|arrived late|came late|aaya late"]),
    ("delivery_attempts", [r"not (at )?home|nobody (is )?(at |in )?home|no one (is )?(at |in )?home|miss(ed)? (the )?(courier|delivery|executive)|delivery attempt|घर पर नहीं|ghar (pe|par) nahi|failed attempt"]),
    ("troubleshooting", [r"won.?t (turn|power|switch|charge|pair|connect)|not (turning|charging|syncing|powering|pairing|connecting)|one (side|earbud)|charging slow|slow charging|doesn.?t (turn|charge|sync|pair)|stopped (working|charging)|no sound|not (on|power)|(sync|charge|charging|on|chalu|connect|pair|chal) nahi|nahi (ho rahi|ho raha|chal|chalu|charge|sync|connect)|नहीं (चल|हो रह|चालू|चार्ज)"]),
    ("return", [r"return|send (it )?back|wapas (kar|bhej|de)|vapas|वापस|लौटा|lauta|exchange|swap|replace|not (my )?size|doesn.?t fit|too small|too big|don.?t (like|want)"]),
    ("refund", [r"refund|money back|paisa|paise|पैसे|रिफ़ंड|रिफंड|रिफन्ड|re-?fund|reimburse"]),
    ("order_status", [r"where is|where.?s my|kahan|कहाँ|track|status|arrive|ship(ped)? (hua|hui|kab|when)|when (was|did|will).*(ship|deliver|arrive)|kab (aay|mil|deliver)|कब (आ|मिल|ship|शिप)|ship hua|hasn.?t (arrived|come)|not (yet )?(arrived|received|delivered)|delivery date|expected"]),
    ("coupon", [r"coupon|promo( code)?|discount code|voucher code|कूपन|offer code"]),
    ("plus", [r"\bplus\b|membership|सदस्य"]),
    ("delivery_area", [r"deliver(y)? (to|in) |ship(ping)? (to|abroad)|do you (deliver|ship)|abroad|outside india|international|dubai|usa|uk\b|nepal|pincode serviceable|cod (available|possible)|express (available|possible)"]),
    ("delivery_fee", [r"delivery (fee|charge)|shipping (fee|charge|cost)|free delivery|cod (fee|charge)|express (fee|charge|cost)"]),
]


def classify(text: str) -> list[str]:
    t = text.lower()
    return [name for name, pats in _RULES if _any(pats, t)]


def asks_for_other_people(text: str) -> bool:
    return "privacy" in classify(text)


def mentions_opened(text: str) -> bool:
    return bool(re.search(r"opened|open (kar|kiya)|used|tried|khol|खोल|इस्तेमाल|use kar|unsealed|seal (tod|broken)", text.lower()))


def mentions_plus(text: str) -> bool:
    return bool(re.search(r"\bplus\b|प्लस", text.lower()))


def mentions_changed_mind(text: str) -> bool:
    return bool(re.search(r"don.?t like|do not like|changed my mind|not (my )?size|doesn.?t fit|too (small|big)|pasand nahi|पसंद नहीं|fit nahi|फिट नहीं|just don.?t|send (it )?back", text.lower()))


def mentions_defect(text: str) -> bool:
    return "damaged" in classify(text) or bool(re.search(r"torn|defect|wrong|damag|broken|crack|फटा|टूटा|खराब", text.lower()))


_CATEGORY_WORDS = [
    ("fashion", r"shoe|sneaker|footwear|sandal|jeans|shirt|t-?shirt|kurta|kurti|dress|saree|clothes|clothing|jacket|trouser|pant|bag\b|tote|जूते|जूता|कपड़े|कुर्ता|साड़ी|बैग"),
    ("innerwear", r"innerwear|lingerie|swimsuit|swimwear|briefs|\bbra\b|underwear|bikini"),
    ("electronics", r"electronics?|gadget|phone|laptop|earbud|headphone|speaker|smartwatch|watch|charger|power bank|tablet|camera|\btv\b|televis"),
    ("beauty_sealed", r"lipstick|perfume|parfum|cream|serum|shampoo|makeup|lotion|moisturi"),
    ("grocery", r"grocery|vegetable|fruit|milk|fresh|atta|rice"),
]


def category_from_text(text: str) -> str | None:
    t = text.lower()
    return next((c for c, p in _CATEGORY_WORDS if re.search(p, t)), None)
