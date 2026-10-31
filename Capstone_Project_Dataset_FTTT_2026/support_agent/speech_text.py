"""Turn a written reply into text that Gnani TTS reads correctly (STY-3):
dates with month names ('the 24th of October'), order IDs in groups of four
digits, amounts as spoken rupees.  The TTS docs warn that raw numbers/IDs are
mispronounced, so everything numeric is spelled out as words."""
from __future__ import annotations
import re

from .i18n import EN_MONTHS, HI_MONTHS

_HI_UNITS = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
             "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
             "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
             "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सरसठ "
             "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
             "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे "
             "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def english_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + english_words(n % 100) if n % 100 else "")
    if n < 100000:
        return english_words(n // 1000) + " thousand" + (" " + english_words(n % 1000) if n % 1000 else "")
    return english_words(n // 100000) + " lakh" + (" " + english_words(n % 100000) if n % 100000 else "")


def english_year(n: int) -> str:
    if 2001 <= n <= 2009:
        return english_words(n)
    return english_words(n // 100) + " " + (english_words(n % 100) if n % 100 else "hundred")


def english_ordinal(n: int) -> str:
    w = english_words(n)
    head, sep, last = w.rpartition("-" if "-" in w else " ")
    last = _ORD.get(last, last + ("th" if not last.endswith("y") else "")) if not last.endswith("y") else last[:-1] + "ieth"
    return head + sep + last


def hindi_words(n: int) -> str:
    if n < 100:
        return _HI_UNITS[n]
    if n < 1000:
        h, r = divmod(n, 100)
        return _HI_UNITS[h] + " सौ" + (" " + hindi_words(r) if r else "")
    if n < 100000:
        t, r = divmod(n, 1000)
        return hindi_words(t) + " हज़ार" + (" " + hindi_words(r) if r else "")
    l, r = divmod(n, 100000)
    return hindi_words(l) + " लाख" + (" " + hindi_words(r) if r else "")


def _num(n: int, lang: str, year: bool = False) -> str:
    if lang == "hi":
        return hindi_words(n)
    return english_year(n) if year else english_words(n)


def _ordinal(n: int) -> str:
    return english_ordinal(n)


def _digits(group: str, lang: str) -> str:
    if lang == "hi":
        return " ".join(_HI_UNITS[int(c)] for c in group)
    return " ".join(english_words(int(c)) for c in group)


def to_speakable(text: str, lang: str = "en") -> str:
    t = text
    # dates: '27 October 2026' / '27 अक्टूबर 2026'
    months = "|".join(EN_MONTHS + HI_MONTHS)

    def date_sub(m):
        day, mon, yr = int(m.group(1)), m.group(2), m.group(3)
        if lang == "hi":
            hi = HI_MONTHS[EN_MONTHS.index(mon)] if mon in EN_MONTHS else mon
            out = f"{hindi_words(day)} {hi}"
            return out + (f" {_num(int(yr), lang, True)}" if yr else "")
        out = f"the {_ordinal(day)} of {mon}"
        return out + (f" {_num(int(yr), lang, True)}" if yr else "")
    t = re.sub(rf"\b(\d{{1,2}}) ({months})(?: (\d{{4}}))?", date_sub, t)
    # amounts: ₹1,499 -> 'one thousand four hundred ninety-nine rupees'
    t = re.sub(r"₹\s?([\d,]+)", lambda m: f"{_num(int(m.group(1).replace(',', '')), lang)} " +
               ("रुपये" if lang == "hi" else "rupees"), t)
    # order IDs / phone suffixes / pincodes: groups of four digits read digit by digit
    t = re.sub(r"\b(\d{8})\b", lambda m: ", ".join(_digits(m.group(1)[i:i + 4], lang) for i in (0, 4)), t)
    t = re.sub(r"\b(\d{6})\b", lambda m: _digits(m.group(1), lang), t)
    # small stand-alone numbers (days, hours, steps)
    t = re.sub(r"\b(\d{1,4})\b", lambda m: _num(int(m.group(1)), lang), t)
    return t
