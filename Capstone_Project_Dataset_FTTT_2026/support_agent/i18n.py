"""Tiny localisation helpers: language pick, Hindi month names, channel framing."""
from __future__ import annotations
import re
from datetime import date

HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
REPLY_LANGS = ("en", "hi", "hi-en")


def pick(lang: str, en: str, hi: str | None = None, hien: str | None = None) -> str:
    if lang == "hi" and hi:
        return hi
    if lang == "hi-en" and hien:
        return hien
    return en


def fdate(d: date, lang: str) -> str:
    """'27 October 2026' (Hindi script: '27 अक्टूबर 2026'); always with the year so a
    date can never be misread across a year boundary (STY-2: month name, not 04/05)."""
    month = (HI_MONTHS if lang == "hi" else EN_MONTHS)[d.month - 1]
    return f"{d.day} {month} {d.year}"


def rupees(n) -> str:
    return f"₹{int(n):,}"


def frame_for_channel(text: str, channel: str, lang: str, first_name: str | None = None) -> str:
    """STY-2: emails/tickets get greeting + answer first + sign-off; chat and
    voice stay short and plain."""
    if channel in ("email", "ticket"):
        greet = pick(lang, f"Hello{' ' + first_name if first_name else ''},",
                     f"नमस्ते{' ' + first_name if first_name else ''},",
                     f"Namaste{' ' + first_name if first_name else ''},")
        sign = pick(lang, "Regards,\nZentroMart Support (AI assistant)",
                    "धन्यवाद,\nZentroMart सपोर्ट (AI असिस्टेंट)",
                    "Dhanyavaad,\nZentroMart Support (AI assistant)")
        return f"{greet}\n\n{text}\n\n{sign}"
    return text
