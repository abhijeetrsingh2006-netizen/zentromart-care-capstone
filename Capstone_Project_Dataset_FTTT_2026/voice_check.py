"""Step-by-step check of the Gnani voice path.

    python voice_check.py                         # offline checks + what is missing
    python voice_check.py --tts                   # needs GNANI_API_KEY: writes voice_samples/*.wav
    python voice_check.py --stt my_clip.wav --lang hi-IN
                                                  # 16 kHz mono 16-bit WAV -> transcript -> agent reply

Language codes for --lang: en-IN, hi-IN, hi-en (Hinglish is recognised as Hindi).
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import sys

import app
from support_agent.speech_text import to_speakable

SAMPLES = {
    "en": "Your refund of ₹1,499 was started on 12 October 2026 for order 89782837 and should reach you by 15 October 2026.",
    "hi": "आपका ₹49 का रिफ़ंड 15 अक्टूबर 2026 तक आ जाना चाहिए।",
    "hi-en": "Aapka refund 27 October 2026 tak aa jaana chahiye. Order 89782837 ke liye ₹79 credit hua hai.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--stt", metavar="WAV")
    ap.add_argument("--lang", default="en-IN", choices=list(app.STT_LANGS))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print("1. SDK installed :", bool(importlib.util.find_spec("gnani")), "(pip install -r requirements-voice.txt)")
    print("2. API key set   :", bool(app.GNANI_KEY), "(set GNANI_API_KEY)")
    print("3. What the TTS will actually read aloud:")
    for lang, text in SAMPLES.items():
        print(f"   [{lang}] {to_speakable(text, 'hi' if lang == 'hi' else 'en')}")
    if not app.GNANI_KEY:
        print("\nOffline checks done. Set GNANI_API_KEY to test the live services.")
        return
    if a.tts:
        os.makedirs("voice_samples", exist_ok=True)
        for lang, text in SAMPLES.items():
            audio = app.gnani_tts(text, lang)
            path = f"voice_samples/{lang}.wav"
            open(path, "wb").write(audio)
            print(f"   wrote {path} ({len(audio)} bytes) - listen: dates, IDs and rupees must sound natural")
    if a.stt:
        transcript = app.gnani_stt(open(a.stt, "rb").read(), app.STT_LANGS[a.lang])
        print("   transcript:", transcript)
        reply, sources, meta = app.response_for(transcript, {}, voice=True)
        print("   agent reply:", reply, "|", meta["decision"])


if __name__ == "__main__":
    main()
