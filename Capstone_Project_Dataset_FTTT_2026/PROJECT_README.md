# ZentroMart Care — Capstone Project

A locally runnable support assistant for the synthetic ZentroMart dataset. It combines Markdown policy retrieval, catalog lookups, identity-gated order lookup, ticket handover, and an optional Gnani speech interface.

## Run it

Use Python 3.10 or newer. No packages are needed for chat and policy search.

```powershell
cd Capstone_Project_Dataset_FTTT_2026
python app.py
```

Open <http://127.0.0.1:8000>. For local speech recognition and speech playback, set a Gnani API key in the environment before starting the app:

```powershell
$env:GNANI_API_KEY = "your-key"
python app.py
```

Install the optional voice SDK with `python -m pip install -r requirements-voice.txt`. Voice capture records mono 16 kHz WAV in the browser and streams PCM frames to Gnani STT over its WebSocket. The reply is spoken by Gnani TTS in the language it is written in (English, Hindi or Hinglish voices), with dates, order IDs and rupee amounts spelled out as words. Optional: set `GNANI_STT_KEYWORDS="ZentroMart,pincode"` to bias recognition toward your vocabulary. The transcript is shown for review before it is sent. The key stays on the server; configure only a key you are authorized to use. Voice service requires network access and a valid Gnani key. Chat, order tools, retrieval, and tickets work offline. See [Gnani STT realtime documentation](https://docs.gnani.ai/api/STT/stt-websocket) and [Gnani TTS realtime documentation](https://docs.gnani.ai/api/TTS/tts-websocket) for supported language codes and model details.

## Project behavior

- Searches all 21 Markdown knowledge documents and keeps section IDs, `status`, and `effective_from` metadata. Current policy documents rank above other source types; superseded material is strongly down-ranked; marketing copy is not policy authority.
- Uses the fixed dataset date **20 October 2026** and uses the included business holiday calendar.
- Answers come from the rule-based engine in `support_agent/` (standard library only): exact facts (return windows, pickup fee, refund dates on the business-day calendar, cancellation rules, high-value damage, escalation triggers) are computed from the CSVs and the current policy; retrieval (BM25 + character n-grams, precedence-aware) supplies the cited sections and the long tail; unanswerable questions get an honest "not documented" reply with an offer of a human colleague. **The practice answers are not used as a lookup.**
- Asks for an order ID and then one verification value (registered phone last four digits or delivery pincode). Allows one retry. Order details stay in in-memory session state, expire after 30 minutes of inactivity, and clear on New conversation or server restart.
- Escalates human requests, legal concerns, and product safety risks by creating a ticket with the provided offline mock tool. It reports success only when the tool confirms `created=True`.
- Does not use `support_logs.csv` as policy authority.

## Dataset and evaluation

Run `python validate_dataset.py` to validate the dataset and `python evaluate.py [-v]` for a strict check through the same entry point as the web app: 52 questions as text and as ASR text, 23 conversations (decision, required facts, forbidden claims, exact tool calls and arguments, safe ticket summaries), cited-evidence recall, raw retrieval recall@5, and a scan of all 1,000 orders for crashes or pre-verification leaks. Exit code 1 on any failure.

`python voice_check.py` checks the voice path step by step.

## API surface

- `GET /api/health` reports local readiness and whether Gnani is configured.
- `POST /api/reset` clears the current conversation's verification state.
- `GET /api/search?q=...` retrieves up to eight policy sections.
- `POST /api/chat` takes `{ "message": "...", "voice": false }` (`voice: true` repairs spoken digits from speech recognition) and returns an answer, cited sections, and tool metadata.
- `POST /api/stt` accepts WAV bytes and an `X-Language-Code` header.
- `POST /api/tts` accepts `{ "text": "...", "language": "auto", "voice": "Nalini" }` and returns WAV audio.

## Limitations

This is an educational local demo. The identity check is the supplied synthetic mock, not production authentication. Replies come from rules, templates (English, Hindi, Hinglish) and retrieved excerpts rather than a hosted LLM, so other languages fall back to English. Gnani requires an account/API key; no key or audio sample is bundled. The voice interface currently exposes one-shot recording and playback rather than continuous duplex calling.
