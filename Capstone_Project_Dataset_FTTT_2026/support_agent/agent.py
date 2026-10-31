"""ZentroMart support agent: verification state machine + policy handlers.

Design (see REPORT.md):
  * Facts that must be exact (deadlines, fees, refund dates, escalation
    triggers) come from deterministic rules over the CSVs and the current
    policy sections - never from free-text generation.
  * Retrieval (hybrid BM25 + char n-gram, precedence-aware) supplies evidence
    IDs for every reply and answers the long tail of policy questions.
  * An optional LLM layer (llm.py) can translate / polish a reply; the agent
    works fully offline without it.
"""
from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date

from . import business as B
from . import nlu
from .i18n import REPLY_LANGS, fdate, frame_for_channel, pick, rupees
from .kb import load_kb
from .retrieval import Retriever

ROOT = B.ROOT
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools import tools as _tools  # noqa: E402  (provided mock tools)

HIGH_VALUE = 10_000
NEED_ORDER = object()


@dataclass
class AgentReply:
    text: str
    decision: str                       # answer | clarify | escalate | refuse
    evidence: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    ticket_id: str | None = None
    language: str = "en"
    intent: str = ""
    speakable: str = ""                 # text prepared for TTS (voice channel)


@dataclass
class Session:
    lang: str = "en"
    introduced: bool = False
    state: str = "idle"                 # idle | await_order | await_verify | failed
    order_id: str | None = None
    order: dict | None = None           # populated ONLY after successful verification
    attempts: int = 0
    pending: str | None = None          # customer's request waiting for verification
    last_sku: str | None = None
    facts: list[str] = field(default_factory=list)
    tickets: list[str] = field(default_factory=list)


class SupportAgent:
    def __init__(self, tools=None):
        self.kb = load_kb(os.path.join(ROOT, "kb"))
        self.retriever = Retriever(self.kb)
        self.chunks = {c.section_id: c for c in self.kb}
        self.tools = tools or _tools

    # ------------------------------------------------------------ public API
    def new_session(self) -> Session:
        return Session()

    def respond(self, s: Session, message: str, channel: str = "chat", asr: bool = False) -> AgentReply:
        text = nlu.normalize_asr(message) if asr else message.strip()
        lang = nlu.detect_language(text, fallback=s.lang)
        s.lang = lang
        rl = lang if lang in REPLY_LANGS else "en"
        ctx = _Turn(self, s, text, rl, channel)
        reply = ctx.run()
        if not s.introduced:
            intro = pick(rl, "I'm ZentroMart's AI support assistant.",
                         "मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ।",
                         "Main ZentroMart ka AI support assistant hoon.")
            reply.text = f"{intro} {reply.text}"
            reply.evidence.insert(0, "STY-1")
            s.introduced = True
        first = (s.order or {}).get("customer_first_name")
        reply.text = frame_for_channel(reply.text, channel, rl, first)
        reply.language = rl
        if channel == "voice":
            from .speech_text import to_speakable
            reply.speakable = to_speakable(reply.text, rl)
        return reply


# =====================================================================
class _Turn:
    """Everything that happens while answering one customer message."""

    def __init__(self, agent: SupportAgent, s: Session, text: str, lang: str, channel: str):
        self.a, self.s, self.t, self.l, self.ch = agent, s, text, lang, channel
        self.intents = nlu.classify(text)
        self.calls: list[str] = []

    # -- small helpers -----------------------------------------------------
    def P(self, en, hi=None, hien=None):
        return pick(self.l, en, hi, hien)

    def D(self, d: date) -> str:
        return fdate(d, self.l)

    def reply(self, text, decision="answer", ev=None, intent="", ticket=None) -> AgentReply:
        return AgentReply(text=text, decision=decision, evidence=list(dict.fromkeys(ev or [])),
                          tool_calls=self.calls, ticket_id=ticket, intent=intent)

    # -- main dispatch -----------------------------------------------------
    def run(self) -> AgentReply:
        s, t = self.s, self.t
        oid = nlu.extract_order_id(t)
        trigger = next((i for i in ("safety", "human", "legal", "fraud") if i in self.intents), None)

        # 1. an escalation trigger always wins over finishing verification
        if trigger:
            return self.escalate(trigger)

        # 2. other people's data is never shared
        if "privacy" in self.intents:
            return self.reply(self.P(
                "I'm sorry, I can't share any details about another customer or their order. "
                "I'm happy to help with your own orders and our policies.",
                "क्षमा करें, मैं किसी दूसरे ग्राहक या उनके ऑर्डर की जानकारी साझा नहीं कर सकता। "
                "आपके अपने ऑर्डर और हमारी नीतियों में मैं ज़रूर मदद करूँगा।",
                "Maaf kijiye, main kisi doosre customer ya unke order ki jaankari share nahi kar sakta. "
                "Aapke apne order aur hamari policies mein main madad kar sakta hoon."),
                "refuse", ["ESC-4", "IDV-4"], "privacy")

        # 3. verification flow ------------------------------------------------
        if oid and oid != s.order_id:
            s.order_id, s.order, s.attempts, s.state = oid, None, 0, "await_verify"
            self.calls.append(f"lookup_order(order_id={oid})")
            self.a.tools.lookup_order(oid)               # unverified call: returns no private data
            if self.intents or self._substantive():
                s.pending = t
            proof = nlu.extract_verification(t, oid)
            if proof:
                return self.verify(proof)
            return self.ask_verification()
        if s.state == "await_verify" and not trigger:
            proof = nlu.extract_verification(t, s.order_id)
            if proof:
                return self.verify(proof)
            if not self.intents:
                return self.ask_verification()
        if s.state == "failed" and self._order_specific():
            return self.refuse_after_failure()

        # 4. answer ----------------------------------------------------------
        out = self.answer()
        if out is NEED_ORDER:
            if s.state == "await_verify":
                return self.ask_verification()
            s.pending = t
            s.state = "await_order"
            return self.reply(self.P(
                "Sure, I can check that. Could you please share your order ID?",
                "ज़रूर, मैं देख लेता हूँ। कृपया अपना ऑर्डर ID बताइए।",
                "Zaroor, main check kar leta hoon. Kripya apna order ID bataiye."),
                "clarify", ["IDV-2"], "need_order")
        return out

    def _substantive(self) -> bool:
        return bool(re.search(r"[a-z\u0900-\u097F]{3,}", re.sub(r"(?i)order|number|hai|नंबर|ऑर्डर|है", "", self.t))) and len(self.t.split()) > 6

    def _order_specific(self) -> bool:
        return bool(set(self.intents) & {"order_status", "refund", "cancel", "return", "damaged",
                                        "late_compensation", "repeat_contact"}) \
            and bool(re.search(r"\b(my|mera|meri|mere)\b|मेरा|मेरी|मेरे|\border\b|ऑर्डर", self.t.lower()))

    # -- verification --------------------------------------------------------
    def ask_verification(self, retry=False) -> AgentReply:
        if retry:
            return self.reply(self.P(
                "That doesn't match our records. Please try again with the last 4 digits of your registered phone number, or the delivery pincode.",
                "यह हमारे रिकॉर्ड से मेल नहीं खाता। कृपया रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड दोबारा बताइए।",
                "Yeh hamare record se match nahi hua. Kripya registered phone number ke aakhri 4 digit ya delivery pincode dobara bataiye."),
                "clarify", ["IDV-3"], "verification")
        return self.reply(self.P(
            "To protect your account, please tell me the last 4 digits of your registered phone number or the delivery pincode.",
            "आपके अकाउंट की सुरक्षा के लिए, कृपया रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड बताइए।",
            "Aapke account ki suraksha ke liye, kripya registered phone number ke aakhri 4 digit ya delivery pincode bataiye."),
            "clarify", ["IDV-2"], "verification")

    def verify(self, proof: str) -> AgentReply:
        s = self.s
        self.calls.append(f"lookup_order(order_id={s.order_id}, verification_value={proof})")
        res = self.a.tools.lookup_order(s.order_id, verification_value=proof)
        if res.get("found") and res.get("verified"):
            s.order, s.state, s.attempts = res, "idle", 0
            s.facts.append(f"order {s.order_id} verified")
            if len(res["items"]) == 1:
                s.last_sku = res["items"][0]["sku"]
            pend, s.pending = s.pending, None
            if pend and pend != self.t:
                self.t = pend
                self.intents = nlu.classify(pend)
                out = self.answer()
                if out is not NEED_ORDER:
                    out.text = self.P("Thank you, you're verified. ", "धन्यवाद, आपकी पहचान सत्यापित हो गई है। ",
                                      "Dhanyavaad, aapki verification ho gayi. ") + out.text
                    out.tool_calls = self.calls
                    return out
            return self.reply(self.P(
                f"Thank you, {res['customer_first_name']}, you're verified. How can I help with this order?",
                f"धन्यवाद {res['customer_first_name']}, आपकी पहचान सत्यापित हो गई है। इस ऑर्डर में मैं कैसे मदद करूँ?",
                f"Dhanyavaad {res['customer_first_name']}, aapki verification ho gayi. Is order mein main kaise madad karoon?"),
                "answer", ["IDV-2"], "verified")
        s.attempts += 1
        if s.attempts == 1:
            return self.ask_verification(retry=True)
        s.state = "failed"
        return self.refuse_after_failure(first=True)

    def refuse_after_failure(self, first=False) -> AgentReply:
        return self.reply(self.P(
            "I'm sorry, I can't share order details because I couldn't verify the registered details. "
            "For security, please contact us from your registered phone number; I can still help with general policy questions.",
            "क्षमा करें, रजिस्टर्ड जानकारी सत्यापित न होने के कारण मैं ऑर्डर की जानकारी साझा नहीं कर सकता। "
            "सुरक्षा के लिए कृपया अपने पंजीकृत फ़ोन नंबर से हमसे संपर्क करें; सामान्य नीतियों के बारे में मैं मदद कर सकता हूँ।",
            "Maaf kijiye, registered details verify na hone ki wajah se main order ki jaankari share nahi kar sakta. "
            "Suraksha ke liye kripya apne registered phone number se humse sampark karein; general policy sawaalon mein main madad kar sakta hoon."),
            "refuse", ["IDV-3"], "verification_failed")

    # -- escalation ----------------------------------------------------------
    def escalate(self, reason: str, detail: str = "", intent: str | None = None, extra_ev=None) -> AgentReply:
        s = self.s
        why = {
            "safety": ("product safety risk (swelling battery / burning smell / sparks / heat)", "safety_risk", ["DWM-5", "ESC-1"]),
            "human": ("customer asked for a human agent", "customer_requested_human", ["ESC-1"]),
            "legal": ("customer mentioned legal action, consumer court or police", "legal_threat", ["ESC-1"]),
            "fraud": ("customer reports activity they did not make", "unauthorised_activity", ["ESC-1"]),
            "repeat": ("third or later contact about the same issue within 7 days", "repeat_contact", ["ESC-1", "ESC-3"]),
            "high_value": ("damaged/defective/wrong item above ₹10,000 - Priority Resolution only", "high_value_damage", ["DWM-3", "ESC-1", "ESC-4"]),
            "refund_overdue": ("refund more than 3 business days past its maximum timeline", "refund_overdue", ["ESC-1", "REF-2", "REF-6"]),
        }[reason]
        order = s.order or {}
        redacted = re.sub(r"(?<!\d)(\d{4}|\d{6})(?!\d)", "[number removed]", self.t)
        summary = (f"Issue: {detail or why[0]}. Detected intent: {intent or reason}. "
                   f"Order ID: {s.order_id or 'not provided'}. Customer said: \"{redacted[:180]}\". "
                   f"Checked: identity {'verified' if order else 'not verified'}"
                   f"{'; ' + '; '.join(s.facts[-3:]) if s.facts else ''}. Escalation reason: {why[0]}.")
        self.calls.append("create_ticket()")
        res = self.a.tools.create_ticket(summary=summary, order_id=s.order_id, reason=why[1], intent=intent or reason)
        ev = why[2] + (extra_ev or [])
        if res.get("created"):
            tid = res["ticket_id"]
            s.tickets.append(tid)
            base = {
                "safety": ("Please stop using the product right away and keep it away from anything flammable. ",
                           "कृपया प्रोडक्ट का इस्तेमाल तुरंत बंद कर दें और उसे ज्वलनशील चीज़ों से दूर रखें। ",
                           "Kripya product ka istemaal turant band kar dein aur use jalne wali cheezon se door rakhein. "),
            }.get(reason, ("", "", ""))
            core = self.P(
                f"I've created ticket {tid} and a human colleague will contact you. I can't approve or promise any outcome myself.",
                f"मैंने टिकट {tid} बना दिया है और हमारे सहयोगी आपसे संपर्क करेंगे। मैं खुद कोई नतीजा तय या वादा नहीं कर सकता।",
                f"Maine ticket {tid} bana diya hai aur hamare colleague aapse sampark karenge. Main khud koi outcome approve ya promise nahi kar sakta.")
            if reason == "refund_overdue" and order.get("return_rrn"):
                core = self.P(f"Your refund reference number (RRN) is {order['return_rrn']}, which your bank can trace. ",
                              f"आपका रिफ़ंड रेफ़रेंस नंबर (RRN) {order['return_rrn']} है, जिसे आपका बैंक ट्रेस कर सकता है। ",
                              f"Aapka refund reference number (RRN) {order['return_rrn']} hai, jise aapka bank trace kar sakta hai. ") + core
            return self.reply(base[self.P(0, 1, 2) if False else {"en": 0, "hi": 1, "hi-en": 2}[self.l]] + core,
                              "escalate", ev, reason, tid)
        return self.reply(self.P(
            "I couldn't create the handover ticket just now, so I can't confirm that a colleague has been notified. Please try again in a moment or contact support directly.",
            "मैं अभी टिकट नहीं बना पाया, इसलिए पुष्टि नहीं कर सकता कि सहयोगी को सूचना मिली है। कृपया थोड़ी देर में दोबारा कोशिश करें या सपोर्ट से सीधे संपर्क करें।",
            "Main abhi ticket nahi bana paaya, isliye confirm nahi kar sakta ki colleague ko inform kiya gaya hai. Kripya thodi der baad dobara try karein ya support se seedhe sampark karein."),
            "escalate", ev, reason)

    # -- policy handlers -----------------------------------------------------
    def answer(self):
        s, it = self.s, self.intents
        if "repeat_contact" in it:
            if s.order:
                if (s.order.get("contacts_last_7_days") or 0) >= 2:
                    return self.escalate("repeat", intent="repeat_contact")
            else:
                return NEED_ORDER
        if "refund" in it and "return" in it and re.search(r"refund|money back|paisa|paise|पैसे|रिफ़ंड|रिफंड", self.t.lower()) \
                and not re.search(r"window|deadline|how many days|kitne din|कितने दिन|can i (still )?return|eligible|वापस कर सक|return kar sak|how long (do|can|to)|last day", self.t.lower()):
            it = [i for i in it if i != "return"]
            self.intents = it
        order_ctx = [("cancel", self.h_cancel), ("damaged", self.h_damaged), ("warranty", self.h_warranty),
                     ("late_compensation", self.h_late), ("delivery_attempts", self.h_attempts),
                     ("troubleshooting", self.h_troubleshoot), ("return", self.h_return),
                     ("refund", self.h_refund), ("order_status", self.h_status), ("coupon", self.h_coupon),
                     ("plus", self.h_plus), ("delivery_area", self.h_area), ("delivery_fee", self.h_fees)]
        for name, fn in order_ctx:
            if name in it:
                out = fn()
                if out is not None:
                    return out
        return self.h_general()

    # product / order helpers
    def product(self):
        o = self.s.order
        cand = {i["sku"] for i in o["items"]} if o else None
        sku, amb = nlu.match_product(self.t, cand) if cand else (None, False)
        if not sku:
            sku, amb = nlu.match_product(self.t)
        if sku:
            self.s.last_sku = sku
            return B.PRODUCTS[sku]
        if self.s.last_sku:
            return B.PRODUCTS[self.s.last_sku]
        o = self.s.order
        if o and len(o["items"]) == 1:
            return B.PRODUCTS[o["items"][0]["sku"]]
        return None

    def order_item_for(self, p):
        o = self.s.order
        if not (o and p):
            return None
        return next((i for i in o["items"] if i["sku"] == p["sku"]), None)

    def which_item(self):
        o = self.s.order
        names = "; ".join(i["title"] for i in o["items"]) if o else ""
        return self.reply(self.P(
            f"Which item do you mean{': ' + names if names else ''}?",
            f"आप किस आइटम के बारे में पूछ रहे हैं{': ' + names if names else ''}?",
            f"Aap kis item ke baare mein pooch rahe hain{': ' + names if names else ''}?"),
            "clarify", ["RET-2"], "need_item")

    # ---- returns
    def h_return(self):
        t = self.t.lower()
        o = self.s.order
        p = self.product()
        if p is None and re.search(r"pickup|pick-up|collection|\bfee\b|charge|शुल्क|फ़ीस|फीस", t):
            cat0 = nlu.category_from_text(self.t) or "fashion"
            plus0 = bool(o and o.get("is_plus")) or nlu.mentions_plus(self.t)
            if cat0 == "fashion" and not nlu.mentions_defect(self.t):
                msg = (self.P("No, the ₹49 reverse pickup fee is waived for ZentroMart Plus members, even when you simply don't like the item.",
                              "नहीं, ZentroMart Plus सदस्यों के लिए ₹49 की रिवर्स पिकअप फ़ीस माफ़ है, आइटम पसंद न आने पर भी।",
                              "Nahi, ZentroMart Plus members ke liye ₹49 ki reverse pickup fee maaf hai, item pasand na aane par bhi.") if plus0 else
                       self.P("Yes, a ₹49 reverse pickup fee applies to change-of-mind returns of clothing, footwear and bags; it is waived for Plus members and for damaged or wrong items.",
                              "हाँ, कपड़े, जूते और बैग की पसंद न आने पर वापसी में ₹49 पिकअप फ़ीस लगती है; Plus सदस्यों और खराब/गलत आइटम के लिए यह माफ़ है।",
                              "Haan, kapde, footwear aur bags ke change-of-mind return par ₹49 pickup fee lagti hai; Plus members aur damaged/wrong items ke liye yeh maaf hai."))
            else:
                msg = self.P("There is no reverse pickup fee for damaged, defective or wrong items, or for Plus members; only change-of-mind returns of clothing, footwear and bags pay ₹49.",
                             "खराब, डिफ़ेक्टिव या गलत आइटम और Plus सदस्यों के लिए कोई रिवर्स पिकअप फ़ीस नहीं है; सिर्फ़ कपड़े, जूते और बैग की पसंद न आने पर वापसी में ₹49 लगते हैं।",
                             "Damaged, defective ya wrong items aur Plus members ke liye koi reverse pickup fee nahi hai; sirf kapde, footwear aur bags ke change-of-mind return par ₹49 lagte hain.")
            return self.reply(msg, "answer", ["RET-6", "PLS-2"], "return_fee")
        if p is None and nlu.category_from_text(self.t):
            p = {"sku": "", "title": self.P("this item", "यह आइटम", "yeh item"),
                 "return_category": nlu.category_from_text(self.t)}
        if p is None:
            if o and len(o["items"]) > 1:
                return self.which_item()
            return self.reply(self.P(
                "Which item would you like to return? Please share its name or SKU.",
                "आप कौन सा आइटम लौटाना चाहते हैं? कृपया उसका नाम या SKU बताइए।",
                "Aap kaun sa item return karna chahte hain? Kripya uska naam ya SKU bataiye."),
                "clarify", ["RET-2"], "return_window")
        cat = p["return_category"]
        ev = ["RET-2"] + (["products.csv:" + p["sku"]] if p.get("sku") else [])
        item = self.order_item_for(p)
        delivered = B.parse_date(o.get("delivered_on")) if o and item else None
        placed = B.parse_date(o["placed_at"]) if o and item else None
        dts = nlu.extract_dates(self.t)
        rel = nlu.relative_date(self.t)
        if dts:
            if re.search(r"deliver", t) and not delivered:
                delivered = dts[0]
            elif not placed and not re.search(r"deliver", t):
                placed = dts[0]
        if rel and not delivered and re.search(r"deliver|mila|aaya|आया|मिला|received", t):
            delivered = rel
        defect = nlu.mentions_defect(self.t) or (o and o.get("damage_sku") == p["sku"] and nlu.mentions_defect(self.t))
        plus = bool(o and o.get("is_plus")) or nlu.mentions_plus(self.t)
        title = p["title"].split(",")[0]
        Title = title[:1].upper() + title[1:]
        fee_q = bool(re.search(r"pickup|pick-up|collection|collect|charged|\bfee\b|charge|शुल्क|फ़ीस|फीस", t))

        if cat in B.NON_RETURNABLE and not defect:
            return self.reply(self.P(
                f"Sorry, {title} cannot be returned for change of mind because it is non-returnable. If it arrives damaged, expired, wrong or missing, report it within 48 hours with photos.",
                f"क्षमा करें, {title} को पसंद/साइज़ न आने पर वापस नहीं किया जा सकता क्योंकि यह वापसी योग्य नहीं है। अगर यह खराब, गलत या कम आए तो 48 घंटे के अंदर फ़ोटो के साथ बताइए।",
                f"Maaf kijiye, {title} change of mind ke liye return nahi ho sakta kyunki yeh non-returnable hai. Agar yeh damaged, wrong ya missing aaye to 48 ghante ke andar photos ke saath report karein."),
                "answer", ev + ["RET-5", "DWM-1"], "return_eligibility")
        if cat == "beauty_sealed" and (nlu.mentions_opened(self.t)):
            return self.reply(self.P(
                f"Sorry, beauty and personal care items can be returned only if unopened with the seal intact, so {title} can't be returned once opened (sealed items get 7 days from delivery). If it was damaged or wrong, report it within 48 hours.",
                f"क्षमा करें, ब्यूटी और पर्सनल केयर प्रोडक्ट तभी वापस हो सकते हैं जब वे बंद हों और सील सही-सलामत हो, इसलिए खोला हुआ {title} वापस नहीं हो सकता। अगर वह खराब या गलत आया था तो 48 घंटे के अंदर बताइए।",
                f"Maaf kijiye, beauty aur personal care items tabhi return hote hain jab woh unopened ho aur seal intact ho, isliye khola hua {title} return nahi ho sakta. Agar woh damaged ya wrong aaya tha to 48 ghante ke andar report karein."),
                "answer", ev + ["RET-4"], "return_eligibility")

        days, rules = B.return_window(cat, delivered, placed)
        ev += rules
        fee = B.pickup_fee(cat, bool(defect), plus)
        if fee_q and cat == "fashion" or (fee_q and defect):
            if defect:
                msg = self.P(f"No, there is no pickup fee for a damaged, defective or wrong item; you'll get a free replacement or full refund if you report it within 48 hours with photos.",
                             "नहीं, खराब, डिफ़ेक्टिव या गलत आइटम पर कोई पिकअप फ़ीस नहीं लगती; 48 घंटे के अंदर फ़ोटो के साथ बताने पर मुफ़्त रिप्लेसमेंट या पूरा रिफ़ंड मिलेगा।",
                             "Nahi, damaged, defective ya wrong item par koi pickup fee nahi lagti; 48 ghante ke andar photos ke saath batane par free replacement ya full refund milega.")
                return self.reply(msg, "answer", ev + ["RET-6", "DWM-2"], "return_fee")
            if plus:
                msg = self.P("No, the ₹49 reverse pickup fee is waived for ZentroMart Plus members, even for a change-of-mind return.",
                             "नहीं, ZentroMart Plus सदस्यों के लिए ₹49 की रिवर्स पिकअप फ़ीस माफ़ है, पसंद न आने पर वापसी में भी।",
                             "Nahi, ZentroMart Plus members ke liye ₹49 ki reverse pickup fee maaf hai, change-of-mind return par bhi.")
            else:
                msg = self.P("Yes, a ₹49 reverse pickup fee applies to change-of-mind returns of clothing, footwear and bags (waived for Plus members).",
                             "हाँ, कपड़े, जूते और बैग की पसंद न आने पर वापसी में ₹49 पिकअप फ़ीस लगती है (Plus सदस्यों के लिए माफ़)।",
                             "Haan, kapde, footwear aur bags ke change-of-mind return par ₹49 pickup fee lagti hai (Plus members ke liye maaf).")
            return self.reply(msg, "answer", ev + ["RET-6", "PLS-2"], "return_fee")

        ask_window = True
        if delivered:
            last = delivered.toordinal() + days
            deadline = date.fromordinal(last)
            closed = B.TODAY > deadline
            if closed:
                msg = self.P(f"The {days}-day return window for {title} ended on {self.D(deadline)}, so it can no longer be returned for change of mind.",
                             f"{title} की {days} दिन की वापसी अवधि {self.D(deadline)} को खत्म हो चुकी है, इसलिए पसंद न आने पर अब वापसी नहीं हो सकती।",
                             f"{title} ki {days} din ki return window {self.D(deadline)} ko khatam ho chuki hai, isliye ab change-of-mind return nahi ho sakta.")
            else:
                msg = self.P(f"Yes, {title} can be returned within {days} days of delivery, so the last day is {self.D(deadline)}.",
                             f"हाँ, {title} डिलीवरी के {days} दिन के अंदर लौटाया जा सकता है, यानी आख़िरी तारीख़ {self.D(deadline)} है।",
                             f"Haan, {title} delivery ke {days} din ke andar return ho sakta hai, yaani aakhri tareekh {self.D(deadline)} hai.")
            if fee and not closed:
                msg += self.P(f" A ₹{fee} reverse pickup fee applies unless it's damaged or you're a Plus member.",
                              f" पिकअप फ़ीस ₹{fee} लगेगी, जब तक आइटम खराब न हो या आप Plus सदस्य न हों।",
                              f" ₹{fee} ki pickup fee lagegi, jab tak item damaged na ho ya aap Plus member na ho.")
            elif cat == "fashion" and plus and not closed:
                msg += self.P(" As a Plus member you pay no pickup fee.", " Plus सदस्य होने के कारण कोई पिकअप फ़ीस नहीं लगेगी।", " Plus member hone ke kaaran koi pickup fee nahi lagegi.")
            return self.reply(msg, "answer", ev + (["RET-6"] if fee or plus else []), "return_window")

        msg = self.P(f"{Title} can be returned within {days} days of delivery.",
                     f"{title} डिलीवरी के {days} दिन के अंदर वापस किया जा सकता है।",
                     f"{title} delivery ke {days} din ke andar return kiya ja sakta hai.")
        if cat == "fashion" and days == 15:
            msg += self.P(" Orders placed between 1 and 15 October 2026 get 30 days.",
                          " 1 से 15 अक्टूबर 2026 के बीच किए गए ऑर्डर पर 30 दिन मिलते हैं।",
                          " 1 se 15 October 2026 ke beech kiye gaye orders par 30 din milte hain.")
        if cat == "electronics":
            msg += self.P(" Electronics delivered before 1 August 2026 keep the earlier 10-day window.",
                          " 1 अगस्त 2026 से पहले डिलीवर हुए इलेक्ट्रॉनिक्स पर पुराने 10 दिन ही लागू हैं।",
                          " 1 August 2026 se pehle deliver hue electronics par purane 10 din hi lagu hain.")
        return self.reply(msg, "answer", ev, "return_window")

    # ---- refunds
    def h_refund(self):
        t = self.t.lower()
        o = self.s.order
        method = nlu.extract_payment_method(self.t)
        dts = nlu.extract_dates(self.t)
        status_q = bool(re.search(r"where|kahan|कहाँ|status|not (received|arrived)|nahi aaya|नहीं आया|abhi tak|अभी तक", t))
        in_progress = bool(o) and (o["status"] in ("return_requested", "return_picked_up", "return_refund_initiated",
                                                    "returned_refunded", "returned_to_origin", "cancelled")
                                  or o.get("return_refund_initiated_on") or o.get("cancellation_refund_initiated_on"))
        if o and in_progress and (status_q or not (method or dts)) and not re.search(r"cash on delivery|\bcod\b", t):
            return self.refund_status()
        # return + COD
        if method == "cod" or (method is None and o and o["payment_method"] == "cod"):
            return self.reply(self.P(
                "For a cash-on-delivery order you choose the refund: ZentroMart Wallet (instant, within 2 hours) or a bank transfer, which starts once your bank details are verified and takes 5 to 7 business days.",
                "कैश ऑन डिलीवरी ऑर्डर में आप रिफ़ंड चुनते हैं: ZentroMart Wallet (तुरंत, 2 घंटे के अंदर) या बैंक ट्रांसफ़र, जो बैंक डिटेल्स वेरिफ़ाई होने के बाद 5 से 7 कामकाजी दिन लेता है।",
                "Cash on delivery order mein aap refund chunte hain: ZentroMart Wallet (turant, 2 ghante ke andar) ya bank transfer, jo bank details verify hone ke baad 5 se 7 business days leta hai."),
                "answer", ["REF-3", "REF-2", "KB-00.1"], "refund_timeline")
        if method is None:
            if o:
                method = o["payment_method"]
            else:
                return NEED_ORDER
        key = "card" if method == "card" else method
        lo, hi = B.REFUND_SLA[key]
        ev = ["REF-2", "KB-00.1"]
        if hi == 0:
            extra = ""
            if o and o.get("is_plus"):
                extra = self.P(" As a Plus member your return refund is initiated at pickup.", " Plus सदस्य होने से रिटर्न का रिफ़ंड पिकअप के समय शुरू होता है।", " Plus member hone se return ka refund pickup ke samay shuru hota hai.")
            elif o:
                extra = self.P(" For a return it is initiated within 2 business days of pickup, once the item passes quality check.", " रिटर्न में यह पिकअप के 2 कामकाजी दिन के अंदर, क्वालिटी चेक पास होने पर शुरू होता है।", " Return mein yeh pickup ke 2 business days ke andar, quality check pass hone par shuru hota hai.")
            return self.reply(self.P(
                "Refunds to your ZentroMart Wallet are instant, within 2 hours of the refund being initiated." + extra,
                "ZentroMart Wallet में रिफ़ंड तुरंत होता है, रिफ़ंड शुरू होने के 2 घंटे के अंदर।" + extra,
                "ZentroMart Wallet mein refund turant hota hai, refund initiate hone ke 2 ghante ke andar." + extra),
                "answer", ev + (["RET-7", "PLS-2"] if o else []), "refund_timeline")
        name = {"upi": "UPI", "card": "credit or debit card", "netbanking": "net banking"}[key]
        hname = {"upi": "UPI", "card": "क्रेडिट/डेबिट कार्ड", "netbanking": "नेट बैंकिंग"}[key]
        base = self.P(f"Refunds to {name} take {lo} to {hi} business days after the refund is initiated (Monday to Saturday, excluding warehouse holidays).",
                      f"{hname} में रिफ़ंड शुरू होने के बाद {lo} से {hi} कामकाजी दिन लगते हैं (सोमवार से शनिवार, वेयरहाउस छुट्टियों को छोड़कर)।",
                      f"{name} mein refund initiate hone ke baad {lo} se {hi} business days lagte hain (Monday se Saturday, warehouse holidays chhodkar).")
        if o and not dts:
            plus = o.get("is_plus")
            base += self.P(" For a return, the refund is initiated " + ("at pickup" if plus else "within 2 business days of pickup, once the item passes quality check") + ".",
                           " रिटर्न में रिफ़ंड " + ("पिकअप के समय" if plus else "पिकअप के 2 कामकाजी दिन के अंदर, क्वालिटी चेक पास होने पर") + " शुरू होता है।",
                           " Return mein refund " + ("pickup ke samay" if plus else "pickup ke 2 business days ke andar, quality check pass hone par") + " shuru hota hai.")
            ev += ["RET-7"]
        if dts:
            start = dts[0]
            _, latest = B.refund_deadline(key, start)
            base += self.P(f" A refund started on {self.D(start)} should reach you by {self.D(latest)}.",
                           f" {self.D(start)} को शुरू हुआ रिफ़ंड {self.D(latest)} तक आ जाना चाहिए।",
                           f" {self.D(start)} ko shuru hua refund {self.D(latest)} tak aa jaana chahiye.")
            ev.append("KB-00.3")
        return self.reply(base, "answer", ev, "refund_timeline")

    def refund_status(self):
        o, lang = self.s.order, self.l
        st = o["status"]
        ev = ["REF-1", "REF-2"]
        item = self.P("your return", "आपकी वापसी", "aapki return")
        if st == "return_requested":
            return self.reply(self.P(
                "Your return has been requested and is waiting for pickup; the refund is initiated after pickup and the quality check.",
                "आपकी वापसी का अनुरोध दर्ज है और पिकअप का इंतज़ार है; पिकअप और क्वालिटी चेक के बाद रिफ़ंड शुरू होगा।",
                "Aapka return request darj hai aur pickup ka intezaar hai; pickup aur quality check ke baad refund shuru hoga."),
                "answer", ev + ["RET-7"], "refund_status")
        if st == "return_picked_up":
            pick_on = B.parse_date(o["return_pickup_on"])
            by = B.add_business_days(pick_on, 2) if pick_on else None
            plus = o.get("is_plus")
            msg = self.P(
                f"Your item was picked up on {self.D(pick_on)} and is going through the quality check (QC). Once it passes, the refund is initiated within 2 business days of pickup, that is by {self.D(by)}.",
                f"आपका आइटम {self.D(pick_on)} को पिक हो गया है और अभी क्वालिटी जाँच (QC) में है। पास होने पर रिफ़ंड पिकअप के 2 कामकाजी दिन के अंदर, यानी {self.D(by)} तक शुरू हो जाएगा।",
                f"Aapka item {self.D(pick_on)} ko pick ho gaya hai aur abhi quality check (QC) mein hai. Pass hone par refund pickup ke 2 business days ke andar, yaani {self.D(by)} tak shuru ho jayega.")
            return self.reply(msg, "answer", ev + ["RET-7", "PLS-2"], "refund_status")
        if st == "return_refund_initiated":
            m, d0 = o["return_refund_method"], B.parse_date(o["return_refund_initiated_on"])
            over = B.refund_overdue_by(m, d0)
            if over > 3:
                return self.escalate("refund_overdue",
                                     detail=f"return refund initiated {d0} via {m}, {over} business days past its maximum timeline",
                                     intent="refund_status", extra_ev=["REF-6"])
            lo_d, hi_d = B.refund_deadline(m, d0)
            return self.reply(self.P(
                f"Your refund of {rupees(o['return_refund_amount_inr'])} was initiated on {self.D(d0)} and should reach you by {self.D(hi_d)}. Reference number (RRN): {o['return_rrn']}.",
                f"आपका {rupees(o['return_refund_amount_inr'])} का रिफ़ंड {self.D(d0)} को शुरू हुआ और {self.D(hi_d)} तक आ जाना चाहिए। रेफ़रेंस नंबर (RRN): {o['return_rrn']}।",
                f"Aapka {rupees(o['return_refund_amount_inr'])} ka refund {self.D(d0)} ko shuru hua aur {self.D(hi_d)} tak aa jaana chahiye. Reference number (RRN): {o['return_rrn']}."),
                "answer", ev + ["REF-6"], "refund_status")
        if st == "returned_refund_completed" or o.get("return_refund_completed_on") or st == "returned_refunded":
            d1 = B.parse_date(o.get("return_refund_completed_on"))
            return self.reply(self.P(
                f"Your refund was completed{' on ' + self.D(d1) if d1 else ''}. If you can't see it, your bank can trace it with reference number {o.get('return_rrn') or 'on request'}.",
                f"आपका रिफ़ंड{' ' + self.D(d1) + ' को' if d1 else ''} पूरा हो चुका है। न दिखे तो आपका बैंक रेफ़रेंस नंबर {o.get('return_rrn') or ''} से ट्रेस कर सकता है।",
                f"Aapka refund{' ' + self.D(d1) + ' ko' if d1 else ''} complete ho chuka hai. Na dikhe to aapka bank reference number {o.get('return_rrn') or ''} se trace kar sakta hai."),
                "answer", ev + ["REF-6"], "refund_status")
        for kind, label in (("cancellation", "cancellation"), ("rto", "return-to-origin")):
            d0 = B.parse_date(o.get(f"{kind}_refund_initiated_on"))
            if d0:
                m = o[f"{kind}_refund_method"]
                _, hi_d = B.refund_deadline(m, d0)
                return self.reply(self.P(
                    f"Your {label} refund of {rupees(o[kind + '_refund_amount_inr'])} was initiated on {self.D(d0)} and should reach you by {self.D(hi_d)}. Reference number (RRN): {o[kind + '_rrn']}.",
                    f"आपका {rupees(o[kind + '_refund_amount_inr'])} का रिफ़ंड {self.D(d0)} को शुरू हुआ और {self.D(hi_d)} तक आ जाना चाहिए। रेफ़रेंस नंबर (RRN): {o[kind + '_rrn']}।",
                    f"Aapka {rupees(o[kind + '_refund_amount_inr'])} ka refund {self.D(d0)} ko shuru hua aur {self.D(hi_d)} tak aa jaana chahiye. Reference number (RRN): {o[kind + '_rrn']}."),
                    "answer", ev + ["REF-6"], "refund_status")
        if st == "returned_to_origin":
            return self.reply(self.P(
                "Your parcel is returning to our warehouse; for prepaid orders the refund starts once it arrives, usually 7 to 10 calendar days after the last delivery attempt.",
                "आपका पार्सल वेयरहाउस लौट रहा है; प्रीपेड ऑर्डर का रिफ़ंड पार्सल पहुँचने पर शुरू होता है, आमतौर पर आख़िरी डिलीवरी कोशिश के 7 से 10 दिन बाद।",
                "Aapka parcel warehouse wapas aa raha hai; prepaid order ka refund parcel pahunchne par shuru hota hai, aamtaur par last delivery attempt ke 7 se 10 din baad."),
                "answer", ev, "refund_status")
        return self.reply(self.P(
            "I don't see a refund in progress on this order. If you returned or cancelled an item, tell me which one and I'll check.",
            "इस ऑर्डर पर मुझे कोई रिफ़ंड प्रोसेस में नहीं दिख रहा। अगर आपने कोई आइटम लौटाया या कैंसिल किया है तो बताइए कौन सा, मैं देखता हूँ।",
            "Is order par mujhe koi refund process mein nahi dikh raha. Agar aapne koi item return ya cancel kiya hai to bataiye kaun sa, main check karta hoon."),
            "clarify", ev, "refund_status")

    # ---- cancellation
    def h_cancel(self):
        o, t = self.s.order, self.t.lower()
        shipped_hint = bool(re.search(r"already (shipped|on its way|dispatched)|on its way|dispatch|shipped|शिप हो गया|ship ho gaya|रास्ते में|raaste mein", t))
        generic_q = bool(re.search(r"\bcan\b|could|is it possible|\bkya\b|क्या|sakta|सकता|सकती|\?", t))
        if not o and not shipped_hint and not generic_q:
            return NEED_ORDER
        stage = None
        if o:
            stage = ("not_shipped" if o["status"] in ("placed", "packed") else
                     "shipped" if o["status"] in ("shipped", "out_for_delivery", "delivery_attempted") else
                     o["status"])
        elif shipped_hint:
            stage = "shipped"
        if stage == "not_shipped" or (stage is None):
            return self.reply(self.P(
                "Orders that haven't shipped can be cancelled free of charge from My Orders in the app. I can't cancel orders myself, and the refund is initiated immediately once you cancel.",
                "जो ऑर्डर अभी शिप नहीं हुआ है उसे ऐप में My Orders से मुफ़्त कैंसिल किया जा सकता है। मैं खुद ऑर्डर कैंसिल नहीं कर सकता; कैंसिल करते ही रिफ़ंड तुरंत शुरू हो जाता है।",
                "Jo order abhi ship nahi hua use app mein My Orders se free cancel kiya ja sakta hai. Main khud order cancel nahi kar sakta; cancel karte hi refund turant shuru ho jata hai."),
                "answer", ["CAN-1", "FAQH-1"] + (["CAN-2"] if stage is None else []), "cancel_order")
        if stage == "shipped":
            pay = o["payment_method"] if o else None
            prepaid = (pay not in (None, "cod"))
            if o and pay == "cod":
                tail = self.P("Since this is cash on delivery, nothing is charged if you refuse it.",
                              "यह कैश ऑन डिलीवरी है, इसलिए मना करने पर कुछ चार्ज नहीं लगेगा।",
                              "Yeh cash on delivery hai, isliye mana karne par kuch charge nahi lagega.")
            elif prepaid:
                tail = self.P("For a prepaid order the refund starts once the parcel is back at our warehouse, usually 7 to 10 calendar days after the last delivery attempt.",
                              "प्रीपेड ऑर्डर का रिफ़ंड पार्सल वेयरहाउस पहुँचने पर शुरू होता है, आमतौर पर आख़िरी डिलीवरी कोशिश के 7 से 10 दिन बाद।",
                              "Prepaid order ka refund parcel warehouse pahunchne par shuru hota hai, aamtaur par last delivery attempt ke 7 se 10 din baad.")
            else:
                tail = self.P("For prepaid orders the refund follows once the parcel returns to our warehouse (usually 7 to 10 calendar days).",
                              "प्रीपेड ऑर्डर में पार्सल वेयरहाउस लौटने पर रिफ़ंड शुरू होता है (आमतौर पर 7 से 10 दिन)।",
                              "Prepaid order mein parcel warehouse wapas aane par refund shuru hota hai (aamtaur par 7 se 10 din).")
            return self.reply(self.P(
                "Sorry, an order that has already shipped can't be cancelled, but you can refuse the parcel at the door, or accept it and return it within its return window. " + tail,
                "क्षमा करें, शिप हो चुका ऑर्डर कैंसिल नहीं हो सकता, लेकिन आप डिलीवरी के समय पार्सल लेने से मना कर सकते हैं, या ले लें और रिटर्न विंडो में वापस करें। " + tail,
                "Maaf kijiye, ship ho chuka order cancel nahi ho sakta, lekin aap delivery ke samay parcel lene se mana kar sakte hain, ya le lein aur return window mein wapas karein. " + tail),
                "refuse" if o else "answer", ["CAN-2", "REF-1", "FAQH-1"], "cancel_order")
        return self.reply(self.P(
            "This order is already delivered or closed, so it can't be cancelled; if the item is returnable you can return it within its return window.",
            "यह ऑर्डर पहले ही डिलीवर या बंद हो चुका है, इसलिए कैंसिल नहीं हो सकता; आइटम वापसी योग्य हो तो रिटर्न विंडो में लौटा सकते हैं।",
            "Yeh order pehle hi deliver ya close ho chuka hai, isliye cancel nahi ho sakta; item returnable ho to return window mein wapas kar sakte hain."),
            "answer", ["CAN-2", "RET-2"], "cancel_order")

    # ---- damaged / wrong / missing
    def h_damaged(self):
        if re.search(r"pickup|pick-up|collection|collect|\bfee\b|charge|शुल्क|फ़ीस|फीस", self.t.lower()):
            out = self.h_return()
            if out is not None:
                return out
        o, p = self.s.order, self.product()
        t = self.t.lower()
        price = None
        if p:
            item = self.order_item_for(p)
            price = int(item["unit_price_inr"]) * int(item.get("qty") or 1) if item else int(p["price_inr"])
        elif o and o.get("damage_sku"):
            it = next((i for i in o["items"] if i["sku"] == o["damage_sku"]), None)
            price = int(it["unit_price_inr"]) if it else None
        if price and price > HIGH_VALUE:
            return self.escalate("high_value", detail=f"damaged/defective item ({(p or {}).get('title', 'item on order')}) valued at {rupees(price)}", intent="damaged_item")
        late = False
        if o and o.get("delivered_on"):
            late = (B.TODAY - B.parse_date(o["delivered_on"])).days > 2
        if late:
            return self.reply(self.P(
                "Reports after 48 hours are handled as a normal return if the item is still within its return window; otherwise the brand or ZentroMart Assured warranty may apply.",
                "48 घंटे के बाद की शिकायत पर, आइटम रिटर्न विंडो में हो तो सामान्य रिटर्न की तरह प्रोसेस होती है; वरना ब्रांड या ZentroMart Assured वारंटी लागू हो सकती है।",
                "48 ghante ke baad ki shikayat normal return ki tarah process hoti hai agar item return window mein ho; warna brand ya ZentroMart Assured warranty lagu ho sakti hai."),
                "answer", ["DWM-4"], "damaged_item")
        return self.reply(self.P(
            "I'm sorry about that. Please report it within 48 hours of delivery with photos of the item and the packaging, and after verification you'll get a free replacement or a full refund including delivery fees.",
            "यह सुनकर खेद है। कृपया डिलीवरी के 48 घंटे के अंदर आइटम और पैकेट की फ़ोटो के साथ शिकायत करें; जाँच के बाद मुफ़्त रिप्लेसमेंट या डिलीवरी फ़ीस सहित पूरा रिफ़ंड मिलेगा।",
            "Yeh sunkar khed hai. Kripya delivery ke 48 ghante ke andar item aur packet ki photos ke saath report karein; verification ke baad free replacement ya delivery fee sahit full refund milega."),
            "answer", ["DWM-1", "DWM-2", "FAQH-3"], "damaged_item")

    # ---- warranty
    def h_warranty(self):
        p = self.product()
        if p is None:
            return self.reply(self.P(
                "Which product do you mean? Please share its name or SKU and I'll check the warranty.",
                "आप किस प्रोडक्ट की बात कर रहे हैं? उसका नाम या SKU बताइए, मैं वारंटी देख लेता हूँ।",
                "Aap kis product ki baat kar rahe hain? Uska naam ya SKU bataiye, main warranty check kar leta hoon."),
                "clarify", ["WAR-1", "WAR-2"], "warranty")
        title, months = p["title"], int(p["warranty_months"] or 0)
        ev = ["WAR-1", "WAR-2", "products.csv:" + p["sku"]]
        first_party = p["fulfilled_by"] == "ZentroMart"
        if p["assured"] == "True" and first_party:
            return self.reply(self.P(
                f"Yes, {title} carries a 12-month ZentroMart Assured warranty from the delivery date{f', plus the {months}-month brand warranty' if months else ''}. Raise a claim in the app under My Orders and ZentroMart arranges pickup and repair or replacement.",
                f"हाँ, {title} पर डिलीवरी की तारीख़ से 12 महीने की ZentroMart Assured वारंटी है{f' और {months} महीने की ब्रांड वारंटी भी' if months else ''}। क्लेम ऐप में My Orders से करें; पिकअप और रिपेयर/रिप्लेसमेंट ZentroMart करवाएगा।",
                f"Haan, {title} par delivery date se 12 mahine ki ZentroMart Assured warranty hai{f' aur {months} mahine ki brand warranty bhi' if months else ''}. Claim app mein My Orders se karein; pickup aur repair/replacement ZentroMart arrange karega."),
                "answer", ev, "warranty")
        if p["assured"] == "True" and not first_party:
            ev.append("MKS-3")
            return self.reply(self.P(
                f"{title} is sold by {p['fulfilled_by']}, so ZentroMart Assured does not apply even though the listing shows it; only the {months}-month brand warranty applies, claimed with the manufacturer's service centre.",
                f"{title} को {p['fulfilled_by']} बेचता है, इसलिए लिस्टिंग में दिखने के बावजूद ZentroMart Assured लागू नहीं है; सिर्फ़ {months} महीने की ब्रांड वारंटी है, जिसका क्लेम निर्माता के सर्विस सेंटर में होगा।",
                f"{title} ko {p['fulfilled_by']} bechta hai, isliye listing mein dikhne ke bawajood ZentroMart Assured lagu nahi hai; sirf {months} mahine ki brand warranty hai, jiska claim manufacturer ke service centre mein hoga."),
                "answer", ev, "warranty")
        return self.reply(self.P(
            f"{title} has a {months}-month brand warranty (no ZentroMart Assured); claim it with the manufacturer's service centre." if months else f"{title} doesn't have a warranty listed in our catalog.",
            f"{title} पर {months} महीने की ब्रांड वारंटी है (ZentroMart Assured नहीं); क्लेम निर्माता के सर्विस सेंटर में करें।" if months else f"{title} पर हमारे कैटलॉग में कोई वारंटी नहीं दी गई है।",
            f"{title} par {months} mahine ki brand warranty hai (ZentroMart Assured nahi); claim manufacturer ke service centre mein karein." if months else f"{title} par hamare catalog mein koi warranty listed nahi hai."),
            "answer", ev, "warranty")

    # ---- late delivery / compensation
    def h_late(self):
        o, t = self.s.order, self.t.lower()
        express_q = bool(re.search(r"express|एक्सप्रेस", t))
        voucher_q = bool(re.search(r"voucher|compensat|rupaye|रुपये|₹|\bcredit\b", t)) and not express_q
        if o:
            if o["delivery_type"] == "express" or express_q:
                if o.get("express_late_credit_on"):
                    d = B.parse_date(o["express_late_credit_on"])
                    return self.reply(self.P(
                        f"Yes, the ₹{o['express_late_credit_inr']} express fee was credited to your ZentroMart Wallet on {self.D(d)}.",
                        f"हाँ, ₹{o['express_late_credit_inr']} की एक्सप्रेस फ़ीस {self.D(d)} को आपके ZentroMart Wallet में क्रेडिट हो गई थी।",
                        f"Haan, ₹{o['express_late_credit_inr']} ki express fee {self.D(d)} ko aapke ZentroMart Wallet mein credit ho gayi thi."),
                        "answer", ["DEL-4"], "late_delivery")
                if o.get("is_plus"):
                    return self.reply(self.P("Plus members pay no express fee, so there is no late-delivery credit; I'm sorry for the delay.",
                                             "Plus सदस्य एक्सप्रेस फ़ीस नहीं देते, इसलिए देरी पर कोई क्रेडिट नहीं मिलता; देरी के लिए खेद है।",
                                             "Plus members express fee nahi dete, isliye late delivery par koi credit nahi milta; deri ke liye khed hai."),
                                      "answer", ["DEL-4", "PLS-2"], "late_delivery")
            # standard delivery
            return self.reply(self.P(
                "I'm sorry your order was late. There's no compensation for late Standard deliveries, so I can't offer a voucher.",
                "आपका ऑर्डर देर से आया, इसके लिए खेद है। स्टैंडर्ड डिलीवरी में देरी पर कोई मुआवज़ा नहीं मिलता, इसलिए मैं वाउचर नहीं दे सकता।",
                "Aapka order late aaya, uske liye maaf kijiye. Standard delivery mein deri par koi compensation nahi milta, isliye main voucher nahi de sakta."),
                "refuse", ["DEL-4", "ESC-4"], "late_delivery")
        if express_q:
            return self.reply(self.P(
                "If an Express order arrives after its promised date, the ₹79 express fee is credited to your ZentroMart Wallet automatically within 48 hours (Plus members pay no express fee, so no credit). Late Standard deliveries get no compensation.",
                "अगर एक्सप्रेस ऑर्डर वादे की तारीख़ के बाद आए तो ₹79 एक्सप्रेस फ़ीस 48 घंटे के अंदर अपने आप आपके ZentroMart Wallet में क्रेडिट हो जाती है (Plus सदस्य फ़ीस नहीं देते, इसलिए क्रेडिट नहीं)। स्टैंडर्ड डिलीवरी की देरी पर कोई मुआवज़ा नहीं।",
                "Agar Express order promised date ke baad aaye to ₹79 express fee 48 ghante ke andar apne aap aapke ZentroMart Wallet mein credit ho jati hai (Plus members fee nahi dete, isliye credit nahi). Late Standard delivery par koi compensation nahi."),
                "answer", ["DEL-4"], "late_delivery")
        if voucher_q or True:
            return self.reply(self.P(
                "I'm sorry for the delay. There's no compensation for late Standard deliveries, and I can't offer vouchers or exceptions; Express orders get the ₹79 fee back as Wallet credit.",
                "देरी के लिए खेद है। स्टैंडर्ड डिलीवरी की देरी पर कोई मुआवज़ा नहीं है और मैं वाउचर या अपवाद नहीं दे सकता; एक्सप्रेस ऑर्डर में ₹79 फ़ीस Wallet क्रेडिट में लौटती है।",
                "Deri ke liye maaf kijiye. Standard delivery ki deri par koi compensation nahi hai aur main voucher ya exception nahi de sakta; Express orders mein ₹79 fee Wallet credit mein wapas milti hai."),
                "refuse", ["DEL-4", "ESC-4"], "late_delivery")

    # ---- delivery attempts
    def h_attempts(self):
        return self.reply(self.P(
            "The courier makes up to 3 delivery attempts, and after a failed attempt you get an SMS to reschedule. If the third attempt fails, the parcel goes back to our warehouse and a prepaid refund follows.",
            "कूरियर 3 बार तक डिलीवरी की कोशिश करता है, और असफल कोशिश के बाद आपको रीशेड्यूल करने का SMS मिलता है। तीसरी कोशिश भी असफल हो तो पार्सल वेयरहाउस लौट जाता है और प्रीपेड ऑर्डर का रिफ़ंड मिलता है।",
            "Courier 3 baar tak delivery ki koshish karta hai, aur fail hone ke baad aapko reschedule karne ka SMS milta hai. Teesri koshish fail ho to parcel warehouse wapas chala jata hai aur prepaid refund milta hai."),
            "answer", ["DEL-3", "REF-1"], "delivery_attempts")

    # ---- tracking
    def h_status(self):
        o, t = self.s.order, self.t.lower()
        if not o:
            return NEED_ORDER
        st = o["status"]
        prom = B.parse_date(o["promised_date"])
        ship, dele = B.parse_date(o.get("shipped_on")), B.parse_date(o.get("delivered_on"))
        ask_ship = bool(re.search(r"ship(ped)?\b.*(kab|when|date|hua)|kab.*ship|when (was|did).*ship|शिप.*कब|कब.*शिप|ship hua", t))
        if ask_ship and ship:
            return self.reply(self.P(f"Your order shipped on {self.D(ship)}.", f"आपका ऑर्डर {self.D(ship)} को शिप हुआ था।",
                                     f"Aapka order {self.D(ship)} ko ship hua tha."), "answer", ["TRK-1"], "order_status")
        if st == "delivered" and dele:
            return self.reply(self.P(f"Your order was delivered on {self.D(dele)}.", f"आपका ऑर्डर {self.D(dele)} को डिलीवर हो चुका है।",
                                     f"Aapka order {self.D(dele)} ko deliver ho chuka hai."), "answer", ["TRK-1", "TRK-2"], "order_status")
        if st in ("placed", "packed"):
            return self.reply(self.P(f"Your order is {st} and hasn't shipped yet; the promised delivery date is {self.D(prom)}.",
                                     f"आपका ऑर्डर अभी शिप नहीं हुआ है ({st}); वादा की गई डिलीवरी की तारीख़ {self.D(prom)} है।",
                                     f"Aapka order abhi ship nahi hua hai ({st}); promised delivery date {self.D(prom)} hai."), "answer", ["TRK-1", "DEL-1"], "order_status")
        if st in ("shipped", "out_for_delivery", "delivery_attempted"):
            label = {"shipped": "shipped and on the way", "out_for_delivery": "out for delivery today",
                     "delivery_attempted": f"marked delivery attempted ({o['delivery_attempts']} attempt(s) so far)"}[st]
            late = prom < B.TODAY
            return self.reply(self.P(
                f"Your order is {label}{' (shipped on ' + self.D(ship) + ')' if ship else ''}, and the promised delivery date is {self.D(prom)}."
                + (" It has passed that date, and I'm sorry for the delay." if late else ""),
                f"आपका ऑर्डर {'शिप हो चुका है और रास्ते में है' if st == 'shipped' else 'आज डिलीवरी के लिए निकला है' if st == 'out_for_delivery' else 'डिलीवरी असफल रही (अटेम्प्ट)'}, और वादा की गई डिलीवरी की तारीख़ {self.D(prom)} है।" + (" यह तारीख़ निकल चुकी है, देरी के लिए खेद है।" if late else ""),
                f"Aapka order {'ship ho chuka hai aur raaste mein hai' if st == 'shipped' else 'aaj delivery ke liye nikla hai' if st == 'out_for_delivery' else 'delivery attempt fail hui'}, aur promised delivery date {self.D(prom)} hai." + (" Yeh date nikal chuki hai, deri ke liye maaf kijiye." if late else "")),
                "answer", ["TRK-1", "DEL-1", "DEL-3"], "order_status")
        return self.reply(self.P(f"Your order's current status is {st.replace('_', ' ')}.", f"आपके ऑर्डर का मौजूदा स्टेटस: {st.replace('_', ' ')}।",
                                 f"Aapke order ka current status: {st.replace('_', ' ')}."), "answer", ["TRK-1"], "order_status")

    # ---- coupons / plus / delivery
    def h_coupon(self):
        t = self.t.lower()
        if re.search(r"expired|नहीं लग|not appl|failed|error|didn.?t work|nahi lag", t):
            return self.reply(self.P(
                "A coupon that was valid but failed to apply because of a technical error can be re-issued once by support, if the order was placed within its validity; coupons that had already expired can't be re-issued.",
                "जो कूपन वैध था पर तकनीकी गड़बड़ी से नहीं लगा, उसे ऑर्डर वैधता के अंदर होने पर सपोर्ट एक बार दोबारा जारी कर सकता है; पहले से एक्सपायर कूपन दोबारा जारी नहीं होते।",
                "Jo coupon valid tha par technical error se nahi laga, use order validity ke andar hone par support ek baar dobara issue kar sakta hai; pehle se expired coupon dobara issue nahi hote."),
                "answer", ["CPN-2"], "coupon")
        return self.reply(self.P(
            "Only one coupon can be used per order; coupons can't be combined, added after the order is placed, or exchanged for cash.",
            "एक ऑर्डर पर सिर्फ़ एक कूपन लग सकता है; कूपन को मिलाया नहीं जा सकता, ऑर्डर के बाद जोड़ा नहीं जा सकता और नकद में नहीं बदला जा सकता।",
            "Ek order par sirf ek (1) coupon lag sakta hai; coupons ko combine nahi kiya ja sakta, order ke baad add nahi kiya ja sakta aur cash mein nahi badla ja sakta."),
            "answer", ["CPN-1"], "coupon")

    def h_plus(self):
        if not re.search(r"plus|प्लस", self.t.lower()):
            return None
        return self.reply(self.P(
            "ZentroMart Plus costs ₹999 a year and gives free Express delivery where available, no reverse pickup fee on any return, and refunds on returns initiated at pickup. It does not change return windows or refund timelines after initiation.",
            "ZentroMart Plus ₹999 प्रति वर्ष है और इसमें उपलब्ध जगहों पर मुफ़्त एक्सप्रेस डिलीवरी, किसी भी रिटर्न पर पिकअप फ़ीस नहीं, और पिकअप के समय रिफ़ंड शुरू होता है। इससे रिटर्न विंडो या रिफ़ंड की अवधि नहीं बदलती।",
            "ZentroMart Plus ₹999 prati varsh hai aur ismein available jagah par free Express delivery, kisi bhi return par pickup fee nahi, aur pickup ke samay refund shuru hota hai. Isse return window ya refund ki avadhi nahi badalti."),
            "answer", ["PLS-1", "PLS-2", "MKT-1"], "plus")

    def h_area(self):
        t = self.t
        m = re.search(r"(?<!\d)(\d{6})(?!\d)", nlu.ascii_digits(t))
        if m:
            pin = B.PINCODES.get(m.group(1))
            if not pin:
                return self.reply(self.P(f"Pincode {m.group(1)} isn't in our serviceable list, so we can't deliver there.",
                                         f"पिनकोड {m.group(1)} हमारी सेवा-क्षेत्र सूची में नहीं है, इसलिए वहाँ डिलीवरी नहीं हो सकती।",
                                         f"Pincode {m.group(1)} hamari serviceable list mein nahi hai, isliye wahan delivery nahi ho sakti."), "answer", ["DEL-5"], "delivery_area")
            yes = lambda v: v == "True"
            return self.reply(self.P(
                f"Pincode {pin['pincode']} ({pin['city']}) is {'serviceable' if yes(pin['serviceable']) else 'not serviceable'}; cash on delivery is {'available' if yes(pin['cod_available']) else 'not available'} and Express is {'available' if yes(pin['express_available']) else 'not available'} there.",
                f"पिनकोड {pin['pincode']} ({pin['city']}) {'सेवा-क्षेत्र में है' if yes(pin['serviceable']) else 'सेवा-क्षेत्र में नहीं है'}; वहाँ कैश ऑन डिलीवरी {'उपलब्ध' if yes(pin['cod_available']) else 'उपलब्ध नहीं'} और एक्सप्रेस {'उपलब्ध' if yes(pin['express_available']) else 'उपलब्ध नहीं'} है।",
                f"Pincode {pin['pincode']} ({pin['city']}) {'serviceable hai' if yes(pin['serviceable']) else 'serviceable nahi hai'}; wahan cash on delivery {'available' if yes(pin['cod_available']) else 'available nahi'} aur Express {'available' if yes(pin['express_available']) else 'available nahi'} hai."),
                "answer", ["DEL-5"], "delivery_area")
        return self.reply(self.P(
            "ZentroMart delivers only within India, to pincodes listed as serviceable; cash on delivery and Express are available only on selected pincodes. Share a pincode and I'll check it.",
            "ZentroMart सिर्फ़ भारत के अंदर, सेवा-क्षेत्र वाले पिनकोड पर डिलीवर करता है; कैश ऑन डिलीवरी और एक्सप्रेस चुनिंदा पिनकोड पर ही मिलते हैं। पिनकोड बताइए, मैं देख लेता हूँ।",
            "ZentroMart sirf India ke andar, serviceable pincodes par deliver karta hai; cash on delivery aur Express chuninda pincodes par hi milte hain. Pincode bataiye, main check kar leta hoon."),
            "answer", ["DEL-5", "FAQ-5"], "delivery_area")

    def h_fees(self):
        return self.reply(self.P(
            "Standard delivery is free on orders of ₹499 or more, otherwise ₹40; Express costs ₹79 (free for Plus members), and cash on delivery adds a ₹25 fee on orders up to ₹10,000.",
            "₹499 या उससे ज़्यादा के ऑर्डर पर स्टैंडर्ड डिलीवरी मुफ़्त है, वरना ₹40; एक्सप्रेस ₹79 है (Plus सदस्यों के लिए मुफ़्त), और ₹10,000 तक के ऑर्डर पर कैश ऑन डिलीवरी में ₹25 फ़ीस लगती है।",
            "₹499 ya usse zyada ke order par Standard delivery free hai, warna ₹40; Express ₹79 hai (Plus members ke liye free), aur ₹10,000 tak ke order par cash on delivery mein ₹25 fee lagti hai."),
            "answer", ["DEL-2", "PAY-2"], "delivery_fee")

    def h_troubleshoot(self):
        t = self.t.lower()
        sec = ("TRB-4" if re.search(r"speaker", t) else "TRB-1" if re.search(r"earbud|ear bud|airpod|buds", t) else
               "TRB-2" if re.search(r"watch|sync", t) else "TRB-3" if re.search(r"charg", t) else None)
        if sec is None:
            return None
        txt = {
            "TRB-4": ("Charge the speaker for at least 30 minutes with its original cable, then hold the power button for 10 seconds to force a restart.",
                      "स्पीकर को उसकी ओरिजिनल केबल से कम से कम 30 मिनट चार्ज करें, फिर पावर बटन 10 सेकंड दबाकर रखें ताकि वह रीस्टार्ट हो जाए।",
                      "Speaker ko original cable se kam se kam 30 minute charge karein, phir power button 10 second dabakar rakhein taaki woh restart ho jaye."),
            "TRB-1": ("Put both earbuds in the case for 10 seconds, forget them in your phone's Bluetooth list, then hold the case button for 15 seconds until the light flashes and pair again.",
                      "दोनों ईयरबड्स को 10 सेकंड केस में रखें, फ़ोन की ब्लूटूथ लिस्ट से उन्हें हटाएँ, फिर केस का बटन 15 सेकंड दबाएँ जब तक लाइट न चमके और दोबारा पेयर करें।",
                      "Dono earbuds ko 10 second case mein rakhein, phone ki Bluetooth list se hata dein, phir case ka button 15 second dabayein jab tak light na chamke aur dobara pair karein."),
            "TRB-2": ("Give the companion app Bluetooth and location permission, restart the watch and phone, then unpair and pair the watch again.",
                      "कम्पैनियन ऐप को ब्लूटूथ और लोकेशन की अनुमति दें, घड़ी और फ़ोन रीस्टार्ट करें, फिर घड़ी को अनपेयर करके दोबारा पेयर करें।",
                      "Companion app ko Bluetooth aur location permission dein, watch aur phone restart karein, phir watch ko unpair karke dobara pair karein."),
            "TRB-3": ("Use the original cable and adapter, gently clean the charging port with a dry soft brush, and try charging in airplane mode.",
                      "ओरिजिनल केबल और एडॉप्टर इस्तेमाल करें, चार्जिंग पोर्ट को सूखे नरम ब्रश से साफ़ करें, और एयरप्लेन मोड में चार्ज करके देखें।",
                      "Original cable aur adapter use karein, charging port ko dry soft brush se saaf karein, aur airplane mode mein charge karke dekhein."),
        }[sec]
        tail = self.P(" If it still doesn't work, you can return it within its window or claim warranty.",
                      " फिर भी न चले तो रिटर्न विंडो में लौटाएँ या वारंटी क्लेम करें।",
                      " Phir bhi na chale to return window mein wapas karein ya warranty claim karein.")
        return self.reply(self.P(*txt) if False else {"en": txt[0], "hi": txt[1], "hi-en": txt[2]}[self.l] + tail,
                          "answer", [sec], "troubleshooting")

    # ---- everything else: retrieval-grounded answer or honest "I don't know"
    def h_general(self):
        t = self.t
        if self.intents == [] and re.fullmatch(r"(?is)\W*(hi|hello|hey|namaste|नमस्ते|hii+)\W*", t):
            return self.reply(self.P("How can I help you today?", "आज मैं आपकी कैसे मदद कर सकता हूँ?", "Aaj main aapki kaise madad kar sakta hoon?"),
                              "clarify", ["STY-1"], "greeting")
        if re.search(r"order|ऑर्डर", t.lower()) and re.search(r"help|madad|मदद|support|sahayata|सहायता", t.lower()) and not self.s.order:
            self.s.pending = None
            self.s.state = "await_order"
            return self.reply(self.P("Happy to help. Could you please share your order ID?", "ज़रूर, मदद करूँगा। कृपया अपना ऑर्डर ID बताइए।",
                                     "Zaroor, madad karunga. Kripya apna order ID bataiye."), "clarify", ["STY-1", "IDV-2"], "need_order")
        if re.search(r"order|ऑर्डर", t.lower()) and re.search(r"help|madad|मदद", t.lower()) and self.s.order:
            return self.reply(self.P("Sure, what would you like to know about this order?", "ज़रूर, इस ऑर्डर के बारे में आप क्या जानना चाहते हैं?",
                                     "Zaroor, is order ke baare mein aap kya jaanna chahte hain?"), "clarify", ["IDV-2"], "followup")
        old = bool(re.search(r"before 1 august|old (policy|rule)|earlier policy|v3|version 3", t.lower()))
        hits = self.a.retriever.search(t, k=3, include_superseded=old)
        cov = self.a.retriever.coverage(t)
        topic = self.a.retriever.topic_recognised(t)
        if hits and (cov >= 0.6 or (topic and cov >= 0.3)):
            best = hits[0][0]
            body = re.sub(r"\s+", " ", re.sub(r"[|*`#>-]+", " ", best.text)).strip()
            sents = re.split(r"(?<=[.!?])\s+", body)
            ans = " ".join(sents[:2])[:360]
            return self.reply(ans, "answer", [h[0].section_id for h in hits], "policy_qa")
        return self.reply(self.P(
            "Sorry, I don't have information about that in our policies. I can connect you to a human colleague who can help; would you like that?",
            "क्षमा करें, हमारी नीतियों में इसकी जानकारी मेरे पास नहीं है। क्या आप चाहेंगे कि मैं आपको हमारे सहयोगी से जोड़ दूँ जो इसमें मदद कर सकें?",
            "Maaf kijiye, hamari policies mein iski jaankari mere paas nahi hai. Kya aap chahenge ki main aapko hamare colleague se connect kar doon jo ismein madad kar sakein?"),
            "answer", [], "unsupported")
