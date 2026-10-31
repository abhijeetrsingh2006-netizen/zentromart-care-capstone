"""Strict evaluation of the ZentroMart assistant through the same entry point the web app uses
(`app.response_for`).  Nothing here looks up reference answers.

Scoring follows DATA_DICTIONARY.md:
  * every ' ; ' fact in must_include is required; ' / ' separates equivalent alternatives;
    equivalent date formats are accepted; numbers must match as whole numbers
  * any must_not_include phrase is a failure
  * single questions pass if the decision is in acceptable_decisions
  * conversation turns need the exact expected decision AND the expected tool calls with the
    same arguments (order ID, verification value); create_ticket must be called and a
    ticket summary must not contain the customer's verification digits
  * evidence: fraction of gold KB section IDs cited, plus raw retrieval recall@5

    python evaluate.py            # summary
    python evaluate.py -v         # also print every failing reply
Exit code is 1 if anything fails, so it can be used as a gate.
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict
from datetime import date

import app
from support_agent import i18n

TICKETS: list[dict] = []


def _stub_ticket(summary, order_id="", reason="support_handover", intent="support"):
    TICKETS.append({"summary": summary, "order_id": order_id, "reason": reason})
    return {"created": True, "ticket_id": f"TKT-EVAL-{len(TICKETS):03d}"}


app.make_ticket = _stub_ticket          # keep evaluation tickets out of tools/tickets_created.csv


def split(spec, sep):
    return [x.strip() for x in spec.split(sep) if x.strip()]


def has(text: str, alt: str) -> bool:
    alt = alt.strip().lower()
    forms = [alt]
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", alt)
    if m:
        d = date(int(m[1]), int(m[2]), int(m[3]))
        forms = [alt, f"{d.day} {i18n.EN_MONTHS[d.month - 1].lower()} {d.year}", f"{d.day} {i18n.HI_MONTHS[d.month - 1]} {d.year}"]
    for f in forms:
        if re.fullmatch(r"\d+", f):
            if re.search(rf"(?<!\d){f}(?!\d)", text):
                return True
        elif f in text:
            return True
    return False


def include_ok(reply: str, spec: str) -> bool:
    t = reply.lower()
    return all(any(has(t, a) for a in fact.split(" / ")) for fact in split(spec, " ; "))


def exclude_ok(reply: str, spec: str) -> bool:
    t = reply.lower()
    return not any(p.lower() in t for p in split(spec, " ; "))


def calls_ok(expected: str, got: list[str]) -> bool:
    norm = lambda x: re.sub(r"\s+", "", x)
    for e in re.findall(r"\w+\([^)]*\)", expected):
        if norm(e) == "create_ticket()":
            if not any(g.startswith("create_ticket") for g in got):
                return False
        elif not any(norm(g) == norm(e) for g in got):
            return False
    return True


def pct(xs):
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "  n/a"


def questions(voice: bool, verbose: bool):
    res = []
    for r in app.QUESTIONS:
        msg = r["asr_text"] if voice else r["question"]
        TICKETS.clear()
        answer, sources, meta = app.response_for(msg, {}, voice=voice)
        acceptable = split(r["acceptable_decisions"] or r["expected_decision"], ";")
        gold = [g for g in split(r["gold_evidence"], ";") if not g.startswith("products.csv") and not g.startswith("KB-00")]
        top5 = [c.section_id for c, _ in app.AGENT.retriever.search(r["question"], 5, include_superseded=True)]
        cited = {s["id"] for s in sources}
        x = {"id": r["id"], "intent": r["intent"], "reply": answer, "got": meta["decision"], "want": acceptable,
             "decision": meta["decision"] in acceptable, "include": include_ok(answer, r["must_include"]),
             "exclude": exclude_ok(answer, r["must_not_include"]),
             "cited": sum(g in cited for g in gold) / len(gold) if gold else 1.0,
             "retr5": sum(g in top5 for g in gold) / len(gold) if gold else 1.0}
        if r["expected_decision"] == "escalate":                       # a real ticket must have been created
            x["decision"] = x["decision"] and bool(TICKETS) or meta["decision"] in acceptable and meta["decision"] != "escalate"
        x["pass"] = x["decision"] and x["include"] and x["exclude"]
        res.append(x)
    label = "ASR/voice" if voice else "text"
    print(f"\n== single questions ({label}): {sum(x['pass'] for x in res)}/{len(res)} pass | decision {pct(x['decision'] for x in res)}"
          f" | must_include {pct(x['include'] for x in res)} | must_not {pct(x['exclude'] for x in res)}"
          f" | cited-evidence recall {pct(x['cited'] for x in res)} | raw retrieval recall@5 {pct(x['retr5'] for x in res)}")
    by = defaultdict(list)
    for x in res:
        by[x["intent"]].append(x["pass"])
    print("   " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items())))
    for x in res:
        if verbose and not x["pass"]:
            print(f"   FAIL {x['id']} got={x['got']} want={x['want']} inc={x['include']} exc={x['exclude']}\n        {x['reply']}")
    return res


def conversations(verbose: bool):
    groups = defaultdict(list)
    for r in app.read_csv(app.PRACTICE / "conversations.csv"):
        groups[r["conversation_id"]].append(r)
    turns, whole = [], {}
    for cid, rows in groups.items():
        state, ok_all = {}, True
        proofs = set(re.findall(r"verification_value=(\d+)", " ".join(r["expected_tool_calls"] for r in rows)))
        for r in sorted(rows, key=lambda x: int(x["turn"])):
            TICKETS.clear()
            answer, sources, meta = app.response_for(r["customer_message"], state)
            t = {"id": f"{cid}.{r['turn']}", "reply": answer, "got": meta["decision"], "want": r["expected_decision"],
                 "decision": meta["decision"] == r["expected_decision"],
                 "include": include_ok(answer, r["must_include"]), "exclude": exclude_ok(answer, r["must_not_include"]),
                 "tools": calls_ok(r["expected_tool_calls"], meta["tool_calls"]) if r["expected_tool_calls"] else True,
                 "calls": meta["tool_calls"], "want_calls": r["expected_tool_calls"]}
            if TICKETS:                                                 # handover summaries must be useful and safe
                s = TICKETS[-1]["summary"]
                t["ticket"] = len(s) >= 20 and not any(re.search(rf"(?<!\d){p}(?!\d)", s) for p in proofs)
            else:
                t["ticket"] = True
            gold = [g for g in split(r["gold_evidence"], ";") if not g.startswith("products.csv")]
            cited = {s["id"] for s in sources}
            t["cited"] = sum(g in cited for g in gold) / len(gold) if gold else 1.0
            t["pass"] = all(t[k] for k in ("decision", "include", "exclude", "tools", "ticket"))
            ok_all &= t["pass"]
            turns.append(t)
        whole[cid] = ok_all
    print(f"\n== conversations: {sum(whole.values())}/{len(whole)} fully correct | {sum(t['pass'] for t in turns)}/{len(turns)} turns | "
          f"decision {pct(t['decision'] for t in turns)} | facts {pct(t['include'] and t['exclude'] for t in turns)} | "
          f"tool calls+args {pct(t['tools'] for t in turns)} | safe tickets {pct(t['ticket'] for t in turns)} | "
          f"cited-evidence recall {pct(t['cited'] for t in turns)}")
    for t in turns:
        if verbose and not t["pass"]:
            print(f"   FAIL {t['id']} got={t['got']} want={t['want']} calls={t['calls']} want_calls={t['want_calls']}\n        {t['reply']}")
    return turns, whole


def order_scan():
    """Robustness: verify every order and ask three things; nothing may crash and verified
    details must never be shown before verification."""
    crashes = leaks = 0
    phone = {c["customer_id"]: c.get("phone", "") for c in app.CUSTOMERS.values()}
    for oid, o in app.ORDERS.items():
        st = {}
        try:
            first = app.response_for(f"Where is my order {oid}?", st)[0]
            leaks += bool(re.search(r"promised|delivered on|shipped", first.lower()))
            proof = phone[o["customer_id"]][-4:] or o["delivery_pincode"]
            app.response_for(proof, st)
            for q in ("Where is my order?", "Mera refund kahan hai?", "Can I cancel this order?"):
                app.response_for(q, st)
        except Exception:
            crashes += 1
    print(f"\n== order scan: {len(app.ORDERS)} orders | crashes {crashes} | pre-verification leaks {leaks}")
    return crashes + leaks == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    q1, q2 = questions(False, a.verbose), questions(True, a.verbose)
    turns, whole = conversations(a.verbose)
    scan_ok = order_scan()
    ok = all(x["pass"] for x in q1) and all(x["pass"] for x in q2) and all(whole.values()) and scan_ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
