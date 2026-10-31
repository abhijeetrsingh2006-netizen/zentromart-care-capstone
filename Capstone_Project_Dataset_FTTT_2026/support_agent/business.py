"""Deterministic business rules: dates, business days, return windows, refunds.

Everything here is computed from the dataset reference date (20 Oct 2026),
never from the computer clock (README: "Today's date for this dataset").
"""
from __future__ import annotations
import csv
import os
from datetime import date, timedelta

TODAY = date(2026, 10, 20)
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _rows(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


HOLIDAYS = {date.fromisoformat(r["date"]) for r in _rows("holidays.csv")}
PRODUCTS = {r["sku"]: r for r in _rows("products.csv")}
PINCODES = {r["pincode"]: r for r in _rows("pincodes.csv")}


def parse_date(v):
    if not v:
        return None
    return date.fromisoformat(str(v)[:10])


# ---------------------------------------------------------------- business days
def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in HOLIDAYS      # Mon-Sat, minus holidays


def add_business_days(start: date, n: int) -> date:
    """Date that is `n` business days after `start` (start itself is day 0)."""
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d


def business_days_between(a: date, b: date) -> int:
    """Business days in the interval (a, b]; 0 if b <= a."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if is_business_day(d):
            n += 1
    return n


# ------------------------------------------------------------------- refunds
# REF-2: (min, max) business days after initiation; (0, 0) = instant (<2 hours)
REFUND_SLA = {
    "upi": (1, 3), "card": (5, 7), "netbanking": (3, 5), "wallet": (0, 0),
    "cod_bank": (5, 7), "cod_wallet": (0, 0),
}
INSTANT = {"wallet", "cod_wallet"}


def refund_deadline(method: str, initiated: date):
    lo, hi = REFUND_SLA[method]
    if hi == 0:
        return initiated, initiated
    return add_business_days(initiated, lo), add_business_days(initiated, hi)


def refund_overdue_by(method: str, initiated: date, today: date = TODAY) -> int:
    """Business days past the *maximum* timeline (ESC-1 #5 triggers at > 3)."""
    _, latest = refund_deadline(method, initiated)
    return business_days_between(latest, today)


# ------------------------------------------------------------------- returns
RETURN_DAYS = {"electronics": 7, "fashion": 15, "home_kitchen": 10,
               "beauty_sealed": 7, "books_media": 10}
NON_RETURNABLE = {"grocery", "innerwear", "digital"}
ELECTRONICS_CUTOFF = date(2026, 8, 1)
FESTIVE_START, FESTIVE_END = date(2026, 10, 1), date(2026, 10, 15)


def return_window(category: str, delivered_on: date | None = None,
                  placed_on: date | None = None):
    """-> (days | None, rule_ids). None means not returnable for change of mind."""
    if category in NON_RETURNABLE:
        return None, ["RET-2", "RET-5"]
    days, rules = RETURN_DAYS.get(category, 7), ["RET-2"]
    if category == "electronics" and delivered_on and delivered_on < ELECTRONICS_CUTOFF:
        return 10, ["RET-3", "RETv3-2"]               # grandfathered v3 window
    if category == "electronics":
        rules.append("RET-3")
    if category == "fashion" and placed_on and FESTIVE_START <= placed_on <= FESTIVE_END:
        return 30, ["RET-8"]
    return days, rules


def pickup_fee(category: str, reason_defective: bool, is_plus: bool) -> int:
    """RET-6: Rs 49 only for change-of-mind fashion, waived for Plus/defective."""
    if category == "fashion" and not reason_defective and not is_plus:
        return 49
    return 0


def money(n) -> str:
    return f"₹{int(n):,}"


def fmt_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"
