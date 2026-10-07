"""A stand-in for a restaurant reservation API (in production: the booking provider's API, e.g. OpenTable).

Idempotent: asking twice for the same booking (a retried request, a double-click) returns the same reference instead
of booking two tables. Real booking APIs take an "idempotency key" for exactly this.
"""
from __future__ import annotations


class BookingService:
    def __init__(self):
        self.bookings = {}                       # idempotency key -> booking

    def book(self, session_id: str, business: str, date: str, time: str, party_size: int) -> str:
        key = (session_id, business, date, time, party_size)
        if key not in self.bookings:
            reference = f"RS-{len(self.bookings) + 1:04d}"
            self.bookings[key] = {"reference": reference, "business": business, "date": date, "time": time,
                                  "party_size": party_size}
        return self.bookings[key]["reference"]
