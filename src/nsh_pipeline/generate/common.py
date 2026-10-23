"""Shared helpers for the synthetic data generator."""
from datetime import date, datetime, timedelta

import numpy as np
from faker import Faker


def make_generators(seed: int):
    """Seeded numpy RNG + Faker; the same seed always produces the same data."""
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return rng, fake


def oid(prefix: str, n: int) -> str:
    """Fake 24-hex ObjectId string, e.g. oid("a0", 1) -> "a0000...001"."""
    return f"{prefix}{n:0{24 - len(prefix)}d}"


def iso(d: date) -> str:
    """Mongo Date field as an ISO-8601 string."""
    return f"{d.isoformat()}T00:00:00Z"


def iso_dt(dt: datetime) -> str:
    """Mongo Date field with a time of day, as an ISO-8601 UTC string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def mdy(d: date) -> str:
    """M/D/YYYY string without zero padding, as NSH stores birthdates and card dates."""
    return f"{d.month}/{d.day}/{d.year}"


def money_str(amount: float) -> str:
    """Dollar amount as NSH stores it in string fields, e.g. 1250.75 -> "1,250.75"."""
    return f"{amount:,.2f}"


def random_date(rng, start: date, end: date) -> date:
    """Uniform random date between `start` and `end` inclusive."""
    return start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))