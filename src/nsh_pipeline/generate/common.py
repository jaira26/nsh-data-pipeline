"""Shared helpers for the synthetic data generator."""
from datetime import date, timedelta

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


def mdy(d: date) -> str:
    """M/D/YYYY string without zero padding, as NSH stores birthdates and card dates."""
    return f"{d.month}/{d.day}/{d.year}"


def random_date(rng, start: date, end: date) -> date:
    """Uniform random date between `start` and `end` inclusive."""
    return start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))