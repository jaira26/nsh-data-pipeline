from datetime import date

import pytest

from nsh_pipeline.generate.activity import make_claims, make_swipes, make_tickets
from nsh_pipeline.generate.cards import make_cards
from nsh_pipeline.generate.clients import make_clients
from nsh_pipeline.generate.common import make_generators
from nsh_pipeline.generate.people import make_people

REF = date(2025, 6, 15)


def build_world(seed=42, n_clients=20, max_members=30) -> dict[str, list[dict]]:
    rng, fake = make_generators(seed)
    clients = make_clients(rng, fake, n_clients, REF)
    people = make_people(rng, fake, clients, REF, max_members)
    swipes = make_swipes(rng, fake, clients, people, REF)
    tickets = make_tickets(rng, people, swipes, REF)
    claims = make_claims(rng, clients, people, REF)
    cards = make_cards(rng, clients, people, swipes, REF)
    return {
        "applications": clients, "people": people, "wex_claims": swipes,
        "tickets": tickets, "claims": claims, "debit_card_status": cards,
    }


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture
def build():
    return build_world