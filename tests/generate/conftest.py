import pytest

from nsh_pipeline.generate.run import build_collections


def make_test_params(seed=42, clients=20, max_members=30) -> dict:
    return {
        "seed": seed,
        "scale": "test",
        "reference_date": "2025-06-15",
        "scales": {"test": {"clients": clients, "max_members_per_client": max_members}},
    }


def build_world(seed=42) -> dict[str, list[dict]]:
    """Generated collections only; build_collections also returns the injected-issues list."""
    return build_collections(make_test_params(seed))[0]


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture
def build():
    return build_world