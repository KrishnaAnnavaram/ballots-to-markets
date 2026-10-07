import pytest

from ballots_to_markets.config import Settings
from ballots_to_markets.pipeline import load_study
from ballots_to_markets.synthetic import SynthSpec


@pytest.fixture(scope="session")
def settings():
    return Settings(seed=7)


@pytest.fixture(scope="session")
def study(settings):
    return load_study(settings, synthetic=True, spec=SynthSpec(seed=7, election_effect=0.06))
