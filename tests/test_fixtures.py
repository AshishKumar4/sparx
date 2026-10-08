"""Every committed fixture is the one its tool wrote, and names the tool and the environment that make it."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("references", ROOT / "tools" / "references.py")
assert _spec is not None and _spec.loader is not None
references = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(references)

COMMITTED = sorted(path.name for path in references.FIXTURES.iterdir() if path.name != "SHA256SUMS")


@pytest.mark.parametrize("name", COMMITTED)
def test_a_fixture_is_the_one_its_checksum_records(name):
    # A fixture changes only with its tool, which tools/regenerate.py --update records.
    assert references.digest(references.FIXTURES / name) == references.recorded()[name]


def test_every_fixture_names_its_tool_and_an_environment_that_is_locked():
    assert set(references.MADE_BY) == set(COMMITTED) == set(references.recorded())
    locks = references.ENVIRONMENTS.iterdir()
    environments = {path.stem for path in locks if path.suffix in (".txt", ".yml")}
    for name, (tool, environment) in references.MADE_BY.items():
        assert (ROOT / "tools" / tool).is_file(), name
        assert environment in environments | {"sparx"}, name
    assert set(references.LOCKED) | {"nest"} == environments
