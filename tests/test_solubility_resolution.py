"""Regression tests for deterministic name resolution in the solubility engine.

`resolve_solvent`/`resolve_polymer` used to return the first substring match
from a `set`, whose iteration order varies across processes under Python string
hash randomization (PYTHONHASHSEED). That made `get_solubility(..., "auto")`
return different values on different runs — e.g. `o-Xylene` resolved to
different DB keys (44% / 5.5% / None) — so the reward model and every
name-resolving tool were unreproducible. These tests pin the fix.
"""

import subprocess
import sys
import textwrap

from strap.solubility import _unique_substring_match, resolve_polymer, resolve_solvent


class TestUniqueSubstringMatch:
    def test_resolves_only_when_unique(self):
        # exactly one candidate contains "toluen" -> resolve
        assert _unique_substring_match("toluen", {"toluene", "heptane"}) == "toluene"

    def test_ambiguous_returns_none(self):
        # "PE" matches four distinct polymers -> refuse to guess (chemically unsafe)
        assert _unique_substring_match("PE", {"LDPE", "HDPE", "PET", "PES"}) is None
        # "dimethylbenzene" matches both xylene isomers -> refuse
        assert _unique_substring_match("dimethylbenzene",
                                       {"1,2-dimethylbenzene", "1,4-dimethylbenzene"}) is None

    def test_order_independent(self):
        a = {"ldpe", "hdpe", "pe-foo"}
        b = set(reversed(list(a)))
        assert _unique_substring_match("pe", a) == _unique_substring_match("pe", b)

    def test_no_match_returns_none(self):
        assert _unique_substring_match("zzz", {"toluene", "heptane"}) is None


class TestResolvers:
    def test_ldpe_and_hdpe_stay_distinct(self):
        known = {"LDPE", "HDPE", "PET", "PES", "PP", "PS", "EVOH"}
        assert resolve_polymer("LDPE", known) == "LDPE"   # exact, distinct
        assert resolve_polymer("HDPE", known) == "HDPE"   # exact, distinct

    def test_generic_pe_maps_to_documented_representative_not_pet_pes(self):
        known = {"LDPE", "HDPE", "PET", "PES", "PP", "PS", "EVOH"}
        # bare "PE" resolves via the explicit alias to HDPE — never to PET/PES,
        # and never by arbitrary substring
        assert resolve_polymer("PE", known) == "HDPE"

    def test_resolve_solvent_isomer_via_alias(self):
        known = {"1,2-dimethylbenzene", "1,4-dimethylbenzene", "toluene"}
        assert resolve_solvent("o-xylene", known) == "1,2-dimethylbenzene"  # alias, correct isomer
        assert resolve_solvent("p-xylene", known) == "1,4-dimethylbenzene"


class TestCrossProcessDeterminism:
    """The real bug only shows across processes (different hash seeds). Run the
    same lookups under two explicit, different PYTHONHASHSEED values and require
    identical results."""

    _SNIPPET = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        from strap.solubility import get_solubility
        pairs = [("PE","o-Xylene",135.0),("EVOH","Ethylene Glycol",140.0),
                 ("LDPE","xylene",90.0),("HDPE","toluene",100.0),("PS","xylene",80.0)]
        print([get_solubility(p, s, t, method="auto") for p, s, t in pairs])
        """
    )

    def _run(self, seed: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", self._SNIPPET],
            capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": _path()},
            cwd=_repo_root(),
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    def test_identical_results_under_different_hash_seeds(self):
        assert self._run("0") == self._run("12345") == self._run("99999")


def _repo_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1])


def _path() -> str:
    import os

    return os.environ.get("PATH", "")
