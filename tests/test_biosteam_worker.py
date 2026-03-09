"""Tests for BioSTEAM worker compatibility patches."""

from types import SimpleNamespace


def test_patch_baseline_process_compat_handles_missing_boiler_for_c2():
    from strap.vendor.biosteam_worker import (
        _NullBoiler,
        _patch_baseline_process_compat,
    )

    class FakeScenario:
        turbogenerator = False

    class FakeProcess:
        create_model_calls = 0

        def __init__(self):
            self.scenario = FakeScenario()
            self.solvent_loss = SimpleNamespace(outs=["loss-stream"])

        def create_model(self):
            type(self).create_model_calls += 1
            if not self.scenario.turbogenerator:
                self.BT = self.B
            self.BT.ins[0] = self.solvent_loss.outs[0]
            return "ok"

    Patched = _patch_baseline_process_compat(FakeProcess)
    process = Patched()

    result = process.create_model()

    assert result == "ok"
    assert isinstance(process.B, _NullBoiler)
    assert process.BT.ins[0] == "loss-stream"
    assert FakeProcess.create_model_calls == 1


def test_patch_baseline_process_compat_reuses_existing_bt_alias():
    from strap.vendor.biosteam_worker import _patch_baseline_process_compat

    existing_bt = SimpleNamespace(ins=[None], natural_gas_price=0.0)

    class FakeScenario:
        turbogenerator = False

    class FakeProcess:
        def __init__(self):
            self.scenario = FakeScenario()
            self.BT = existing_bt
            self.solvent_loss = SimpleNamespace(outs=["loss-stream"])

        def create_model(self):
            if not self.scenario.turbogenerator:
                self.BT = self.B
            self.BT.ins[0] = self.solvent_loss.outs[0]
            return "ok"

    Patched = _patch_baseline_process_compat(FakeProcess)
    process = Patched()

    process.create_model()

    assert process.B is existing_bt
    assert process.BT is existing_bt
    assert existing_bt.ins[0] == "loss-stream"


def test_patch_baseline_process_compat_is_idempotent():
    from strap.vendor.biosteam_worker import _patch_baseline_process_compat

    class FakeScenario:
        turbogenerator = True

    class FakeProcess:
        def __init__(self):
            self.scenario = FakeScenario()

        def create_model(self):
            return "ok"

    once = _patch_baseline_process_compat(FakeProcess)
    twice = _patch_baseline_process_compat(FakeProcess)

    assert once is FakeProcess
    assert twice is FakeProcess
    assert getattr(FakeProcess, "_strap_worker_compat_patched", False) is True
