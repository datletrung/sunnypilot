"""
Plan-Aligned Braking tests.
"""

import types

import numpy as np

from openpilot.cereal import custom
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib import plan_align
from openpilot.sunnypilot.selfdrive.controls.lib.plan_align import PlanAlignBraking


Runner = custom.ModelManagerSP.Runner
T = np.asarray(ModelConstants.T_IDXS, dtype=float)


class FakeParams:
  def __init__(self, runner, toggle):
    self.runner = runner
    self.toggle = toggle

  def get(self, key):
    if key == "ModelRunnerTypeCache":
      # Match the real Params API: values are returned as bytes.
      return str(int(self.runner)).encode("utf-8")
    return None

  def get_bool(self, key):
    if key == plan_align.PARAM:
      return self.toggle
    return False


def make(runner=Runner.tinygrad, toggle=True):
  pa = PlanAlignBraking()
  pa.params = FakeParams(runner, toggle)
  pa._read_params()
  return pa


def stop_plan(v0, d_stop):
  """Create a constant-deceleration plan that stops at d_stop."""
  acceleration = v0 ** 2 / (2.0 * d_stop)
  stop_time = v0 / acceleration

  velocity = np.clip(v0 - acceleration * T, 0.0, None)

  position = np.where(
    T < stop_time,
    v0 * T - 0.5 * acceleration * T ** 2,
    d_stop,
  )

  return types.SimpleNamespace(
    velocity=types.SimpleNamespace(x=list(velocity)),
    position=types.SimpleNamespace(x=list(position)),
  )


def cruise_plan(v0=20.0):
  return types.SimpleNamespace(
    velocity=types.SimpleNamespace(x=[v0] * len(T)),
    position=types.SimpleNamespace(x=list(v0 * T)),
  )


def run(pa, msg, a_action, has_lead=False, n=40):
  out = a_action

  for _ in range(n):
    out = pa.update(msg, a_action, has_lead)

  return out


class TestPlanAlignBraking(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self._enforce = plan_align.ENFORCE

  def tearDown(self):
    plan_align.ENFORCE = self._enforce
    super().tearDown()

  # Runner and toggle

  def test_default_model_is_excluded(self):
    assert not make(runner=Runner.stock, toggle=True).enabled

  def test_custom_model_needs_toggle(self):
    assert make(runner=Runner.tinygrad, toggle=True).enabled
    assert not make(runner=Runner.tinygrad, toggle=False).enabled

  def test_params_bytes_runner_is_recognized(self):
    pa = PlanAlignBraking()
    pa.params = FakeParams(Runner.tinygrad, True)
    pa._read_params()
    assert pa.enabled

  def test_disabled_is_passthrough(self):
    assert run(
      make(Runner.stock, True),
      stop_plan(80 / 3.6, 120),
      -1.3,
    ) == -1.3

  # Stage 1

  def test_blend_high_speed_stop_brakes_harder(self):
    plan_align.ENFORCE = False

    out = run(make(), stop_plan(80 / 3.6, 120), -1.3)

    assert -2.1 < out < -1.5

  def test_blend_never_weaker_and_respects_floor(self):
    plan_align.ENFORCE = False

    out = run(make(), stop_plan(80 / 3.6, 60), -1.3)

    assert plan_align.FLOOR - 1e-6 <= out <= -1.3

    assert run(
      make(),
      stop_plan(15.0, 60),
      -2.5,
    ) == -2.5

  # Stage 2

  def test_enforcement_brakes_to_stop_point(self):
    out = run(make(), stop_plan(15.0, 46.0), -0.5)

    assert plan_align.ENFORCE_FLOOR <= out < -2.7

  def test_enforcement_high_speed(self):
    out = run(make(), stop_plan(80 / 3.6, 100), -1.3)

    assert -3.2 < out < -2.6

  def test_no_enforcement_until_plan_commits_to_stop(self):
    plan = stop_plan(80 / 3.6, 120)

    on = run(make(), plan, -1.3)

    plan_align.ENFORCE = False
    off = run(make(), plan, -1.3)

    assert on == off

  def test_enforcement_hits_floor_when_stop_is_too_close(self):
    out = run(make(), stop_plan(22.0, 25.0), -0.5)

    assert out == plan_align.ENFORCE_FLOOR

  def test_enforcement_never_weakens_existing_braking(self):
    assert run(
      make(),
      stop_plan(15.0, 46.0),
      -3.4,
    ) == -3.4

    assert run(
      make(),
      stop_plan(10.0, 120.0),
      -1.5,
    ) == -1.5

  def test_enforcement_overrides_positive_action(self):
    out = run(make(), stop_plan(15.0, 46.0), 0.8)

    assert out < -2.7

  def test_enforcement_holds_through_plan_flicker(self):
    pa = make()

    run(pa, stop_plan(15.0, 46.0), -0.5)

    out = run(
      pa,
      cruise_plan(15.0),
      -0.5,
      n=plan_align.ENFORCE_HOLD - 2,
    )

    assert out < -2.0

  def test_enforcement_releases_after_hold(self):
    pa = make()

    run(pa, stop_plan(15.0, 46.0), -0.5)

    out = run(
      pa,
      cruise_plan(15.0),
      -0.5,
      n=plan_align.ENFORCE_HOLD + 80,
    )

    assert abs(out - (-0.5)) < 0.05

  def test_enforcement_off_below_min_speed(self):
    plan = stop_plan(plan_align.ENFORCE_MIN_V - 0.5, 3.0)

    on = run(make(), plan, -0.2)

    plan_align.ENFORCE = False
    off = run(make(), plan, -0.2)

    assert on == off

  # Unmodified behavior

  def test_lead_is_untouched(self):
    assert run(
      make(),
      stop_plan(80 / 3.6, 120),
      -1.3,
      has_lead=True,
    ) == -1.3

  def test_cruise_is_untouched(self):
    assert run(make(), cruise_plan(), 0.1) == 0.1

  def test_bad_plan_length_is_passthrough(self):
    bad = types.SimpleNamespace(
      velocity=types.SimpleNamespace(x=[1.0, 2.0]),
      position=types.SimpleNamespace(x=[1.0, 2.0]),
    )

    assert run(make(), bad, -1.0) == -1.0
