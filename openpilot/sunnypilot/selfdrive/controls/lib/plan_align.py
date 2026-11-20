"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Plan-Aligned Braking (alpha).

Stage 1:
Blend the learned action acceleration toward the stronger of:
- The plan's average acceleration over the configured horizon.
- The deceleration required to reach the plan's predicted stop point.

Stage 2:
When the plan predicts a stop within the model horizon, enforce at least
the deceleration needed to reach that stop point, compensating for brake
actuation delay.

Both stages are disabled when a lead vehicle is present.

Only active for the Tinygrad custom model and when PlanAlignBraking is enabled.
"""

import numpy as np

from openpilot.cereal import custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants


PARAM = "PlanAlignBraking"


# Stage 1: plan/action blending
BETA = 0.7
HORIZON = 1.0
FLOOR = -3.0
STOP_V = 0.3
MIN_V = 2.0
MIN_DIST = 3.0
TAU = 0.5
RAMP = 0.3


# Stage 2: stop-point enforcement
ENFORCE = True
ENFORCE_STOP_V = 1.0
ENFORCE_FLOOR = -3.5
ENFORCE_DELAY = 0.3
ENFORCE_MARGIN = 1.1
ENFORCE_TAU = 0.3
ENFORCE_HOLD = 15
ENFORCE_MIN_V = 2.0
ENFORCE_MIN_DIST = 3.0
ENFORCE_ACTIVE = 0.05


PARAM_CHECK_FRAMES = max(1, int(1.0 / DT_MDL))


def _runner_value(value):
  """
  Params.get() returns bytes, not an enum.

  Convert the stored value into an integer before comparing it with
  the Tinygrad runner enum. Also tolerate None and malformed values.
  """
  if value is None:
    return -1

  try:
    if isinstance(value, bytes):
      value = value.decode("utf-8")
    return int(value)
  except (TypeError, ValueError, UnicodeDecodeError):
    return -1


class PlanAlignBraking:
  def __init__(self):
    self.params = Params()
    self.enabled = False
    self._frame = 0

    self.a_stop_f = FirstOrderFilter(0.0, TAU, DT_MDL)
    self.a_enf_f = FirstOrderFilter(0.0, ENFORCE_TAU, DT_MDL)

    self._hold = 0
    self._last_req = 0.0

    self._read_params()

  def _read_params(self):
    runner = _runner_value(self.params.get("ModelRunnerTypeCache"))

    tinygrad_runner = int(custom.ModelManagerSP.Runner.tinygrad)

    toggle_enabled = self.params.get_bool(PARAM)

    self.enabled = runner == tinygrad_runner and toggle_enabled

  def update_params(self):
    if self._frame % PARAM_CHECK_FRAMES == 0:
      self._read_params()

    self._frame += 1

  def _reset(self):
    self.a_stop_f.x = 0.0
    self.a_enf_f.x = 0.0
    self._hold = 0
    self._last_req = 0.0

  def update(self, model_msg, a_action: float, has_lead: bool) -> float:
    self.update_params()

    if not self.enabled:
      self._reset()
      return a_action

    t = np.asarray(ModelConstants.T_IDXS, dtype=float)
    v = np.asarray(model_msg.velocity.x, dtype=float)
    x = np.asarray(model_msg.position.x, dtype=float)

    if (
      len(v) != len(t)
      or len(x) != len(t)
      or len(t) < 2
      or not np.all(np.isfinite(v))
      or not np.all(np.isfinite(x))
      or not np.all(np.isfinite(t))
      or not np.isfinite(a_action)
    ):
      self._reset()
      return a_action

    if v[0] < 0.0:
      self._reset()
      return a_action

    # ------------------------------------------------------------
    # Stage 1: calculate plan acceleration and stop-point braking.
    # ------------------------------------------------------------

    a_plan = (np.interp(HORIZON, t, v) - v[0]) / HORIZON

    a_stop = 0.0

    if v[0] > MIN_V:
      stop_indices = np.flatnonzero(v < STOP_V)

      if stop_indices.size:
        stop_distance = max(float(x[stop_indices[0]]), MIN_DIST)
        a_stop = -(v[0] ** 2) / (2.0 * stop_distance)

    a_stop = self.a_stop_f.update(float(a_stop))

    # ------------------------------------------------------------
    # Stage 2: calculate required deceleration to reach stop point.
    # ------------------------------------------------------------

    a_req = 0.0

    if ENFORCE and v[0] > ENFORCE_MIN_V:
      stop_indices = np.flatnonzero(v < ENFORCE_STOP_V)

      if stop_indices.size:
        stop_distance = float(x[stop_indices[0]])

        # Account for distance traveled during brake actuation delay.
        effective_distance = stop_distance - v[0] * ENFORCE_DELAY
        effective_distance = max(effective_distance, ENFORCE_MIN_DIST)

        self._last_req = (
          -ENFORCE_MARGIN
          * v[0] ** 2
          / (2.0 * effective_distance)
        )

        self._hold = ENFORCE_HOLD
        a_req = self._last_req

      elif self._hold > 0:
        self._hold -= 1
        a_req = self._last_req

      else:
        self._last_req = 0.0

    else:
      self._hold = 0
      self._last_req = 0.0

    a_enf = self.a_enf_f.update(float(a_req))

    # Lead-car behavior is unchanged. The planner retains responsibility
    # for choosing the appropriate lead-car braking command.
    if has_lead:
      return a_action

    # ------------------------------------------------------------
    # Stage 1: blend toward the stronger plan-based deceleration.
    # ------------------------------------------------------------

    a_target = min(a_plan, a_stop)

    weight = float(np.clip(-a_action / RAMP, 0.0, 1.0))

    a_out = (
      a_action
      + BETA * weight * min(a_target - a_action, 0.0)
    )

    # Do not allow stage 1 to exceed its configured braking floor.
    a_out = float(max(a_out, min(a_action, FLOOR)))

    # ------------------------------------------------------------
    # Stage 2: enforce stopping deceleration.
    # ------------------------------------------------------------

    if a_enf < -ENFORCE_ACTIVE:
      required = max(a_enf, ENFORCE_FLOOR)
      a_out = min(a_out, required)

    return float(a_out)
