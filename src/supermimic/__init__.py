"""SuperMimic: humanoid motion tracking with one differentiable world model."""

import ctypes.util
import os
import sys

# mjlab defaults MUJOCO_GL to "egl" on Linux, and mujoco then fails at import on
# machines without libEGL (headless servers, CI runners), even for commands that
# never render. Training and evaluation do not render, so fall back to
# "disable" when EGL is missing. Runs before mjlab is imported (every entry
# point lives in this package); an explicit MUJOCO_GL always wins.
if (
  sys.platform.startswith("linux")
  and "MUJOCO_GL" not in os.environ
  and ctypes.util.find_library("EGL") is None
):
  os.environ["MUJOCO_GL"] = "disable"
