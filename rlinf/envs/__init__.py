# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import faulthandler
import multiprocessing as mp
import os
import warnings
from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"


def _ensure_libero_start_method() -> None:
    start_method = mp.get_start_method(allow_none=True)
    if start_method is None:
        mp.set_start_method("spawn")
        return
    if start_method != "spawn":
        if os.environ.get("RLINF_LIBERO_FORCE_SPAWN") == "1":
            mp.set_start_method("spawn", force=True)
            return
        warnings.warn(
            "Libero import can hang when multiprocessing uses fork with active "
            "threads. Set multiprocessing start method to 'spawn' in the entrypoint "
            "or set RLINF_LIBERO_FORCE_SPAWN=1 to override.",
            RuntimeWarning,
            stacklevel=2,
        )


def _maybe_enable_libero_import_watchdog():
    timeout_raw = os.environ.get("RLINF_LIBERO_IMPORT_TIMEOUT", "").strip()
    if not timeout_raw:
        return None
    try:
        timeout = float(timeout_raw)
    except ValueError:
        warnings.warn(
            "Invalid RLINF_LIBERO_IMPORT_TIMEOUT value. Expected seconds as float.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if timeout <= 0:
        warnings.warn(
            "RLINF_LIBERO_IMPORT_TIMEOUT must be greater than 0 to enable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    was_enabled = faulthandler.is_enabled()
    if not was_enabled:
        faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(timeout, repeat=False)

    def cleanup() -> None:
        with contextlib.suppress(Exception):
            faulthandler.cancel_dump_traceback_later()
        if not was_enabled:
            with contextlib.suppress(Exception):
                faulthandler.disable()

    return cleanup


def get_env_cls(env_type: str, env_cfg=None, enable_offload=False):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.

    Notes:
        For LIBERO, set RLINF_LIBERO_IMPORT_TIMEOUT to a positive number of seconds
        to dump stack traces if the import stalls. Use RLINF_LIBERO_FORCE_SPAWN=1 if
        you need to force multiprocessing to use the "spawn" start method in this
        process.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if not enable_offload:
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
        else:
            from rlinf.envs.maniskill.maniskill_offload_env import (
                ManiskillOffloadEnv as ManiskillEnv,
            )

        return ManiskillEnv
    elif env_type == SupportedEnvType.LIBERO:
        _ensure_libero_start_method()
        watchdog_cleanup = _maybe_enable_libero_import_watchdog()
        try:
            from rlinf.envs.libero.libero_env import LiberoEnv
        finally:
            if watchdog_cleanup:
                watchdog_cleanup()

        return LiberoEnv
    elif env_type == SupportedEnvType.ROBOTWIN:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    elif env_type == SupportedEnvType.ISAACLAB:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        assert task_id in REGISTER_ISAACLAB_ENVS, (
            f"Task type {task_id} has not been registered! "
            f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
        )
        return REGISTER_ISAACLAB_ENVS[task_id]
    elif env_type == SupportedEnvType.METAWORLD:
        from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == SupportedEnvType.BEHAVIOR:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == SupportedEnvType.CALVIN:
        from rlinf.envs.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == SupportedEnvType.ROBOCASA:
        from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == SupportedEnvType.REALWORLD:
        from rlinf.envs.realworld.realworld_env import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
