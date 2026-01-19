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

import faulthandler

import pytest

from rlinf import envs


def test_libero_import_watchdog_disabled(monkeypatch):
    monkeypatch.delenv("RLINF_LIBERO_IMPORT_TIMEOUT", raising=False)
    calls = {}

    def fake_dump(*_args, **_kwargs):
        calls["called"] = True

    monkeypatch.setattr(faulthandler, "dump_traceback_later", fake_dump)
    monkeypatch.setattr(faulthandler, "enable", lambda **_kwargs: None)

    assert envs._maybe_enable_libero_import_watchdog() is None
    assert "called" not in calls


def test_libero_import_watchdog_invalid_timeout(monkeypatch):
    monkeypatch.setenv("RLINF_LIBERO_IMPORT_TIMEOUT", "invalid")
    monkeypatch.setattr(faulthandler, "enable", lambda **_kwargs: None)

    with pytest.warns(RuntimeWarning, match="RLINF_LIBERO_IMPORT_TIMEOUT"):
        assert envs._maybe_enable_libero_import_watchdog() is None


def test_libero_import_watchdog_non_positive_timeout(monkeypatch):
    monkeypatch.setenv("RLINF_LIBERO_IMPORT_TIMEOUT", "0")
    monkeypatch.setattr(faulthandler, "enable", lambda **_kwargs: None)

    with pytest.warns(RuntimeWarning, match="greater than 0"):
        assert envs._maybe_enable_libero_import_watchdog() is None


def test_libero_import_watchdog_enabled(monkeypatch):
    monkeypatch.setenv("RLINF_LIBERO_IMPORT_TIMEOUT", "0.1")
    calls = {}

    monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)

    def fake_enable(**kwargs):
        calls["enable"] = kwargs

    def fake_dump(timeout, **kwargs):
        calls["timeout"] = timeout
        calls["kwargs"] = kwargs

    def fake_cancel():
        calls["cancel"] = True

    def fake_disable():
        calls["disable"] = True

    monkeypatch.setattr(faulthandler, "dump_traceback_later", fake_dump)
    monkeypatch.setattr(faulthandler, "enable", fake_enable)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", fake_cancel)
    monkeypatch.setattr(faulthandler, "disable", fake_disable)

    cleanup = envs._maybe_enable_libero_import_watchdog()
    assert callable(cleanup)
    cleanup()
    assert calls["enable"] == {"all_threads": True}
    assert calls["timeout"] == 0.1
    assert calls["kwargs"]["repeat"] is False
    assert calls["cancel"] is True
    assert calls["disable"] is True
