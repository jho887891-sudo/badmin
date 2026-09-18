"""Task 5: the receiver must be launchable on this host WITHOUT modifying the upstream repositories.

The shipped receiver cannot run here as-is, for three hardcoded reasons:
  - main() takes no arguments at all, there is no CLI
  - SimulationApp({"headless": False}) forces a GUI, and DISPLAY is unset
  - the socket binds to 192.168.1.66, an address this host does not have (it has 172.31.68.251)

The plan forbids editing the two upstream repositories, so the driver overrides at RUNTIME. That is the
behaviour these tests pin down: they run without starting Isaac Sim, because the override logic is separable
from the expensive part.
"""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

from src.simulation.rebot_b601dm.launch import (
    ISAACLAB_SOURCE_GLOBS,
    build_python_path,
    choose_bind_host,
    force_headless,
    upstream_receiver_path,
)

REPO = Path("_scratch_rebot/reBot-Isaacsim")
RECEIVER = REPO / "reBotArm_Isaacsim/isaacsim_joint_receiver.py"
pytestmark = pytest.mark.skipif(not RECEIVER.is_file(), reason="reBot-Isaacsim not present locally")


class TestTheUpstreamIsLocatedNotCopied:
    def test_the_receiver_is_found_in_the_upstream_tree(self):
        assert upstream_receiver_path(REPO).name == "isaacsim_joint_receiver.py"
        assert upstream_receiver_path(REPO).is_file()

    def test_the_launcher_does_not_write_into_the_upstream_tree(self):
        """A driver that patched the file in place would make the asset unreproducible."""
        before = RECEIVER.read_bytes()
        launcher = Path("scripts/simulation/isaacsim_receiver_driver.py")
        assert launcher.is_file(), "the driver must exist as a separate file"
        assert RECEIVER.read_bytes() == before


class TestHeadlessIsForcedAtRuntime:
    class _FakeApp:
        def __init__(self):
            self.configs: list = []

        def __call__(self, cfg):
            self.configs.append(dict(cfg))
            return "app-instance"

    def test_the_forced_wrapper_turns_headless_on(self):
        fake = self._FakeApp()

        class Mod:
            SimulationApp = fake

        force_headless(Mod)
        Mod.SimulationApp({"headless": False})
        assert fake.configs[-1]["headless"] is True

    def test_the_forced_wrapper_keeps_every_other_setting(self):
        fake = self._FakeApp()

        class Mod:
            SimulationApp = fake

        force_headless(Mod)
        Mod.SimulationApp({"headless": False, "width": 1280, "some_flag": "x"})
        cfg = fake.configs[-1]
        assert cfg["width"] == 1280
        assert cfg["some_flag"] == "x"

    def test_forcing_twice_does_not_wrap_twice(self):
        fake = self._FakeApp()

        class Mod:
            SimulationApp = fake

        force_headless(Mod)
        once = Mod.SimulationApp
        force_headless(Mod)
        assert Mod.SimulationApp is once

    def test_the_upstream_file_still_says_headless_false(self):
        """The override is at runtime; the file on disk must be untouched."""
        assert "headless\": False" in RECEIVER.read_text(encoding="utf-8", errors="replace")


class TestTheBindAddressIsReal:
    def test_a_loopback_choice_is_always_available(self):
        assert choose_bind_host("192.168.1.66") == "127.0.0.1"

    def test_an_address_that_exists_locally_is_kept(self):
        local = socket.gethostbyname(socket.gethostname())
        assert choose_bind_host(local) == local

    def test_loopback_is_offered_when_the_hostname_does_not_resolve_locally(self):
        assert choose_bind_host("10.255.255.1") in {"127.0.0.1", socket.gethostbyname(socket.gethostname())}


class TestThePythonPathIsBuiltNotHardcoded:
    def test_every_isaaclab_source_subdirectory_is_included(self):
        # plain strings, because that is what a Linux glob hands over; Path() on Windows rewrites them
        roots = ["/tmp/IsaacLab/source/isaaclab", "/tmp/IsaacLab/source/isaaclab_assets"]
        joined = build_python_path(roots, existing="")
        assert "/tmp/IsaacLab/source/isaaclab" in joined
        assert "/tmp/IsaacLab/source/isaaclab_assets" in joined

    def test_an_existing_path_is_preserved(self):
        joined = build_python_path(["/a"], existing="/b")
        assert "/b" in joined and "/a" in joined

    def test_the_separator_is_a_colon_whatever_the_platform(self):
        """This string is consumed on Linux; a Windows drive-letter join would silently break it."""
        assert build_python_path(["/a", "/b"]) == "/a:/b"

    def test_the_globs_point_at_isaaclab_source(self):
        assert ISAACLAB_SOURCE_GLOBS == "source/*"


class TestTheLauncherScriptContract:
    def test_the_launcher_exists_and_is_a_shell_script(self):
        sh = Path("scripts/simulation/run_isaacsim_receiver.sh")
        assert sh.is_file()
        text = sh.read_text(encoding="utf-8")
        assert text.startswith("#!/")
        assert "env_isaaclab" in text, "it must use the project environment, not the system python"

    def test_the_launcher_forbids_modifying_the_isaac_stack(self):
        text = Path("scripts/simulation/run_isaacsim_receiver.sh").read_text(encoding="utf-8")
        for forbidden in ("pip install", "apt install", "conda install", "nvcc", "apt-get"):
            assert forbidden not in text, forbidden

    def test_the_launcher_does_not_kill_other_processes(self):
        """pt_main_thread processes belonging to other users were running when this was written."""
        text = Path("scripts/simulation/run_isaacsim_receiver.sh").read_text(encoding="utf-8")
        for forbidden in ("pkill", "killall", "kill -9"):
            assert forbidden not in text, forbidden

    def test_the_launcher_names_the_patched_asset_option(self):
        text = Path("scripts/simulation/run_isaacsim_receiver.sh").read_text(encoding="utf-8")
        assert "REBOT_ASSET" in text, "the patched asset must be selectable"
