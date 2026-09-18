"""Launch logic for the upstream Isaac Sim receiver, adapted at RUNTIME.

The shipped receiver (reBotArm_Isaacsim/isaacsim_joint_receiver.py) cannot run on this host as-is:

  - main() parses no arguments, so nothing can be overridden from the command line
  - SimulationApp({"headless": False}) forces a GUI, and DISPLAY is unset here
  - the socket binds to 192.168.1.66, which this host does not have

The plan forbids editing either upstream repository, so every adaptation happens here, in memory, and the file
on disk keeps saying headless False. These helpers are deliberately separable from the expensive part - none
of them starts Isaac Sim - so they can be tested without a 100-200 second app launch.
"""
from __future__ import annotations

import socket
from pathlib import Path

# Isaac Lab is importable only with its source subdirectories on PYTHONPATH.
ISAACLAB_SOURCE_GLOBS = "source/*"

_RECEIVER_REL = Path("reBotArm_Isaacsim/isaacsim_joint_receiver.py")


def upstream_receiver_path(repo_root: Path | str) -> Path:
    """Locate the receiver inside the upstream tree, without touching it."""
    return Path(repo_root) / _RECEIVER_REL


def build_python_path(source_roots, *, existing: str = "") -> str:
    """Join Isaac Lab source roots with whatever PYTHONPATH already held.

    Deliberately does NOT run the inputs through Path. This is a Linux string operation, and Path on Windows
    rewrites "/a" to "\\a" - which is how the first version of the accompanying tests failed. The shell
    launcher builds the same string with a glob, so nothing here may normalise it.
    """
    parts = [str(p) for p in source_roots]
    if existing:
        parts.append(str(existing))
    return ":".join(p for p in parts if p)


def _local_addresses() -> set:
    addresses = {"127.0.0.1", "localhost"}
    try:
        addresses.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addresses.add(info[4][0])
    except OSError:
        pass
    return addresses


def choose_bind_host(configured: str) -> str:
    """Return an address that actually exists here.

    The upstream default is 192.168.1.66, which belongs to the author's lab network. Binding to an address the
    machine does not own fails with EADDRNOTAVAIL, so an unavailable value falls back to loopback - which is
    right anyway for a receiver and its sender on the same host.
    """
    if configured in _local_addresses():
        return configured
    return "127.0.0.1"


def choose_asset_path(patched_dir, upstream_dir) -> Path:
    """Prefer the patched asset tree when it exists, otherwise the upstream one."""
    if patched_dir is not None:
        candidate = Path(patched_dir) / "reBot_B601_DM.usda"
        if candidate.is_file():
            return candidate
    return Path(upstream_dir) / "usd/reBot_B601_DM/reBot_B601_DM.usda"


def force_headless(module) -> None:
    """Wrap the module's SimulationApp so every launch is headless.

    Done by replacement rather than by editing, and idempotent: wrapping twice would nest wrappers and make
    the call stack misleading for no benefit.
    """
    current = getattr(module, "SimulationApp", None)
    if current is None:
        raise AttributeError("module has no SimulationApp to wrap")
    if getattr(current, "_rebot_headless_forced", False):
        return

    def headless_simulation_app(config=None):
        cfg = dict(config or {})
        cfg["headless"] = True
        return current(cfg)

    headless_simulation_app._rebot_headless_forced = True
    headless_simulation_app._rebot_original = current
    module.SimulationApp = headless_simulation_app
