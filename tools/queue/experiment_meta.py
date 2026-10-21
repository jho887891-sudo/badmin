#!/usr/bin/env python
"""A05: experiment config + environment fingerprint + manifest-vs-applied validation.

Builds a machine-readable metadata block for an experiment directory and *fails*
(non-zero exit, explicit reason) when what the manifest claims disagrees with what
the command actually applied.  The motivating real bug, quoted from the master spec
(EXECUTION_QUEUE_CONTINUOUS_P0_P8.md section 1) and from p2_gate_runner.py
(commit ce278c2, lines 629-630, check_manifest_consistency):

    the manifest recorded gravity [0,0,0] while the command ran with
    --gravity-z -9.81, so every "zero-gravity" claim in that run was false

Scope rules
-----------
* pure CPU, standard library only, never imports Isaac Sim / pxr / Kit
* the environment fingerprint is python + platform + importable package versions.
  It is deliberately NOT a GPU probe: GPU state is not reproducible and is not
  needed to reproduce a CPU-side contract.
* the config hash is sha256 over the *sorted, normalised* parameters, so key order
  cannot change it and a single value change always changes it.
* this module is the audit-side counterpart of p2_gate_runner.py.  The runner
  enforces a gravity/manifest rule *before* a run starts (check_manifest_consistency,
  tools/simulation/p2_gate_runner.py:626 at commit ce278c2); this module re-checks
  the committed evidence afterwards, on CPU, without trusting the producer.

Usage
-----
    python tools/queue/experiment_meta.py validate --experiment <run-dir>
    python tools/queue/experiment_meta.py build    --experiment <run-dir> --out meta.json
    python tools/queue/experiment_meta.py validate --experiment <run-dir> --applied gravity_z=-9.81

Exit codes: 0 = valid, 2 = mismatch/validation failure, 1 = usage or IO error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
from pathlib import Path

SCHEMA_VERSION = 1

#: tolerance for float comparisons between two independent recordings
FLOAT_TOL = 1e-6

#: packages whose version belongs in the fingerprint *when importable*
FINGERPRINT_PACKAGES = (
    "numpy",
    "scipy",
    "pytest",
    "yaml",
    "jsonschema",
    "pxr",
    "usd-core",
)

#: command flag -> applied parameter name
FLAG_TO_APPLIED = {
    "--gravity-z": "gravity_z",
    "--gravity_z": "gravity_z",
    "--lift": "lift_m",
    "--steps": "steps",
    "--dt": "dt_s",
    "--run-id": "run_id",
    "--run_id": "run_id",
    "--out-dir": "out_dir",
    "--out_dir": "out_dir",
    "--impl": "impl",
    "--gate-mode": "gate_mode",
    "--seed": "seed",
}

#: flags that are booleans and carry no value token
BOOLEAN_FLAGS = {"--read-contacts": "read_contacts", "--contacts": "read_contacts"}

#: asset extensions whose sha256 is recorded
ASSET_SUFFIXES = (".usda", ".usd", ".usdc", ".json", ".jsonl", ".urdf", ".glb", ".stl")


# --------------------------------------------------------------------- hashing
def normalize(value):
    """Recursively canonicalise a parameter value.

    dict keys are sorted by their string form, floats are rounded to 12 decimals so
    that 0.4 and 0.40000000000000002 hash identically, tuples become lists.
    """
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, int):
        return value
    return str(value)


def canonical_json(value) -> str:
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def config_hash(parameters) -> str:
    """sha256 over the sorted, normalised parameters (stable across key order)."""
    return "sha256:" + hashlib.sha256(canonical_json(parameters).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------- environment
def environment_fingerprint(packages=FINGERPRINT_PACKAGES) -> dict:
    """python + platform + importable package versions.  No GPU probe, ever."""
    import importlib
    import importlib.metadata

    found = {}
    for name in packages:
        module_name = name.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - importability is the question, not the error
            found[name] = None
            continue
        try:
            found[name] = importlib.metadata.version(name)
        except Exception:  # noqa: BLE001
            found[name] = "unknown"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable.replace(os.sep, "/"),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "packages_importable": found,
        "gpu_probe": None,  # deliberately absent: not reproducible, not needed
    }


# ------------------------------------------------------------------- command
def parse_command(command: str) -> dict:
    """Extract the parameters a shell command line actually applied."""
    applied: dict = {}
    if not isinstance(command, str) or not command.strip():
        return applied
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag, sep, inline = token.partition("=")
        if flag in BOOLEAN_FLAGS:
            applied[BOOLEAN_FLAGS[flag]] = True
            index += 1
            continue
        key = FLAG_TO_APPLIED.get(flag)
        if key is None:
            index += 1
            continue
        if sep:
            raw_value = inline
            index += 1
        elif index + 1 < len(tokens):
            raw_value = tokens[index + 1]
            index += 2
        else:
            index += 1
            continue
        applied[key] = _coerce(raw_value)
    if isinstance(applied.get("steps"), float):
        applied["steps"] = int(applied["steps"])
    return applied


def _coerce(raw_value: str):
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return raw_value


# ---------------------------------------------------------------- asset shas
def asset_shas(experiment_dir: Path, suffixes=ASSET_SUFFIXES) -> list:
    """sha256 of every evidence/asset file inside the run directory."""
    rows = []
    for path in sorted(Path(experiment_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rows.append({
            "path": path.relative_to(Path(experiment_dir)).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return rows


# ------------------------------------------------------------------ metadata
def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, "unreadable: %s" % exc


def _manifest_claimed(manifest: dict) -> dict:
    claimed = {}
    gravity = manifest.get("gravity_m_s2")
    if isinstance(gravity, (list, tuple)) and len(gravity) >= 3:
        claimed["gravity_m_s2"] = [normalize(float(g)) for g in gravity[:3]]
        claimed["gravity_z_m_s2"] = float(gravity[2])
    elif isinstance(gravity, (int, float)):
        claimed["gravity_z_m_s2"] = float(gravity)
    for manifest_key, applied_key in (
        ("lift_m", "lift_m"),
        ("steps", "steps"),
        ("physics_dt_s", "dt_s"),
        ("run_id", "run_id"),
        ("impl", "impl"),
        ("asset_sha256", "asset_sha256"),
    ):
        if manifest_key in manifest:
            claimed[applied_key] = manifest[manifest_key]
    claimed["command"] = manifest.get("command")
    return claimed


def build_metadata(experiment_dir, applied=None, source_ref=None, fingerprint=None) -> dict:
    """Assemble the A05 metadata block for one experiment directory."""
    experiment_dir = Path(experiment_dir)
    manifest, manifest_error = _load_json(experiment_dir / "manifest.json")
    result, result_error = _load_json(experiment_dir / "result.json")
    manifest = manifest if isinstance(manifest, dict) else {}
    claimed = _manifest_claimed(manifest)

    applied_parameters = dict(applied or {})
    if not applied_parameters and isinstance(claimed.get("command"), str):
        applied_parameters = parse_command(claimed["command"])
    applied_parameters = normalize(applied_parameters)

    hash_payload = {
        "applied": applied_parameters,
        "manifest_parameters": manifest.get("parameters") or manifest.get("params") or {},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "A05",
        "run_id": manifest.get("run_id") or experiment_dir.name,
        "experiment_dir": experiment_dir.as_posix(),
        "source_ref": source_ref,
        "config_hash": config_hash(hash_payload),
        "applied_parameters": applied_parameters,
        "manifest_claimed": normalize(claimed),
        "environment": fingerprint or environment_fingerprint(),
        "assets": asset_shas(experiment_dir),
        "inputs": {
            "manifest": "manifest.json" if manifest_error is None else manifest_error,
            "result": "result.json" if result_error is None else result_error,
            "result_gate_result": (result or {}).get("gate_result")
            if isinstance(result, dict) else None,
        },
    }


# ---------------------------------------------------------------- validation
def validate_metadata(meta: dict, expect_config_hash=None, experiment_dir=None) -> list:
    """Return a list of failure dicts; empty means valid.  Each has code + reason."""
    failures: list = []
    experiment_dir = Path(experiment_dir or meta.get("experiment_dir") or ".")
    claimed = meta.get("manifest_claimed") or {}
    applied = meta.get("applied_parameters") or {}

    def fail(code, reason, **extra):
        row = {"code": code, "reason": reason}
        row.update(extra)
        failures.append(row)

    if meta.get("inputs", {}).get("manifest") != "manifest.json":
        fail("MISSING_MANIFEST",
             "%s has no readable manifest.json (%s); without it nothing is verifiable"
             % (experiment_dir.as_posix(), meta.get("inputs", {}).get("manifest")))

    # ---- the bug we actually hit: manifest gravity vs applied --gravity-z
    claimed_gz = claimed.get("gravity_z_m_s2")
    applied_gz = applied.get("gravity_z")
    if applied_gz is None and claimed.get("command"):
        fail("COMMAND_UNPARSED",
             "manifest.command carries no --gravity-z, so the gravity actually applied "
             "is not evidenced: %r" % (claimed.get("command"),))
    if claimed_gz is not None and applied_gz is not None:
        claimed_gz = float(claimed_gz)
        applied_gz = float(applied_gz)
        if abs(claimed_gz - applied_gz) > FLOAT_TOL:
            zero_claim = abs(claimed_gz) <= FLOAT_TOL and abs(applied_gz) > FLOAT_TOL
            fail("GRAVITY_MISMATCH",
                 "manifest gravity z is %r but the command applied --gravity-z %r "
                 "(delta %r > %r)%s"
                 % (claimed_gz, applied_gz, abs(claimed_gz - applied_gz), FLOAT_TOL,
                    "; the manifest claims ZERO gravity for a run that was NOT "
                    "zero-gravity, so every zero-gravity conclusion in it is false"
                    if zero_claim else ""),
                 manifest_value=claimed_gz, applied_value=applied_gz)
    elif applied_gz is not None and claimed_gz is None:
        fail("GRAVITY_UNCLAIMED",
             "the command applied --gravity-z %r but the manifest declares no gravity_m_s2"
             % (applied_gz,))

    # ---- run id / directory identity
    dir_name = experiment_dir.name
    claimed_run_id = claimed.get("run_id")
    applied_run_id = applied.get("run_id")
    for label, value in (("manifest", claimed_run_id), ("command", applied_run_id)):
        if isinstance(value, str) and value and value != dir_name:
            fail("RUN_ID_MISMATCH",
                 "%s run_id %r is not the directory name %r: evidence may be inherited"
                 % (label, value, dir_name),
                 manifest_value=claimed_run_id, applied_value=applied_run_id)
    if isinstance(claimed_run_id, str) and isinstance(applied_run_id, str) \
            and claimed_run_id != applied_run_id:
        fail("RUN_ID_MISMATCH",
             "manifest run_id %r != command --run-id %r" % (claimed_run_id, applied_run_id))

    # ---- out-dir identity (evidence cannot come from another directory)
    applied_out = applied.get("out_dir")
    if isinstance(applied_out, str) and applied_out:
        out_base = applied_out.replace(chr(92), "/").rstrip("/").split("/")[-1]
        if out_base and out_base != dir_name:
            fail("OUT_DIR_MISMATCH",
                 "the command wrote to --out-dir %r (basename %r) but this directory is %r"
                 % (applied_out, out_base, dir_name))

    # ---- scalar parameter agreement
    for key, label in (("lift_m", "lift"), ("steps", "steps"), ("dt_s", "physics_dt_s")):
        if key in claimed and key in applied:
            left, right = claimed[key], applied[key]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if abs(float(left) - float(right)) > FLOAT_TOL:
                    fail("%s_MISMATCH" % key.upper(),
                         "manifest %s is %r but the command applied %r"
                         % (label, left, right),
                         manifest_value=left, applied_value=right)

    # ---- assets
    assets = meta.get("assets") or []
    if not assets:
        fail("ASSET_MISSING",
             "no asset/evidence file with a known suffix in %s: the run is not "
             "self-describing" % experiment_dir.as_posix())
    for asset in assets:
        if not (experiment_dir / asset["path"]).is_file():
            fail("ASSET_MISSING", "recorded asset %r no longer exists" % asset["path"])
    declared = claimed.get("asset_sha256")
    if isinstance(declared, dict):
        present = {row["path"]: row["sha256"] for row in assets}
        for name, expected in declared.items():
            if name not in present:
                fail("ASSET_MISSING",
                     "manifest declares asset %r which is not in the run" % name)
            elif isinstance(expected, str) and present[name] != expected.replace("sha256:", ""):
                fail("ASSET_SHA_MISMATCH",
                     "asset %r sha256 %s != manifest %s" % (name, present[name], expected))

    # ---- config hash agreement
    if expect_config_hash:
        expected = expect_config_hash
        if not expected.startswith("sha256:"):
            expected = "sha256:" + expected
        if meta.get("config_hash") != expected:
            fail("CONFIG_HASH_MISMATCH",
                 "config hash %r != expected %r" % (meta.get("config_hash"), expected))

    return failures


def validate_experiment(experiment_dir, applied=None, expect_config_hash=None,
                        source_ref=None) -> tuple:
    meta = build_metadata(experiment_dir, applied=applied, source_ref=source_ref)
    return meta, validate_metadata(meta, expect_config_hash=expect_config_hash,
                                   experiment_dir=Path(experiment_dir))


# ----------------------------------------------------------------------- CLI
def _parse_applied(pairs) -> dict:
    applied = {}
    for pair in pairs or ():
        key, sep, raw_value = pair.partition("=")
        if not sep:
            raise SystemExit("--applied needs key=value, got %r" % pair)
        applied[key.strip()] = _coerce(raw_value.strip())
    return applied


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="A05 experiment metadata + validation")
    parser.add_argument("mode", choices=("build", "validate", "fingerprint"))
    parser.add_argument("--experiment", help="experiment run directory")
    parser.add_argument("--applied", action="append", default=[], metavar="KEY=VALUE",
                        help="parameter the command really applied "
                             "(overrides manifest.command)")
    parser.add_argument("--expect-config-hash", default=None)
    parser.add_argument("--ref", default=None, help="git ref the evidence came from")
    parser.add_argument("--out", default=None, help="write the metadata block here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "fingerprint":
        print(json.dumps(environment_fingerprint(), indent=2, sort_keys=True))
        return 0
    if not args.experiment:
        print("--experiment is required for build/validate", file=sys.stderr)
        return 1
    experiment = Path(args.experiment)
    if not experiment.is_dir():
        print("FAIL (EXPERIMENT_MISSING): %s is not a directory" % experiment.as_posix(),
              file=sys.stderr)
        return 1

    meta, failures = validate_experiment(
        experiment, applied=_parse_applied(args.applied),
        expect_config_hash=args.expect_config_hash, source_ref=args.ref)

    if args.out:
        Path(args.out).write_text(
            json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
            encoding="utf-8")
    if args.mode == "build":
        if not args.quiet:
            print("config_hash %s" % meta["config_hash"])
            print("run_id %s assets %d" % (meta["run_id"], len(meta["assets"])))
        if args.out is None:
            print(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if not failures else 2

    if failures:
        print("FAIL (A05_VALIDATION): %d mismatch(es) in %s"
              % (len(failures), experiment.as_posix()), file=sys.stderr)
        for failure in failures:
            print("  - %s: %s" % (failure["code"], failure["reason"]), file=sys.stderr)
        return 2
    print("PASS (A05_VALIDATION): %s config_hash %s assets %d env python %s"
          % (experiment.as_posix(), meta["config_hash"], len(meta["assets"]),
             meta["environment"]["python_version"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
