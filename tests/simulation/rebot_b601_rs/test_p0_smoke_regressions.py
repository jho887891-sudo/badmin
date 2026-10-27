# -*- coding: utf-8 -*-
"""P0-R63 regression gates for fail-closed smoke report and USD mount authoring.

Intentionally import-free wrt Isaac Lab: test the actual main() finalizer body
and code placement, while existing usd-core and target-host tests cover runtime.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "simulation" / "p0_b601_racket_scene_smoke.py"


def _main_ast() -> ast.FunctionDef:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _main_finally() -> list[ast.stmt]:
    entry = _main_ast()
    candidates = [
        node for node in entry.body
        if isinstance(node, ast.Try)
        and node.finalbody
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "close"
            for stmt in node.finalbody
            for child in ast.walk(stmt)
        )
    ]
    assert len(candidates) == 1, "expected one main-level cleanup/finalizer"
    return candidates[0].finalbody


def _finalizer_callable(writer):
    """Execute the source finalizer, not an independently rewritten example."""
    fn = ast.FunctionDef(
        name="actual_finalizer",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="app"), ast.arg(arg="output"), ast.arg(arg="report")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=copy.deepcopy(_main_finally()),
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    namespace = {"write_report_exclusive": writer}
    program = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    return namespace["actual_finalizer"]


class NativeExitSimulation(BaseException):
    """Models an application close that cannot return to normal Python code."""


class ExitOnCloseApp:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True
        raise NativeExitSimulation("close terminated interpreter")


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_report_persisted_before_app_close_can_terminate_python(tmp_path, status):
    output = tmp_path / "smoke.json"
    report = {"status": status, "checks": {"steps": 120}}

    def write_exclusive(path, value):
        with path.open("x", encoding="utf-8") as fp:
            json.dump(value, fp)

    finalize = _finalizer_callable(write_exclusive)
    app = ExitOnCloseApp()
    with pytest.raises(NativeExitSimulation):
        finalize(app, output, report)
    assert app.closed
    assert output.is_file(), "app.close() must never run before JSON is written"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_finalizer_rejects_existing_report_before_close(tmp_path):
    output = tmp_path / "smoke.json"
    output.write_text('{"immutable": true}', encoding="utf-8")
    calls = []

    def write_exclusive(path, value):
        calls.append("write")
        with path.open("x", encoding="utf-8") as fp:
            json.dump(value, fp)

    finalize = _finalizer_callable(write_exclusive)
    app = ExitOnCloseApp()
    with pytest.raises(NativeExitSimulation):
        finalize(app, output, {"status": "PASS"})
    assert calls == ["write"]
    assert output.read_text(encoding="utf-8") == '{"immutable": true}'
    assert app.closed


def test_uninstanceable_must_be_applied_before_authoring_under_gripper():
    main = _main_ast()
    calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
    ]
    uninstanceable = [
        c for c in calls if ast.unparse(c.func) == "sim_utils.make_uninstanceable"
    ]
    assert len(uninstanceable) == 1, (
        "B601-RS USD hierarchy is instanceable; call make_uninstanceable"
        " on the runtime stage before authoring /gripper_end/RacketMount"
    )
    call = uninstanceable[0]
    assert len(call.args) >= 2
    assert ast.unparse(call.args[0]) == "REBOT_RS_PRIM_PATH"
    assert ast.unparse(call.args[1]) == "stage"
    add_visual = [c for c in calls if ast.unparse(c.func) == "attach_racket_visual"]
    assert len(add_visual) == 1
    assert call.lineno < add_visual[0].lineno
    assert any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "stage" for t in n.targets)
        and n.lineno < call.lineno
        for n in ast.walk(main)
    ), "make_uninstanceable must receive an initialized runtime USD stage"


def test_main_does_not_author_into_usd_asset_path():
    """Never edit the source USD; stage surgery targets the spawned prim path."""
    script = SCRIPT.read_text(encoding="utf-8")
    assert "sim_utils.make_uninstanceable(REBOT_RS_PRIM_PATH, stage)" in script
    assert "sim_utils.make_uninstanceable(str(robot_usd)" not in script
