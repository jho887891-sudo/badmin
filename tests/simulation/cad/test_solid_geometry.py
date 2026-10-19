"""Unit tests for the per-solid STEP geometry tool.

The fixtures are hand-written miniature STEP files so that the tests pin down
the *conservative* behaviour that matters: extents must come from BREP vertices,
never from LINE origins; shape labels are ratio hints; unnamed parts stay
UNKNOWN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.step_geometry_census import parse_step  # noqa: E402
from scripts.simulation.step_solid_geometry import (  # noqa: E402
    SHAPE_BLOCK,
    SHAPE_DISC,
    SHAPE_ROD,
    SHAPE_SLAB,
    build_report,
    circle_facts,
    part_group,
    part_root,
    shape_fingerprint,
    shape_hint,
    solid_geometry,
    vertex_bbox,
)

HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('mini.step','2026-10-20T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#14=(GEOMETRIC_REPRESENTATION_CONTEXT(3)REPRESENTATION_CONTEXT('',''));
#20=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#100),#14);
"""

TRAILER = """ENDSEC;
END-ISO-10303-21;
"""

# One solid: vertices on a 10 mm segment, but its EDGE_CURVE is carried by a LINE
# whose origin is 900 m away -- that origin must not influence any extent.
SOLID_WITH_FAR_LINE_ORIGIN = """#100=MANIFOLD_SOLID_BREP('M1.2-00V3 底盘模块-1/M1.2.2-00V3 舵轮前组件-2/减震器-1/减震器弹簧-1-solid1',#101);
#101=CLOSED_SHELL('',(#110));
#110=ADVANCED_FACE('',(#111),#150,.T.);
#111=FACE_OUTER_BOUND('',#112,.T.);
#112=EDGE_LOOP('',(#113,#116));
#113=ORIENTED_EDGE('',*,*,#114,.T.);
#114=EDGE_CURVE('',#120,#121,#160,.T.);
#116=ORIENTED_EDGE('',*,*,#117,.T.);
#117=EDGE_CURVE('',#120,#121,#180,.T.);
#120=VERTEX_POINT('',#130);
#121=VERTEX_POINT('',#131);
#130=CARTESIAN_POINT('',(0.,0.,0.));
#131=CARTESIAN_POINT('',(10.,0.,0.));
#135=CARTESIAN_POINT('',(900000.,0.,0.));
#160=LINE('',#135,#172);
#172=VECTOR('',#173,1.);
#173=DIRECTION('',(1.,0.,0.));
#180=CIRCLE('',#181,5.5);
#181=AXIS2_PLACEMENT_3D('',#130,#170,#171);
#150=PLANE('',#151);
#151=AXIS2_PLACEMENT_3D('',#130,#170,#171);
#170=DIRECTION('',(0.,0.,1.));
#171=DIRECTION('',(1.,0.,0.));
#800=MANIFOLD_SOLID_BREP('NONE',#801);
#801=CLOSED_SHELL('',(#810));
#810=ADVANCED_FACE('',(#811),#850,.T.);
#811=FACE_OUTER_BOUND('',#812,.T.);
#812=EDGE_LOOP('',(#813));
#813=ORIENTED_EDGE('',*,*,#814,.T.);
#814=EDGE_CURVE('',#820,#821,#860,.T.);
#820=VERTEX_POINT('',#830);
#821=VERTEX_POINT('',#831);
#830=CARTESIAN_POINT('',(50.,500.,0.));
#831=CARTESIAN_POINT('',(70.,500.,0.));
#860=LINE('',#830,#172);
#850=PLANE('',#851);
#851=AXIS2_PLACEMENT_3D('',#830,#170,#171);
#900=MANIFOLD_SOLID_BREP('NONE',#901);
#901=CLOSED_SHELL('',(#910));
#910=ADVANCED_FACE('',(#911),#950,.T.);
#911=FACE_OUTER_BOUND('',#912,.T.);
#912=EDGE_LOOP('',(#913));
#913=ORIENTED_EDGE('',*,*,#914,.T.);
#914=EDGE_CURVE('',#920,#921,#960,.T.);
#920=VERTEX_POINT('',#930);
#921=VERTEX_POINT('',#931);
#930=CARTESIAN_POINT('',(0.,500.,0.));
#931=CARTESIAN_POINT('',(20.,500.,0.));
#960=LINE('',#930,#172);
#950=PLANE('',#951);
#951=AXIS2_PLACEMENT_3D('',#930,#170,#171);
"""


@pytest.fixture()
def step_path(tmp_path) -> Path:
    path = tmp_path / "solid_geometry.step"
    path.write_text(HEADER + SOLID_WITH_FAR_LINE_ORIGIN + TRAILER, encoding="utf-8")
    return path


def test_vertex_bbox_ignores_the_far_line_origin(step_path):
    model = parse_step(step_path)
    bbox = vertex_bbox(model, 100)
    assert bbox is not None
    assert bbox["size"] == [10.0, 0.0, 0.0]
    assert bbox["vertex_points"] == 2
    # the naive "all Cartesian points" extent would be catastrophic here
    assert max(p[0] for p in model.points) > 800000.0


def test_shape_hint_uses_two_ratios(step_path):
    assert shape_hint([100.0, 100.0, 10.0])["hint"] == SHAPE_DISC
    assert shape_hint([100.0, 90.0, 5.0])["hint"] == SHAPE_DISC
    assert shape_hint([100.0, 50.0, 5.0])["hint"] == SHAPE_ROD
    assert shape_hint([100.0, 100.0, 50.0])["hint"] == SHAPE_BLOCK
    assert shape_hint([100.0, 79.0, 30.0])["hint"] == SHAPE_SLAB
    assert shape_hint([0.0, 0.0, 0.0])["hint"] == "unknown"


def test_part_group_reads_the_module_from_an_assembly_path():
    name = (
        "M1.2-00V3 底盘模块-1/M1.2.2-00V3 舵轮前组件-2/"
        "M1.2.2.1-00V3 舵轮转动组件-2/减震器-1/减震器弹簧-1-solid1"
    )
    assert part_group(name) == "M1.2.2-00V3 舵轮前组件-2"
    # a flat path keeps its single root module
    assert part_group("M1.1-00V3 主体模块-1/GB╱T 818-2016[螺钉M3×8]-2-solid1") == "M1.1-00V3 主体模块-1"
    assert part_root(name) == "M1.2-00V3 底盘模块-1"
    assert part_group("NONE") == "UNKNOWN"
    assert part_group("") == "UNKNOWN"
    assert part_group("only-a-part") == "UNKNOWN"


def test_circle_facts_reports_radius_and_axis(step_path):
    model = parse_step(step_path)
    facts = circle_facts(model, 100)
    assert facts["circles"] == 1
    assert facts["circle_radius_mm"]["median"] == pytest.approx(5.5)
    assert facts["dominant_circle_axes"][0][0] == "(0,0,1)"


def test_solid_geometry_marks_named_and_unnamed_parts(step_path):
    model = parse_step(step_path)
    rows = {row["eid"]: row for row in solid_geometry(model)}
    assert rows[100]["named"] is True
    assert rows[100]["part_group"] == "M1.2.2-00V3 舵轮前组件-2"
    assert rows[900]["named"] is False
    assert rows[900]["part_group"] == "UNKNOWN"
    assert rows[900]["vertex_bbox_mm"]["size"] == [20.0, 0.0, 0.0]


def test_shape_fingerprint_is_translation_invariant(step_path):
    """The same shape at two positions must hash identically; a different shape must not."""
    model = parse_step(step_path)
    first = shape_fingerprint(model, 900)
    second = shape_fingerprint(model, 800)
    different = shape_fingerprint(model, 100)
    assert first["fingerprint"] == second["fingerprint"]
    assert first["vertices"] == second["vertices"] == 2
    assert different["fingerprint"] != first["fingerprint"]


def test_build_report_finds_repeated_shapes(step_path):
    model = parse_step(step_path)
    report = build_report(model)
    repeated = {item["fingerprint"]: item for item in report["identical_shape_groups"]}
    assert len(repeated) == 1
    only = list(repeated.values())[0]
    assert only["copies"] == 2
    assert sorted(only["solids"]) == [800, 900]


def test_build_report_aggregates_groups_and_caveats(step_path):
    model = parse_step(step_path)
    report = build_report(model)
    assert report["solids_total"] == 3
    assert report["solids_named"] == 1
    assert report["solids_unnamed"] == 2
    assert report["scene_vertex_bbox_mm"]["size"] == [70.0, 500.0, 0.0]
    groups = {group["part_group"]: group for group in report["groups"]}
    assert groups["M1.2.2-00V3 舵轮前组件-2"]["solids"] == 1
    assert groups["UNKNOWN"]["solids"] == 2
    assert len(report["caveats"]) == 3
    assert any("no wheel radius" in caveat for caveat in report["caveats"])
