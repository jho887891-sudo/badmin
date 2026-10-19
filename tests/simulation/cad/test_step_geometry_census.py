"""Unit tests for the kernel-free STEP census / dependency-graph tool.

These tests use a hand-written miniature STEP file so that they run anywhere and
can check parsing, the outlier band, dependency closure, attribution and subset
extraction without the 115 MB chassis model.  They deliberately do not assert
any property of the real model: those facts come from the recorded census JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.step_geometry_census import (  # noqa: E402
    Entity,
    attribute_points_to_solids,
    build_census,
    extract_subset,
    forward_closure,
    parse_step,
    point_statistics,
    reverse_index,
)

HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('mini.step','2026-10-20T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 2 1 1 }'));
ENDSEC;
DATA;
"""

SCAFFOLD_V1 = """#1=PRODUCT('MINI','MINI','',(#2));
#2=PRODUCT_CONTEXT('',#3,'mechanical');
#3=APPLICATION_CONTEXT('core data for automotive mechanical design processes');
#4=PRODUCT_DEFINITION_FORMATION('','',#1);
#5=PRODUCT_DEFINITION_CONTEXT('part definition',#3,'design');
#6=PRODUCT_DEFINITION('design','',#4,#5);
#7=PRODUCT_DEFINITION_SHAPE('','',#6);
#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#11=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));
#12=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());
#13=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#10,'distance_accuracy_value','');
#14=(GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNIT_ASSIGNED_CONTEXT((#10,#11,#12))
REPRESENTATION_CONTEXT('',''));
#20=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#100,#200),#14);
#21=SHAPE_DEFINITION_REPRESENTATION(#7,#20);
"""

# Solid #100 near the origin, solid #200 far away in +y, with one vertex that is
# a blatant spatial outlier so attribution has something unambiguous to prove.
SOLIDS = """#100=MANIFOLD_SOLID_BREP('near',#101);
#101=CLOSED_SHELL('',(#110));
#110=ADVANCED_FACE('',(#111),#150,.T.);
#111=FACE_OUTER_BOUND('',#112,.T.);
#112=EDGE_LOOP('',(#113));
#113=ORIENTED_EDGE('',*,*,#114,.T.);
#114=EDGE_CURVE('',#120,#121,#160,.T.);
#120=VERTEX_POINT('',#130);
#121=VERTEX_POINT('',#131);
#130=CARTESIAN_POINT('',(0.,0.,0.));
#131=CARTESIAN_POINT('',(10.,0.,0.));
#150=PLANE('',#151);
#151=AXIS2_PLACEMENT_3D('',#130,#170,#171);
#160=LINE('',#130,#172);
#170=DIRECTION('',(0.,0.,1.));
#171=DIRECTION('',(1.,0.,0.));
#172=VECTOR('',#173,1.);
#173=DIRECTION('',(1.,0.,0.));
#200=MANIFOLD_SOLID_BREP('far',#201);
#201=CLOSED_SHELL('',(#210));
#210=ADVANCED_FACE('',(#211),#250,.T.);
#211=FACE_OUTER_BOUND('',#212,.T.);
#212=EDGE_LOOP('',(#213));
#213=ORIENTED_EDGE('',*,*,#214,.T.);
#214=EDGE_CURVE('',#220,#221,#260,.T.);
#220=VERTEX_POINT('',#230);
#221=VERTEX_POINT('',#231);
#230=CARTESIAN_POINT('',(0.,5000.,0.));
#231=CARTESIAN_POINT('',(10.,5000.,0.));
#250=PLANE('',#251);
#251=AXIS2_PLACEMENT_3D('',#230,#270,#271);
#260=LINE('',#230,#272);
#270=DIRECTION('',(0.,0.,1.));
#271=DIRECTION('',(1.,0.,0.));
#272=VECTOR('',#273,1.);
#273=DIRECTION('',(1.,0.,0.));
"""

TRAILER = """ENDSEC;
END-ISO-10303-21;
"""


@pytest.fixture()
def mini_step(tmp_path) -> Path:
    path = tmp_path / "mini.step"
    path.write_text(HEADER + SCAFFOLD_V1 + SOLIDS + TRAILER, encoding="utf-8")
    return path


def test_parse_step_reads_entities_refs_and_points(mini_step):
    model = parse_step(mini_step)
    assert model.file_schema.startswith("AUTOMOTIVE_DESIGN")
    assert model.parse_warnings == []
    assert model.entities[114].etype == "EDGE_CURVE"
    assert model.entities[114].refs == (120, 121, 160)
    assert model.entities[20].etype == "ADVANCED_BREP_SHAPE_REPRESENTATION"
    assert model.point_ids == [130, 131, 230, 231]
    assert model.points[1] == (10.0, 0.0, 0.0)
    assert model.points[2] == (0.0, 5000.0, 0.0)


def test_parse_step_keeps_multiline_records_together(mini_step):
    model = parse_step(mini_step)
    # #14 spans three physical lines; it must still be one entity.
    assert model.entities[14].etype == "GEOMETRIC_REPRESENTATION_CONTEXT"
    assert 10 in model.entities[14].refs


def test_census_reports_millimetre_units_and_products(mini_step):
    census = build_census(parse_step(mini_step))
    assert census["units"]["length_unit_is_millimetre"] is True
    assert census["units"]["length_unit_metre"] is True
    assert census["products"]["products"][0]["name"] == "MINI"
    assert census["entity_total"] == len(parse_step(mini_step).entities)
    assert census["points"]["count"] == 4


def test_census_is_deterministic(mini_step):
    first = build_census(parse_step(mini_step))
    second = build_census(parse_step(mini_step))
    first.pop("_outlier_indices")
    second.pop("_outlier_indices")
    assert first == second


def test_point_statistics_flags_only_the_far_point():
    points = [(float(i), 0.0, 0.0) for i in range(10)] + [(0.0, 9000.0, 0.0)]
    bands, outliers = point_statistics(points, low_percentile=10.0, high_percentile=90.0)
    assert bands[1].axis == "y"
    assert outliers == [10]
    assert bands[0].allows(4.0) and not bands[1].allows(9000.0)


def test_point_statistics_rejects_bad_percentiles():
    with pytest.raises(ValueError):
        point_statistics([(0.0, 0.0, 0.0)], low_percentile=90.0, high_percentile=10.0)


def test_point_statistics_on_empty_input():
    bands, outliers = point_statistics([])
    assert bands == [] and outliers == []


def test_forward_closure_does_not_leak_between_solids(mini_step):
    model = parse_step(mini_step)
    near = forward_closure(model.entities, [100])
    far = forward_closure(model.entities, [200])
    assert {100, 101, 110, 114, 120, 130} <= near
    assert near.isdisjoint({200, 201, 210, 214, 220, 230})
    assert far.isdisjoint({100, 101, 110})


def test_reverse_index_links_point_to_vertex_point(mini_step):
    model = parse_step(mini_step)
    reverse = reverse_index(model.entities)
    # the point is shared by its vertex_point, the placement axis and the line
    assert sorted(reverse[130]) == [120, 151, 160]
    assert reverse[131] == [121]


def test_attribution_proves_the_owning_solid(mini_step):
    model = parse_step(mini_step)
    attribution = attribute_points_to_solids(model.entities, [130, 230])
    assert attribution[130] == [100]
    assert attribution[230] == [200]


def test_extract_subset_keeps_only_requested_solid(tmp_path, mini_step):
    model = parse_step(mini_step)
    text, stats = extract_subset(model, [100])
    out = tmp_path / "subset.step"
    out.write_text(text, encoding="utf-8")

    subset = parse_step(out)
    assert stats["requested_solids"] == [100]
    assert 100 in subset.entities and 200 not in subset.entities
    assert 130 in subset.entities
    # scaffolding survived so the reduced file is still a valid product model
    assert {1, 2, 3, 6, 7, 10, 14, 20, 21} <= set(subset.entities)
    # the representation must no longer advertise the dropped solid
    assert 200 not in subset.entities[20].refs
    assert 100 in subset.entities[20].refs
    assert subset.point_ids == [130, 131]
    # a reduced file must not reference entities it dropped
    assert stats["dangling_references"] == 0
    assert stats["rewritten_item_lists"] == 1


def test_extract_subset_pulls_world_placement_from_item_list(tmp_path):
    """A placement that only appears in the item list must still be exported.

    The real chassis file puts an AXIS2_PLACEMENT_3D inside the representation's
    item list; requiring every reference up front skipped the whole
    representation and produced a geometry-free subset.
    """
    text = (HEADER + SCAFFOLD_V1 + SOLIDS + TRAILER).replace(
        "#20=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#100,#200),#14);",
        "#152=AXIS2_PLACEMENT_3D('',#133,#170,#171);\n"
        "#133=CARTESIAN_POINT('',(0.,0.,0.));\n"
        "#20=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#100,#200,#152),#14);",
    )
    path = tmp_path / "mini_placement.step"
    path.write_text(text, encoding="utf-8")

    subset_text, stats = extract_subset(parse_step(path), [100])
    out = tmp_path / "subset_placement.step"
    out.write_text(subset_text, encoding="utf-8")
    subset = parse_step(out)

    assert {20, 133, 152} <= set(subset.entities)
    assert 200 not in subset.entities
    assert 152 in subset.entities[20].refs
    assert 200 not in subset.entities[20].refs
    assert stats["dangling_references"] == 0


def test_extract_subset_rejects_unknown_and_non_solid_ids(mini_step):
    model = parse_step(mini_step)
    with pytest.raises(KeyError):
        extract_subset(model, [999999])
    with pytest.raises(TypeError):
        extract_subset(model, [130])


def test_entity_to_record_roundtrips():
    entity = Entity(eid=7, etype="DIRECTION", body="'',(0.,1.,0.)", refs=())
    assert entity.to_record() == "#7=DIRECTION('',(0.,1.,0.));"
