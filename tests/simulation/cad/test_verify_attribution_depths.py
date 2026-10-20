"""Tests for the attribution depth verifier.

The fixture deliberately shares one AXIS2_PLACEMENT_3D between the faces of two
different solids, so the shared point reaches one solid at a shallower depth than
the other.  That is exactly the situation the pre-audit early stop could not see,
and it lets these tests prove the verifier detects the difference instead of
merely re-running the fixed code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.step_geometry_census import parse_step, reverse_index  # noqa: E402
from scripts.simulation.verify_attribution_depths import owners_by_level, verify  # noqa: E402

HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('depth.step','2026-10-20T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#14=(GEOMETRIC_REPRESENTATION_CONTEXT(3)REPRESENTATION_CONTEXT('',''));
#20=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#100,#600),#14);
"""

TRAILER = """ENDSEC;
END-ISO-10303-21;
"""

# Solid 100 reaches point #130 through a vertex (deep path).
# Solid 600 reaches the same point through the shared placement #151 (shallower).
BODY = """#100=MANIFOLD_SOLID_BREP('near',#101);
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
#600=MANIFOLD_SOLID_BREP('far',#601);
#601=CLOSED_SHELL('',(#610));
#610=ADVANCED_FACE('',(#611),#151,.T.);
#611=FACE_OUTER_BOUND('',#612,.T.);
#612=EDGE_LOOP('',(#613));
#613=ORIENTED_EDGE('',*,*,#614,.T.);
#614=EDGE_CURVE('',#220,#221,#160,.T.);
#220=VERTEX_POINT('',#230);
#221=VERTEX_POINT('',#231);
#230=CARTESIAN_POINT('',(0.,100.,0.));
#231=CARTESIAN_POINT('',(10.,100.,0.));
"""


@pytest.fixture()
def shared_placement_step(tmp_path) -> Path:
    path = tmp_path / "depth.step"
    path.write_text(HEADER + BODY + TRAILER, encoding="utf-8")
    return path


def test_owners_are_grouped_by_depth(shared_placement_step):
    model = parse_step(shared_placement_step)
    reverse = reverse_index(model.entities)
    levels = owners_by_level(model.entities, reverse, 130)
    assert levels, "expected at least one owning solid"
    depths = [depth for depth, _ in levels]
    assert depths == sorted(depths)
    # the shared placement reaches solid 600 before the vertex chain reaches 100
    assert levels[0][1] == [600]
    assert any(100 in owners for _, owners in levels)
    assert len(levels) >= 2


def test_verifier_flags_the_point_the_early_stop_would_have_truncated(shared_placement_step):
    model = parse_step(shared_placement_step)
    report = verify(model, [130])
    assert report["points_checked"] == 1
    assert report["legacy_solids"] == 1
    assert report["full_solids"] == 2
    assert report["points_where_early_stop_would_lose_an_owner"] == 1
    difference = report["differences"][0]
    assert difference["point_id"] == 130
    assert difference["legacy_owners"] == [600]
    assert difference["full_owners"] == [100, 600]
    # a real difference must come with entity-level evidence
    chains = difference["classification"]["chains"]
    assert chains
    assert all(chain[0]["eid"] == 130 for chain in chains)
    assert all(chain[-1]["type"].startswith(("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")) for chain in chains)


def test_verifier_reports_no_difference_for_a_single_owner_point(shared_placement_step):
    model = parse_step(shared_placement_step)
    report = verify(model, [131])
    assert report["points_checked"] == 1
    assert report["points_where_early_stop_would_lose_an_owner"] == 0
    assert report["legacy_histogram"] == report["full_histogram"]
