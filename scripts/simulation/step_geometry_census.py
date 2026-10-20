"""Kernel-free, reproducible census and dependency analysis of a STEP (ISO 10303-21) file.

Why this module exists
----------------------
The Morph One M1-4WD4S chassis STEP file is a ~115 MB AP203 text file and the
analysis machine has no CAD kernel.  We must be able to answer "what is in this
model?" and "which solid does a suspicious point belong to?" *without* editing
the model.  Deleting or snapping points would silently break BREP topology, so
this module never rewrites the input; it only reads, and can *extract* a
self-consistent subset of solids for downstream smoke tests.

What it computes
----------------
* mode "census"    - entity-type census, header/product facts, unit declarations.
* mode "points"    - Cartesian-point statistics and a *candidate* outlier band.
* mode "attribute" - reverse dependency graph: for every candidate outlier point,
                     the solids (MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS) that
                     reference it, i.e. proof of ownership instead of a guess.
* mode "subset"    - emit a smaller STEP that keeps only selected solids plus the
                     product/unit scaffolding, so the file still parses on its own.

An outlier stays a *candidate* everywhere in this code: the band rule is a
property of the text, not a geometric validity proof.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

__all__ = [
    "Entity",
    "StepModel",
    "parse_step",
    "point_statistics",
    "OutlierBand",
    "forward_closure",
    "reverse_index",
    "attribute_points_to_solids",
    "scaffold_types",
    "is_scaffold",
    "split_item_and_metadata_refs",
    "extract_subset",
    "decode_step_name",
    "entity_name",
    "walk_up_paths",
    "collect_ancestors",
    "classify_point",
    "POINT_KIND_BSPLINE_CONTROL",
    "POINT_KIND_BREP_VERTEX",
    "POINT_KIND_AUXILIARY",
    "POINT_KIND_UNKNOWN",
]

# Entity types that carry the product/structure chain rather than raw geometry.
# They are kept by extract_subset so that a reduced file is still valid.
_SCAFFOLD_TYPES: frozenset = frozenset(
    {
        "PRODUCT",
        "PRODUCT_RELATED_PRODUCT_CATEGORY",
        "PRODUCT_DEFINITION_FORMATION",
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
        "PRODUCT_DEFINITION",
        "PRODUCT_DEFINITION_WITH_ASSOCIATED_DOCUMENTS",
        "PRODUCT_DEFINITION_SHAPE",
        "SHAPE_DEFINITION_REPRESENTATION",
        "SHAPE_REPRESENTATION",
        "SHAPE_REPRESENTATION_RELATIONSHIP",
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        "GEOMETRIC_REPRESENTATION_CONTEXT",
        "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        "UNCERTAINTY_MEASURE_WITH_UNIT",
        "DIMENSIONAL_EXPONENTS",
        "NAMED_UNIT",
        "SI_UNIT",
        "LENGTH_UNIT",
        "CONVERSION_BASED_UNIT",
        "MEASURE_WITH_UNIT",
        "MEASURE_REPRESENTATION_ITEM",
    }
)

# Bulk BREP containers.  Scaffolding references these through an items list;
# when we reduce a file, those lists are rewritten instead of being followed.
_BULK_GEOMETRY_TYPES: frozenset = frozenset(
    {
        "MANIFOLD_SOLID_BREP",
        "BREP_WITH_VOIDS",
        "CLOSED_SHELL",
        "OPEN_SHELL",
        "ORIENTED_CLOSED_SHELL",
    }
)

SOLID_TYPES: frozenset = frozenset({"MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"})

_RECORD_HEAD_RE = re.compile(r"#(\d+)\s*=\s*")
_SIMPLE_TYPE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_COMPONENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_REF_RE = re.compile(r"#(\d+)")
# STEP reals may be written as "0.", "1.5", ".5", "1.E-07" or "1.0E+02".
_NUMBER = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[EeDd][-+]?[0-9]+)?"
_COORD_TRIPLE_RE = re.compile(
    r"\(\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*\)"
)


@dataclass(frozen=True)
class Entity:
    """One '#id = TYPE(body);' record.

    STEP also allows *complex instances*, written '#id=(A(..)B(..));', where a
    single record belongs to several types at once.  Real SolidWorks exports use
    them for the representation context and for units, so the parser keeps the
    full component list instead of pretending they are unknown records.
    """

    eid: int
    etype: str
    body: str
    refs: Tuple[int, ...]
    components: Tuple[str, ...] = ()
    is_complex: bool = False

    def to_record(self, body: "str | None" = None) -> str:
        payload = self.body if body is None else body
        if self.is_complex:
            return "#%d=(%s);" % (self.eid, payload)
        return "#%d=%s(%s);" % (self.eid, self.etype, payload)


@dataclass
class StepModel:
    path: Path
    size_bytes: int
    header: str
    trailer: str
    entities: Dict[int, Entity]
    point_ids: List[int] = field(default_factory=list)
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    file_schema: str = ""
    parse_warnings: List[str] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(e.etype for e in self.entities.values())

    def complex_types(self) -> Counter:
        return Counter(
            "+".join(e.components) for e in self.entities.values() if e.is_complex
        )

    def of_type(self, etype: str) -> List[Entity]:
        return [e for e in self.entities.values() if etype in e.components]

    def has_type(self, entity: Entity, types: frozenset) -> bool:
        return bool(set(entity.components) & types)


def _scan_records(body: str) -> Iterable[str]:
    """Yield raw '#id=...;' records, tolerating multi-line records."""
    pos = 0
    length = len(body)
    while True:
        start = body.find("#", pos)
        if start < 0:
            return
        depth = 0
        i = start
        while i < length:
            ch = body[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth <= 0:
                yield body[start : i + 1]
                pos = i + 1
                break
            i += 1
        else:  # pragma: no cover - truncated file
            return


def _component_names(body: str) -> Tuple[str, ...]:
    """Names of the sub-types of a complex instance body: A(..)B(..) -> (A, B)."""
    names: List[str] = []
    index = 0
    while True:
        match = _COMPONENT_RE.search(body, index)
        if match is None:
            return tuple(names)
        names.append(match.group(1).upper())
        depth = 1
        cursor = match.end()
        while cursor < len(body) and depth:
            if body[cursor] == "(":
                depth += 1
            elif body[cursor] == ")":
                depth -= 1
            cursor += 1
        index = cursor


def _parse_record(record: str) -> "Entity | None":
    """Split a raw record into id, component types and the raw argument text."""
    head = _RECORD_HEAD_RE.match(record)
    if head is None:
        return None
    eid = int(head.group(1))
    text = record[head.end() :].rstrip()
    if text.endswith(";"):
        text = text[:-1]
    if not text:
        return None

    if text.startswith("(") and text.endswith(")"):
        body = text[1:-1]
        components = _component_names(body)
        if not components:
            return None
        return Entity(
            eid=eid,
            etype=components[0],
            body=body,
            refs=tuple(int(r) for r in _REF_RE.findall(body)),
            components=components,
            is_complex=True,
        )

    simple = _SIMPLE_TYPE_RE.match(text)
    if simple is None:
        return None
    name = simple.group(1).upper()
    body = text[simple.end() :]
    if body.endswith(")"):
        body = body[:-1]
    return Entity(
        eid=eid,
        etype=name,
        body=body,
        refs=tuple(int(r) for r in _REF_RE.findall(body)),
        components=(name,),
        is_complex=False,
    )


def parse_step(path: "str | Path") -> StepModel:
    """Parse a STEP file into an entity map without any CAD kernel.

    The reader is deliberately forgiving: unrecognised records are collected in
    parse_warnings instead of aborting, because the goal is a census of a
    real-world export, not schema validation.
    """
    path = Path(path)
    raw = path.read_bytes()
    # ISO 10303-21 is ASCII; latin-1 keeps every byte addressable (the PRODUCT
    # name in the Morph One export is mojibake, so strict decoding would fail).
    text = raw.decode("latin-1")
    data_at = text.find("DATA;")
    if data_at < 0:
        raise ValueError("no DATA; section found in %s" % path)
    header = text[: data_at + len("DATA;")]
    rest = text[data_at + len("DATA;") :]
    tail_at = rest.rfind("ENDSEC;")
    if tail_at < 0:
        raise ValueError("no closing ENDSEC; found in %s" % path)
    body = rest[:tail_at]
    trailer = rest[tail_at:]

    model = StepModel(
        path=path,
        size_bytes=len(raw),
        header=header,
        trailer=trailer,
        entities={},
    )
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", header)
    model.file_schema = schema_match.group(1) if schema_match else ""

    for record in _scan_records(body):
        entity = _parse_record(record)
        if entity is None:
            model.parse_warnings.append("unparsed record: %r" % record[:80])
            continue
        model.entities[entity.eid] = entity
        if entity.etype == "CARTESIAN_POINT":
            coords = _extract_point(entity.body)
            if coords is None:
                model.parse_warnings.append(
                    "no coords for CARTESIAN_POINT #%d" % entity.eid
                )
                continue
            model.point_ids.append(entity.eid)
            model.points.append(coords)
    return model


def _extract_point(body: str) -> "Tuple[float, float, float] | None":
    matches = _COORD_TRIPLE_RE.findall(body)
    if not matches:
        return None
    x, y, z = matches[-1]
    return (
        float(x.replace("D", "E").replace("d", "e")),
        float(y.replace("D", "E").replace("d", "e")),
        float(z.replace("D", "E").replace("d", "e")),
    )


@dataclass(frozen=True)
class OutlierBand:
    """Per-axis acceptance band derived from robust percentiles."""

    axis: str
    low_percentile: float
    high_percentile: float
    low_value: float
    high_value: float
    span: float
    allowed_low: float
    allowed_high: float

    def allows(self, value: float) -> bool:
        return self.allowed_low <= value <= self.allowed_high


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile on an already sorted sequence."""
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def point_statistics(
    points: Sequence[Tuple[float, float, float]],
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
    widen_fraction: float = 0.5,
) -> "Tuple[List[OutlierBand], List[int]]":
    """Return per-axis bands and the indices of *candidate* outlier points.

    The band is [p_low, p_high] widened by widen_fraction of its own span on
    both sides.  A point is a candidate when it falls outside the band on at
    least one axis.  A candidate is never assumed invalid.
    """
    if not points:
        return [], []
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    bands: List[OutlierBand] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        values = sorted(p[axis_index] for p in points)
        lo = _percentile(values, low_percentile)
        hi = _percentile(values, high_percentile)
        span = hi - lo
        pad = span * widen_fraction
        bands.append(
            OutlierBand(
                axis=axis_name,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                low_value=lo,
                high_value=hi,
                span=span,
                allowed_low=lo - pad,
                allowed_high=hi + pad,
            )
        )

    outliers: List[int] = []
    for index, point in enumerate(points):
        if not all(band.allows(point[i]) for i, band in enumerate(bands)):
            outliers.append(index)
    return bands, outliers


def forward_closure(
    entities: Dict[int, Entity],
    seeds: Iterable[int],
    prune_types: "frozenset | None" = None,
) -> Set[int]:
    """All entity ids transitively referenced by seeds (including the seeds).

    prune_types stops traversal *through* entities of those types: the entity is
    listed but its own references are not followed.  Subset extraction uses it so
    that following a product-chain reference to a shape representation cannot
    drag every solid of the assembly back in.
    """
    seeds = list(seeds)
    seen: Set[int] = set()
    stack = list(seeds)
    while stack:
        eid = stack.pop()
        if eid in seen:
            continue
        entity = entities.get(eid)
        if entity is None:
            continue
        if prune_types is not None and eid not in seeds:
            if set(entity.components) & prune_types:
                # Listed nowhere: the caller prunes whole bulk containers, and a
                # pruned container must not leave its id dangling in the output.
                continue
        seen.add(eid)
        stack.extend(entity.refs)
    return seen


def reverse_index(
    entities: Dict[int, Entity], restrict_types: "Set[str] | None" = None
) -> Dict[int, List[int]]:
    """Map referenced id -> [ids that reference it].

    restrict_types limits which referencing entities are indexed, which keeps
    attribution cheap when only structural types matter.
    """
    index: Dict[int, List[int]] = defaultdict(list)
    for eid, entity in entities.items():
        if restrict_types is not None and entity.etype not in restrict_types:
            continue
        for ref in entity.refs:
            index[ref].append(eid)
    return index


def attribute_points_to_solids(
    entities: Dict[int, Entity],
    point_ids: Sequence[int],
    reverse: "Dict[int, List[int]] | None" = None,
    max_hops: int = 32,
) -> Dict[int, List[int]]:
    """Map each point id to the ids of solids that (transitively) reference it.

    Walking the *reverse* graph is what makes this a proof of ownership: a point
    can only be blamed on a solid that actually includes it through
    vertex -> edge -> loop -> face -> shell -> solid.
    """
    if reverse is None:
        reverse = reverse_index(entities)
    result: Dict[int, List[int]] = {}
    for point_id in point_ids:
        owners: Set[int] = set()
        frontier = [point_id]
        visited: Set[int] = {point_id}
        for _ in range(max_hops):
            next_frontier: List[int] = []
            for eid in frontier:
                for parent in reverse.get(eid, ()):  # type: ignore[arg-type]
                    if parent in visited:
                        continue
                    visited.add(parent)
                    entity = entities.get(parent)
                    if entity is None:
                        continue
                    if set(entity.components) & SOLID_TYPES:
                        owners.add(parent)
                        continue
                    next_frontier.append(parent)
            # Finding one owner does NOT mean we found all owners: a point
            # can reach different solids through paths of different lengths.
            # Keep searching other branches, stopping each branch at its solid.
            if not next_frontier:
                break
            frontier = next_frontier
        result[point_id] = sorted(owners)
    return result


# Complex-instance components that own a control_points_list.  A point that is a
# control point of a polynomial curve or surface is NOT a point on the surface,
# so a "far away" coordinate there says nothing about the part's real extent.
_BSPLINE_TYPES: frozenset = frozenset(
    {
        "B_SPLINE_CURVE_WITH_KNOTS",
        "B_SPLINE_CURVE",
        "RATIONAL_B_SPLINE_CURVE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
        "B_SPLINE_SURFACE",
        "RATIONAL_B_SPLINE_SURFACE",
        "BEZIER_CURVE",
        "BEZIER_SURFACE",
        "UNIFORM_CURVE",
        "UNIFORM_SURFACE",
    }
)

# Structural types whose presence in an upward path tells us how a point is used.
_VERTEX_TYPES: frozenset = frozenset({"VERTEX_POINT"})
_EDGE_TYPES: frozenset = frozenset({"EDGE_CURVE"})
_FACE_TYPES: frozenset = frozenset({"ADVANCED_FACE", "FACE_SURFACE"})

POINT_KIND_BSPLINE_CONTROL = "bspline_control_point"
POINT_KIND_BREP_VERTEX = "brep_vertex"
POINT_KIND_AUXILIARY = "auxiliary_geometry"
POINT_KIND_UNKNOWN = "unknown"


def decode_step_name(raw: "str | bytes") -> str:
    """Decode a STEP name that was exported from a GBK (Chinese) SolidWorks.

    The Morph One export stores GBK bytes inside STEP strings, so a latin-1 read
    shows mojibake such as "µ×ÅÌÄ£¿é" instead of "底盘模块".  Decoding through
    GB18030 recovers the designer's part names, which are legitimate evidence of
    intent (not of measured geometry).
    """
    if isinstance(raw, bytes):
        candidates = [raw]
    else:
        candidates = []
        try:
            candidates.append(raw.encode("latin-1"))
        except UnicodeEncodeError:
            pass
        candidates.append(raw.encode("utf-8", "ignore"))
    for blob in candidates:
        for encoding in ("gb18030", "utf-8"):
            try:
                text = blob.decode(encoding)
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
            if text and not any(ch == "\ufffd" for ch in text):
                return text
    return raw if isinstance(raw, str) else raw.decode("latin-1", "ignore")


def entity_name(entity: Entity) -> str:
    """First string argument of an entity body, decoded from GBK when needed."""
    body = entity.body.lstrip()
    if not body.startswith("'"):
        return ""
    end = body.find("'", 1)
    if end < 0:
        return ""
    return decode_step_name(body[1:end])


def walk_up_paths(
    entities: Dict[int, Entity],
    point_id: int,
    reverse: Dict[int, List[int]],
    max_hops: int = 40,
    max_paths: int = 2000,
) -> List[List[int]]:
    """All upward reference paths from a point to the solids that own it.

    Returning whole paths (not just the owner set) is what makes it possible to
    say *how* a point is attached: as a BREP vertex of an edge, or as a control
    point of a B-spline the point never touches.
    """
    paths: List[List[int]] = []
    stack: List[List[int]] = [[point_id]]
    while stack and len(paths) < max_paths:
        path = stack.pop()
        head = path[-1]
        entity = entities.get(head)
        if entity is not None and head != point_id and (set(entity.components) & SOLID_TYPES):
            paths.append(path)
            continue
        if len(path) > max_hops:
            continue
        parents = [p for p in reverse.get(head, ()) if p not in path]
        # Breadth of the upward graph can explode on shared points; keep the
        # walk bounded and deterministic.
        for parent in sorted(parents, reverse=True):
            stack.append(path + [parent])
    return paths


def collect_ancestors(
    entities: Dict[int, Entity],
    reverse: Dict[int, List[int]],
    point_id: int,
    max_nodes: int = 200000,
) -> dict:
    """Walk every upward reference from a point and bucket the ancestors by role.

    The walk deliberately continues *past* nothing and stops *at* solids: a
    control point of a B-spline that no solid references must still be explained
    (it is geometry, just not BREP geometry), otherwise it would look like an
    orphan and be silently mislabelled.
    """
    visited: Set[int] = {point_id}
    parent_of: Dict[int, int] = {}
    nearest_solid: "int | None" = None
    frontier = [point_id]
    faces: Set[int] = set()
    edges: Set[int] = set()
    vertices: Set[int] = set()
    bsplines: Set[int] = set()
    solids: Set[int] = set()
    seen_nodes = 0
    while frontier and seen_nodes < max_nodes:
        nxt: List[int] = []
        for eid in frontier:
            for parent in reverse.get(eid, ()):  # type: ignore[arg-type]
                if parent in visited:
                    continue
                visited.add(parent)
                parent_of[parent] = eid
                entity = entities.get(parent)
                if entity is None:
                    continue
                components = set(entity.components)
                if components & SOLID_TYPES:
                    solids.add(parent)
                    if nearest_solid is None:
                        nearest_solid = parent
                    continue
                if components & _FACE_TYPES:
                    faces.add(parent)
                if components & _EDGE_TYPES:
                    edges.add(parent)
                if components & _VERTEX_TYPES:
                    vertices.add(parent)
                if components & _BSPLINE_TYPES:
                    bsplines.add(parent)
                nxt.append(parent)
        seen_nodes += len(nxt)
        frontier = nxt
    shortest_path: List[int] = []
    if nearest_solid is not None:
        cursor = nearest_solid
        while True:
            shortest_path.append(cursor)
            if cursor == point_id:
                break
            previous = parent_of.get(cursor)
            if previous is None:
                break
            cursor = previous
        shortest_path.reverse()

    return {
        "solids": solids,
        "faces": faces,
        "edges": edges,
        "vertices": vertices,
        "bsplines": bsplines,
        "ancestors_visited": len(visited),
        "shortest_path": shortest_path,
    }


def classify_point(
    entities: Dict[int, Entity], reverse: Dict[int, List[int]], point_id: int
) -> dict:
    """Explain what a Cartesian point is used for, with entity-level evidence."""
    buckets = collect_ancestors(entities, reverse, point_id)
    shortest_path: List[int] = buckets["shortest_path"]
    solids = buckets["solids"]
    faces = buckets["faces"]
    edges = buckets["edges"]
    vertices = buckets["vertices"]
    bsplines = buckets["bsplines"]

    direct_parent_types = sorted(
        {
            entities[parent].etype
            for parent in reverse.get(point_id, ())
            if parent in entities
        }
    )
    ancestor_types: List[str] = []
    for group in (vertices, edges, faces, bsplines, solids):
        for eid in sorted(group):
            entity = entities.get(eid)
            if entity is not None and entity.etype not in ancestor_types:
                ancestor_types.append(entity.etype)

    if vertices:
        kind = POINT_KIND_BREP_VERTEX
    elif bsplines:
        kind = POINT_KIND_BSPLINE_CONTROL
    elif solids or edges or faces:
        kind = POINT_KIND_AUXILIARY
    else:
        kind = POINT_KIND_UNKNOWN

    return {
        "point_id": point_id,
        "kind": kind,
        "solid_count": len(solids),
        "solid_ids": sorted(solids)[:8],
        "face_count": len(faces),
        "face_ids_sample": sorted(faces)[:8],
        "edge_count": len(edges),
        "edge_ids_sample": sorted(edges)[:8],
        "vertex_point_ids": sorted(vertices)[:8],
        "bspline_ids_sample": sorted(bsplines)[:8],
        "bspline_signatures": sorted(
            {
                "+".join(entities[eid].components)
                for eid in bsplines
                if eid in entities
            }
        )[:4],
        "direct_parent_types": direct_parent_types,
        "direct_parent_count": len(reverse.get(point_id, ())),
        "ancestor_types": ancestor_types[:10],
        "ancestors_visited": buckets["ancestors_visited"],
        "shortest_path_types": [entities[e].etype for e in shortest_path if e in entities],
        "shortest_path_ids": shortest_path,
    }


def scaffold_types() -> frozenset:
    return _SCAFFOLD_TYPES


def is_scaffold(entity: Entity) -> bool:
    """True when an entity belongs to the product/unit chain, not to raw geometry.

    Any '*_CONTEXT' component counts: STEP product chains always route through
    APPLICATION_CONTEXT / PRODUCT_CONTEXT / PRODUCT_DEFINITION_CONTEXT, and
    missing one of them silently truncates the product structure of a subset.
    """
    for component in entity.components:
        if component in _SCAFFOLD_TYPES or component.endswith("_CONTEXT"):
            return True
    return False


def _split_top_level_args(body: str) -> List[Tuple[int, int]]:
    """Character spans of the comma-separated arguments of a record body."""
    spans: List[Tuple[int, int]] = []
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(body)))
    return spans


def split_item_and_metadata_refs(
    body: str, entities: Dict[int, Entity]
) -> Tuple[List[int], List[int]]:
    """Separate references that live in a positional item list from the rest.

    A representation record looks like TYPE('name',(#item,#item),#context): the
    ids inside the parenthesised argument are children that may legitimately be
    dropped when a subset keeps only some of them, while the remaining ids are
    metadata (context, definition) that must already be present.  Confusing the
    two is what silently produced geometry-free subsets: the world placement
    axis lives only inside the item list, so requiring it up front never
    succeeded and the whole representation was skipped.
    """
    item_refs: List[int] = []
    metadata_refs: List[int] = []
    for start, end in _split_top_level_args(body):
        argument = body[start:end]
        stripped = argument.strip()
        target = item_refs if stripped.startswith("(") and stripped.endswith(")") else metadata_refs
        for ref in _REF_RE.findall(stripped):
            value = int(ref)
            if value in entities:
                target.append(value)
    return item_refs, metadata_refs


def _rewrite_item_lists(
    body: str, kept: Set[int], entities: Dict[int, Entity]
) -> str:
    """Drop references to non-kept solids from any positional item list.

    STEP representations carry their children as a positional argument, e.g.
    ADVANCED_BREP_SHAPE_REPRESENTATION('name',(#1,#2),#context); there is no
    'items=' keyword in the file, that is an importer-side spelling.  Only a
    list that actually contains bulk geometry is touched, so unit or placement
    arguments are never rewritten.
    """
    out = body
    for start, end in reversed(_split_top_level_args(body)):
        argument = body[start:end]
        stripped = argument.strip()
        if not (stripped.startswith("(") and stripped.endswith(")")):
            continue
        refs = [int(r) for r in _REF_RE.findall(stripped)]
        if not refs:
            continue
        bulk = [
            r
            for r in refs
            if r in entities and set(entities[r].components) & _BULK_GEOMETRY_TYPES
        ]
        if not bulk:
            continue
        kept_refs = [r for r in refs if r in kept]
        if len(kept_refs) == len(refs):
            continue
        lead = argument[: len(argument) - len(argument.lstrip())]
        trail = argument[len(argument.rstrip()) :]
        replacement = lead + "(" + ",".join("#%d" % r for r in kept_refs) + ")" + trail
        out = out[:start] + replacement + out[end:]
    return out


def extract_subset(model: StepModel, solid_ids: Sequence[int]) -> "Tuple[str, dict]":
    """Build a self-contained STEP text containing only solid_ids.

    The result keeps the original header and the product/unit scaffolding, and
    rewrites every items list so it only points at the retained solids.  No
    entity inside a retained solid is dropped, so its BREP stays untouched.
    """
    missing = [sid for sid in solid_ids if sid not in model.entities]
    if missing:
        raise KeyError("unknown entity ids: %s" % missing)
    wrong_type = [
        sid for sid in solid_ids if not (set(model.entities[sid].components) & SOLID_TYPES)
    ]
    if wrong_type:
        raise TypeError(
            "not a solid: %s" % [(sid, model.entities[sid].etype) for sid in wrong_type]
        )

    keep: Set[int] = forward_closure(model.entities, solid_ids)

    # Pull in scaffolding (product chain, units, placement contexts) but never
    # follow scaffolding into bulk geometry we did not ask for.
    for _ in range(64):
        added = False
        for eid, entity in model.entities.items():
            if eid in keep or not is_scaffold(entity):
                continue
            # Metadata references (context, definition, ...) must already be
            # present; item-list references are pulled in when they are not bulk
            # geometry, and the bulk ones are pruned and rewritten below.
            item_refs, metadata_refs = split_item_and_metadata_refs(
                entity.body, model.entities
            )
            if any(r not in keep for r in metadata_refs):
                continue
            seeds = [
                r
                for r in item_refs
                if model.entities[r].etype not in _BULK_GEOMETRY_TYPES
            ] + metadata_refs
            keep.add(eid)
            keep |= forward_closure(
                model.entities, seeds, prune_types=_BULK_GEOMETRY_TYPES
            )
            added = True
        if not added:
            break

    lines: List[str] = []
    rewritten = 0
    emitted_refs: Dict[int, Tuple[int, ...]] = {}
    for eid in sorted(keep):
        entity = model.entities[eid]
        body = entity.body
        if not entity.is_complex and "(" in body:
            new_body = _rewrite_item_lists(body, keep, model.entities)
            if new_body != body:
                rewritten += 1
            body = new_body
        emitted_refs[eid] = tuple(int(r) for r in _REF_RE.findall(body))
        lines.append(entity.to_record(body))

    text = model.header + "\n" + "\n".join(lines) + "\n" + model.trailer

    # Self-check: a subset with references to undefined ids would parse here but
    # fail in any CAD reader, so report it instead of hoping.
    dangling: List[Tuple[int, int]] = []
    for eid in sorted(keep):
        for ref in emitted_refs[eid]:
            if ref not in keep:
                dangling.append((eid, ref))

    stats = {
        "requested_solids": list(solid_ids),
        "kept_entities": len(lines),
        "rewritten_item_lists": rewritten,
        "bytes": len(text),
        "dangling_references": len(dangling),
        "dangling_sample": dangling[:20],
    }
    return text, stats


def _unit_facts(model: StepModel) -> dict:
    si_units = [e.body for e in model.of_type("SI_UNIT")]
    declared = [b for b in si_units if "METRE" in b.upper()]
    millimeters = any("MILLI" in b.upper() for b in declared)
    return {
        "si_unit_declarations": len(si_units),
        "length_unit_metre": bool(declared),
        "length_unit_is_millimetre": millimeters,
        "raw_length_units": declared[:3],
        "conversion_based_units": [e.body for e in model.of_type("CONVERSION_BASED_UNIT")][:3],
    }


def _product_facts(model: StepModel) -> dict:
    products = []
    for entity in model.of_type("PRODUCT"):
        name_match = re.match(r"\s*'([^']*)'", entity.body)
        products.append(
            {
                "entity_id": entity.eid,
                "name": name_match.group(1) if name_match else "",
                "raw": entity.body[:120],
            }
        )
    definitions = [
        {"entity_id": e.eid, "raw": e.body[:120]} for e in model.of_type("PRODUCT_DEFINITION")
    ]
    return {
        "products": products,
        "product_definitions": definitions,
        "next_assembly_usage_occurrences": len(model.of_type("NEXT_ASSEMBLY_USAGE_OCCURRENCE")),
    }


def build_census(model: StepModel, outlier_limit: int = 50) -> dict:
    counts = model.counts()
    bands, outlier_indices = point_statistics(model.points)
    coords = [model.points[i] for i in outlier_indices]
    per_axis_violation: Counter = Counter()
    for point in coords:
        for axis_index, band in enumerate(bands):
            if not band.allows(point[axis_index]):
                per_axis_violation[band.axis] += 1

    if model.points:
        xs = [p[0] for p in model.points]
        ys = [p[1] for p in model.points]
        zs = [p[2] for p in model.points]
        extent = {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
            "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        }
    else:
        extent = {}

    rule = (
        "outside [p%.1f, p%.1f] widened by 50%% of the span on any axis"
        % (bands[0].low_percentile, bands[0].high_percentile)
        if bands
        else "n/a"
    )

    return {
        "input": {
            "path": str(model.path),
            "size_bytes": model.size_bytes,
            "file_schema": model.file_schema,
            "parse_warnings": len(model.parse_warnings),
        },
        "entity_total": len(model.entities),
        # A complex instance is counted under its first component here and
        # listed again, in full, under complex_instance_types.
        "entity_types": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "complex_instance_count": sum(1 for e in model.entities.values() if e.is_complex),
        "complex_instance_types": dict(
            sorted(model.complex_types().items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "units": _unit_facts(model),
        "products": _product_facts(model),
        "points": {
            "count": len(model.points),
            "bounds": extent,
            "bands": [
                {
                    "axis": b.axis,
                    "low_percentile": b.low_percentile,
                    "high_percentile": b.high_percentile,
                    "low_value": b.low_value,
                    "high_value": b.high_value,
                    "span": b.span,
                    "allowed_low": b.allowed_low,
                    "allowed_high": b.allowed_high,
                }
                for b in bands
            ],
            "candidate_outliers": len(outlier_indices),
            "candidate_outlier_rule": rule,
            "candidate_outlier_axis_violations": dict(sorted(per_axis_violation.items())),
            "candidate_outlier_sample": [
                {"point_id": model.point_ids[i], "coords": list(coords[k])}
                for k, i in enumerate(outlier_indices[:outlier_limit])
            ],
        },
        "_outlier_indices": outlier_indices,
    }


def _write_outliers_csv(model: StepModel, outlier_indices: Sequence[int], path: Path) -> None:
    rows = ["point_id,x,y,z"]
    for index in outlier_indices:
        pid = model.point_ids[index]
        x, y, z = model.points[index]
        rows.append("%d,%.6f,%.6f,%.6f" % (pid, x, y, z))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _solid_bbox(entities: Dict[int, Entity], solid_id: int) -> "List[float] | None":
    closure = forward_closure(entities, [solid_id])
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for eid in closure:
        entity = entities[eid]
        if entity.etype != "CARTESIAN_POINT":
            continue
        coords = _extract_point(entity.body)
        if coords is None:
            continue
        xs.append(coords[0])
        ys.append(coords[1])
        zs.append(coords[2])
    if not xs:
        return None
    return [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="STEP file to analyse")
    parser.add_argument("--output-dir", required=True, help="directory for JSON/CSV outputs")
    parser.add_argument(
        "--mode",
        choices=("census", "attribute", "subset", "trace"),
        default="census",
        help="census only, plus outlier->solid attribution, or extract a subset",
    )
    parser.add_argument("--outlier-limit", type=int, default=50)
    parser.add_argument(
        "--solid-id",
        type=int,
        action="append",
        default=[],
        help="solid entity id to export (subset mode); repeatable",
    )
    parser.add_argument(
        "--pick-largest-solids",
        type=int,
        default=0,
        help="subset mode: auto-pick the N solids with the largest BREP closure",
    )
    parser.add_argument("--subset-out", default="", help="path of the extracted STEP")
    parser.add_argument(
        "--attribute-limit",
        type=int,
        default=0,
        help="attribute at most N outliers (0 = all); attribution is the slow part",
    )
    parser.add_argument(
        "--point-id",
        type=int,
        action="append",
        default=[],
        help="trace mode: Cartesian point entity id to trace (repeatable); default all outliers",
    )
    parser.add_argument(
        "--trace-sample",
        type=int,
        default=20,
        help="trace mode: how many per-point detail records to store in the report",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=0,
        help="trace mode: trace at most N candidate outliers (0 = all)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = parse_step(args.input)
    census = build_census(model, outlier_limit=args.outlier_limit)
    outlier_indices = census.pop("_outlier_indices")
    _write_outliers_csv(model, outlier_indices, out_dir / "cartesian_point_outliers.csv")

    report = {"census": census, "input_path": str(Path(args.input).resolve())}

    if args.mode in ("attribute", "subset", "trace"):
        targets = outlier_indices
        if args.attribute_limit:
            targets = targets[: args.attribute_limit]
        reverse = reverse_index(model.entities)
        point_ids = [model.point_ids[i] for i in targets]
        attribution = attribute_points_to_solids(model.entities, point_ids, reverse=reverse)
        owner_counter: Counter = Counter()
        unattributed: List[int] = []
        for point_id, owners in attribution.items():
            if not owners:
                unattributed.append(point_id)
            for owner in owners:
                owner_counter[owner] += 1
        report["attribution"] = {
            "points_attributed": len(attribution) - len(unattributed),
            "points_unattributed": len(unattributed),
            "unattributed_sample": unattributed[:20],
            "solids_hit": len(owner_counter),
            "solid_hit_histogram": dict(
                sorted(owner_counter.items(), key=lambda kv: -kv[1])[:20]
            ),
        }
        owner_bboxes = {}
        for solid_id in list(owner_counter)[:10]:
            bbox = _solid_bbox(model.entities, solid_id)
            if bbox is not None:
                owner_bboxes[solid_id] = bbox
        report["attribution"]["solid_bboxes_xyzxyz"] = owner_bboxes

    if args.mode == "trace":
        reverse = reverse_index(model.entities)
        if args.point_id:
            point_ids = [pid for pid in args.point_id if pid in model.entities]
            missing = [pid for pid in args.point_id if pid not in model.entities]
            if missing:
                report["trace_missing_point_ids"] = missing
        else:
            point_ids = [model.point_ids[i] for i in outlier_indices]
            if args.trace_limit:
                point_ids = point_ids[: args.trace_limit]
        details = [classify_point(model.entities, reverse, pid) for pid in point_ids]
        kinds: Counter = Counter(d["kind"] for d in details)
        by_solid: Counter = Counter()
        control_in_solid: Counter = Counter()
        vertex_in_solid: Counter = Counter()
        for detail in details:
            for solid_id in detail["solid_ids"]:
                by_solid[solid_id] += 1
                if detail["kind"] == POINT_KIND_BSPLINE_CONTROL:
                    control_in_solid[solid_id] += 1
                elif detail["kind"] == POINT_KIND_BREP_VERTEX:
                    vertex_in_solid[solid_id] += 1
        report["trace"] = {
            "points_traced": len(details),
            "kind_histogram": dict(kinds),
            "kind_meaning": {
                POINT_KIND_BREP_VERTEX: "the point is a BREP vertex (lies on the model)",
                POINT_KIND_BSPLINE_CONTROL: "the point is a B-spline control point (need not lie on the surface)",
                POINT_KIND_AUXILIARY: "referenced through auxiliary geometry only",
                POINT_KIND_UNKNOWN: "no upward path found",
            },
            "solid_hit_histogram": dict(
                sorted(by_solid.items(), key=lambda kv: -kv[1])[:20]
            ),
            "solid_bspline_control_histogram": dict(
                sorted(control_in_solid.items(), key=lambda kv: -kv[1])[:20]
            ),
            "solid_brep_vertex_histogram": dict(
                sorted(vertex_in_solid.items(), key=lambda kv: -kv[1])[:20]
            ),
            "usage_histogram": dict(
                Counter(",".join(d["direct_parent_types"]) for d in details).most_common()
            ),
            "samples": details[: args.trace_sample],
        }
        solid_names = {}
        for solid_id in list(by_solid)[:20]:
            entity = model.entities.get(solid_id)
            if entity is not None:
                solid_names[str(solid_id)] = entity_name(entity)
        report["trace"]["solid_names"] = solid_names

    if args.mode == "subset":
        solid_ids = list(args.solid_id)
        if args.pick_largest_solids:
            solids = model.of_type("MANIFOLD_SOLID_BREP") + model.of_type("BREP_WITH_VOIDS")
            sized = []
            for solid in solids:
                closure = forward_closure(model.entities, [solid.eid])
                sized.append((len(closure), solid.eid))
            sized.sort(reverse=True)
            solid_ids.extend(eid for _, eid in sized[: args.pick_largest_solids])
        if not solid_ids:
            parser.error("subset mode needs --solid-id and/or --pick-largest-solids")
        text, stats = extract_subset(model, solid_ids)
        subset_path = Path(args.subset_out) if args.subset_out else out_dir / "subset.step"
        subset_path.write_text(text, encoding="utf-8")
        stats["path"] = str(subset_path.resolve())
        report["subset"] = stats

    report_path = out_dir / ("step_%s.json" % args.mode)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    summary = {
        "report": str(report_path),
        "mode": args.mode,
        "entities": census["entity_total"],
        "points": census["points"]["count"],
        "candidate_outliers": census["points"]["candidate_outliers"],
    }
    if "trace" in report:
        summary["trace_kind_histogram"] = report["trace"]["kind_histogram"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
