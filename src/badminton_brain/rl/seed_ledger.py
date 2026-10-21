# -*- coding: utf-8 -*-
"""P8-03: PROVABLY disjoint train/val/test seed derivation + the seed ledger (JSON).

THE PROMISE
-----------
Given one master seed and a task family, derive three seed sets (train/val/test) such that

  * no seed can appear in two splits -- guaranteed by CONSTRUCTION, not by sampling:
    split s owns the half-open integer band [i*B, (i+1)*B) with i = SPLITS.index(s) and
    B = BAND_WIDTH.  The three bands are disjoint, therefore any seed of one split is
    numerically smaller or larger than every seed of every other split.
  * within one split no seed repeats either:
    seed(i) = band_lo + slot(i)*MAX_DERIVATIVES_PER_BASE, where
    slot(i) = (a*i + b) mod SLOTS_PER_BAND and a is FORCED ODD.  SLOTS_PER_BAND is a
    power of two, so an odd multiplier is a bijection on Z/SLOTS_PER_BAND: distinct i give
    distinct slots.  a and b are SHA-256 derived from (master_seed, task_family, split),
    so changing the master seed changes every derived seed while disjointness is preserved.
  * scene derivatives keyed by seed + parameter stay disjoint:
    derivative k of base seed j is band_lo + slot(j)*MAX_DERIVATIVES_PER_BASE + k with
    0 <= k < MAX_DERIVATIVES_PER_BASE.  All 64 derivatives of one base seed live inside that
    base seed's own 64-wide window, hence inside the split band, hence they can never collide
    with a derivative of another base seed or of another split.  A parameter name is hashed to
    k (with a documented bucketing function), so it cannot escape the window either.

The ledger is a JSON document that records the master seed, the task family, the band geometry
and every derived seed, plus a SHA-256 over exactly that content.  verify() re-derives the
checks and refuses any hand-edited ledger whose seeds are out of band, duplicated or
mis-counted; the hash changes whenever any single seed changes (tested).

This module performs no training, no rollout and no simulation: it derives integers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

SEED_LEDGER_VERSION = "p8-seeds-v0.1"
SPLITS: Tuple[str, ...] = ("train", "val", "test")
BAND_WIDTH = 1 << 40
MAX_DERIVATIVES_PER_BASE = 64  # 2**6: power of two keeps the window arithmetic exact
_SPLIT_INDEX: Dict[str, int] = {s: i for i, s in enumerate(SPLITS)}


class SeedLedgerError(ValueError):
    """Base class for every seed/ledger refusal."""


class UnknownSplitError(SeedLedgerError):
    """A split name that is not train/val/test."""


class OverlappingSplitRangeError(SeedLedgerError):
    """Two requested splits were given overlapping integer ranges."""


class SeedExhaustionError(SeedLedgerError):
    """More seeds were requested for a split than its band can hold."""


class LedgerIntegrityError(SeedLedgerError):
    """A ledger contradicts its own derivation rules (duplicate, out-of-band, bad hash)."""


def _digest(*parts: Any) -> int:
    """Stable integer digest of the parts (SHA-256, first 8 bytes, non-negative)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_band(split: str, band_width: int = BAND_WIDTH) -> Tuple[int, int]:
    """Half-open integer band owned by split; disjointness across splits is structural."""
    if split not in _SPLIT_INDEX:
        raise UnknownSplitError("unknown split {0!r}; expected one of {1}"
                                .format(split, SPLITS))
    if int(band_width) <= 0:
        raise SeedLedgerError("band_width must be > 0, got {0!r}".format(band_width))
    if int(band_width) % MAX_DERIVATIVES_PER_BASE != 0:
        raise SeedLedgerError("band_width must be a multiple of MAX_DERIVATIVES_PER_BASE={0}"
                              .format(MAX_DERIVATIVES_PER_BASE))
    i = _SPLIT_INDEX[split]
    lo = i * int(band_width)
    return lo, lo + int(band_width)


def _slots_per_band(band_width: int) -> int:
    return int(band_width) // MAX_DERIVATIVES_PER_BASE


def allocate_split_ranges(ranges: Mapping[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Validate an explicit split->(lo, hi) layout, refusing any overlap.

    An overlapping request is a hard refusal: silently shifting a split would invalidate every
    experiment that was already run against the old layout.
    """
    if not isinstance(ranges, Mapping) or not ranges:
        raise SeedLedgerError("ranges must be a non-empty mapping split -> (lo, hi)")
    checked: Dict[str, Tuple[int, int]] = {}
    for split, span in ranges.items():
        if split not in _SPLIT_INDEX:
            raise UnknownSplitError("unknown split {0!r}; expected one of {1}"
                                    .format(split, SPLITS))
        try:
            lo, hi = int(span[0]), int(span[1])
        except (TypeError, IndexError, ValueError):
            raise SeedLedgerError("split {0!r}: range must be a (lo, hi) pair of integers"
                                  .format(split))
        if lo < 0:
            raise SeedLedgerError("split {0!r}: lo must be >= 0, got {1}".format(split, lo))
        if hi <= lo:
            raise SeedLedgerError("split {0!r}: hi ({1}) must be > lo ({2})"
                                  .format(split, hi, lo))
        checked[split] = (lo, hi)
    names = sorted(checked)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (alo, ahi), (blo, bhi) = checked[a], checked[b]
            overlap_lo, overlap_hi = max(alo, blo), min(ahi, bhi)
            if overlap_lo < overlap_hi:
                raise OverlappingSplitRangeError(
                    "splits {0!r} [{1}, {2}) and {3!r} [{4}, {5}) overlap on [{6}, {7}); a seed "
                    "must belong to exactly one split, so this layout is refused"
                    .format(a, alo, ahi, b, blo, bhi, overlap_lo, overlap_hi))
    return checked


def default_band_layout(band_width: int = BAND_WIDTH) -> Dict[str, Tuple[int, int]]:
    """The canonical layout: train < val < test, three adjacent disjoint bands."""
    ranges = {s: split_band(s, band_width) for s in SPLITS}
    return allocate_split_ranges(ranges)


def _permutation(master_seed: int, task_family: str, split: str,
                 band_width: int) -> Tuple[int, int]:
    """Odd multiplier + shift, bijective on Z/SLOTS_PER_BAND (power-of-two modulus)."""
    slots = _slots_per_band(band_width)
    salt = _digest(int(master_seed), task_family, split, "permutation")
    a = (_digest(salt, "multiplier") % slots) | 1  # odd => coprime with a power of two
    b = _digest(salt, "shift") % slots
    return a, b


def base_slot(master_seed: int, task_family: str, split: str, index: int,
              band_width: int = BAND_WIDTH) -> int:
    """Slot of the index-th seed of split (bijective per split, so unique)."""
    if int(index) < 0:
        raise SeedLedgerError("seed index must be >= 0, got {0!r}".format(index))
    slots = _slots_per_band(band_width)
    if int(index) >= slots:
        raise SeedExhaustionError(
            "split {0!r} cannot hold {1} seeds: its band of {2} numbers divided into "
            "{3}-wide derivative windows gives only {4} slots"
            .format(split, int(index) + 1, band_width, MAX_DERIVATIVES_PER_BASE, slots))
    a, b = _permutation(master_seed, task_family, split, band_width)
    return (a * int(index) + b) % slots


def derive_seed(master_seed: int, task_family: str, split: str, index: int,
                band_width: int = BAND_WIDTH) -> int:
    """The base seed (derivative 0) of the index-th episode of split."""
    lo, _hi = split_band(split, band_width)
    return lo + base_slot(master_seed, task_family, split, index, band_width) * MAX_DERIVATIVES_PER_BASE


def derivative_bucket(parameter: Any, buckets: int = MAX_DERIVATIVES_PER_BASE) -> int:
    """Deterministic bucket 0 <= k < buckets for a derivative parameter (name or value)."""
    if int(buckets) <= 0:
        raise SeedLedgerError("buckets must be > 0")
    return _digest("derivative", parameter) % int(buckets)


def derive_derivative_seed(master_seed: int, task_family: str, split: str, index: int,
                           parameter: Any, band_width: int = BAND_WIDTH) -> int:
    """Scene-derivative seed: stays inside the base seed's own window, hence inside its band."""
    base = derive_seed(master_seed, task_family, split, index, band_width)
    return base + derivative_bucket(parameter)


@dataclass(frozen=True)
class SeedLedger:
    """The recorded seed sets plus the hash that pins them."""
    task_family: str
    master_seed: int
    seeds: Dict[str, Tuple[int, ...]]
    band_width: int = BAND_WIDTH
    max_derivatives_per_base: int = MAX_DERIVATIVES_PER_BASE
    version: str = SEED_LEDGER_VERSION

    # -- construction ---------------------------------------------------------------
    @classmethod
    def derive(cls, *, task_family: str, master_seed: int, counts: Mapping[str, int],
               band_layout: Optional[Mapping[str, Tuple[int, int]]] = None) -> "SeedLedger":
        if not isinstance(task_family, str) or not task_family.strip():
            raise SeedLedgerError("task_family must be a non-empty string")
        if int(master_seed) < 0:
            raise SeedLedgerError("master_seed must be >= 0, got {0!r}".format(master_seed))
        layout = allocate_split_ranges(band_layout if band_layout is not None
                                       else default_band_layout())
        widths = {lo_hi[1] - lo_hi[0] for lo_hi in layout.values()}
        if len(widths) != 1:
            raise SeedLedgerError("all split bands must have the same width, got {0}"
                                  .format(sorted(widths)))
        band_width = widths.pop()
        names = list(counts)
        missing = [s for s in SPLITS if s not in counts]
        if missing:
            raise SeedLedgerError("counts must declare every split; missing " + ", ".join(missing))
        unknown = sorted(set(names) - set(SPLITS))
        if unknown:
            raise UnknownSplitError("counts contains unknown split(s): " + ", ".join(unknown))
        seeds: Dict[str, Tuple[int, ...]] = {}
        for split in SPLITS:
            n = int(counts[split])
            if n < 0:
                raise SeedLedgerError("count for {0!r} must be >= 0, got {1}".format(split, n))
            lo, _hi = layout[split]
            slots = _slots_per_band(band_width)
            if n > slots:
                raise SeedExhaustionError(
                    "split {0!r} cannot hold {1} seeds; band {2} gives {3} slots"
                    .format(split, n, layout[split], slots))
            a, b = _permutation(int(master_seed), task_family, split, band_width)
            seeds[split] = tuple(lo + ((a * i + b) % slots) * MAX_DERIVATIVES_PER_BASE
                                 for i in range(n))
        ledger = cls(task_family=task_family, master_seed=int(master_seed), seeds=seeds,
                     band_width=band_width)
        ledger.verify()
        return ledger

    # -- access ---------------------------------------------------------------------
    def seeds_for(self, split: str) -> Tuple[int, ...]:
        if split not in _SPLIT_INDEX:
            raise UnknownSplitError("unknown split {0!r}; expected one of {1}".format(split, SPLITS))
        if split not in self.seeds:
            raise LedgerIntegrityError("ledger has no seed set for split {0!r}".format(split))
        return self.seeds[split]

    def seed(self, split: str, index: int) -> int:
        values = self.seeds_for(split)
        try:
            return values[int(index)]
        except IndexError:
            raise SeedLedgerError("split {0!r} has {1} seeds; index {2} is out of range"
                                  .format(split, len(values), index))

    def contains(self, split: str, seed: int) -> bool:
        return int(seed) in set(self.seeds_for(split))

    def derivative_seeds(self, split: str, index: int,
                         parameters: Sequence[Any]) -> Tuple[int, ...]:
        """Scene-derivative seeds for one base seed, one per parameter."""
        if len(set(parameters)) > self.max_derivatives_per_base:
            raise SeedExhaustionError(
                "at most {0} distinct derivatives per base seed".format(
                    self.max_derivatives_per_base))
        base = self.seed(split, index)
        return tuple(base + derivative_bucket(p, self.max_derivatives_per_base)
                     for p in parameters)

    # -- integrity ------------------------------------------------------------------
    def verify(self) -> None:
        """Refuse a ledger that contradicts the derivation rules (hand edits, mergers)."""
        seen: Dict[int, str] = {}
        for split in SPLITS:
            values = list(self.seeds_for(split))
            lo, hi = split_band(split, self.band_width)
            local: Dict[int, int] = {}
            for i, seed in enumerate(values):
                if not isinstance(seed, int) or isinstance(seed, bool):
                    raise LedgerIntegrityError(
                        "split {0!r} seed #{1} is not an int: {2!r}".format(split, i, seed))
                if not (lo <= seed < hi):
                    raise LedgerIntegrityError(
                        "split {0!r} seed #{1}={2} is outside its band [{3}, {4}); it would "
                        "collide with another split and is refused"
                        .format(split, i, seed, lo, hi))
                if seed in local:
                    raise LedgerIntegrityError(
                        "split {0!r} repeats seed {1} at indexes {2} and {3}"
                        .format(split, seed, local[seed], i))
                local[seed] = i
                if seed in seen:
                    raise LedgerIntegrityError(
                        "seed {0} appears in both {1!r} and {2!r}; splits must be disjoint"
                        .format(seed, seen[seed], split))
                seen[seed] = split

    def counts(self) -> Dict[str, int]:
        return {s: len(self.seeds_for(s)) for s in SPLITS}

    # -- hashing / serialisation -----------------------------------------------------
    def payload(self) -> Dict[str, Any]:
        """Exactly the content the hash covers (no self-reference)."""
        return {
            "version": self.version,
            "task_family": self.task_family,
            "master_seed": int(self.master_seed),
            "band_width": int(self.band_width),
            "max_derivatives_per_base": int(self.max_derivatives_per_base),
            "counts": self.counts(),
            "seeds": {s: list(self.seeds_for(s)) for s in SPLITS},
        }

    def ledger_hash(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        doc = dict(self.payload())
        doc["ledger_hash"] = self.ledger_hash()
        return doc

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def write(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "SeedLedger":
        if not isinstance(doc, Mapping):
            raise LedgerIntegrityError("ledger document must be a mapping")
        for name in ("version", "task_family", "master_seed", "band_width",
                     "max_derivatives_per_base", "seeds"):
            if name not in doc:
                raise LedgerIntegrityError("ledger document is missing {0!r}".format(name))
        if doc["version"] != SEED_LEDGER_VERSION:
            raise LedgerIntegrityError("unsupported ledger version {0!r} (expected {1})"
                                       .format(doc["version"], SEED_LEDGER_VERSION))
        if int(doc["max_derivatives_per_base"]) != MAX_DERIVATIVES_PER_BASE:
            raise LedgerIntegrityError(
                "max_derivatives_per_base={0} differs from the derivation rule {1}; a different "
                "window size cannot be verified against this code"
                .format(doc["max_derivatives_per_base"], MAX_DERIVATIVES_PER_BASE))
        seeds_doc = doc["seeds"]
        seeds: Dict[str, Tuple[int, ...]] = {}
        for split in SPLITS:
            if split not in seeds_doc:
                raise LedgerIntegrityError("ledger seeds are missing split {0!r}".format(split))
            seeds[split] = tuple(int(v) for v in seeds_doc[split])
        ledger = cls(task_family=str(doc["task_family"]), master_seed=int(doc["master_seed"]),
                     seeds=seeds, band_width=int(doc["band_width"]))
        ledger.verify()
        expected = doc.get("ledger_hash")
        if expected is not None and expected != ledger.ledger_hash():
            raise LedgerIntegrityError(
                "ledger hash mismatch: document says {0}, recomputed {1}; at least one seed "
                "changed after the hash was written".format(expected, ledger.ledger_hash()))
        return ledger

    @classmethod
    def load(cls, path: Any) -> "SeedLedger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace_seed(self, split: str, index: int, new_seed: int) -> "SeedLedger":
        """Copy of this ledger with one seed replaced (used by the hash-sensitivity tests)."""
        values = list(self.seeds_for(split))
        try:
            values[int(index)] = int(new_seed)
        except IndexError:
            raise SeedLedgerError("split {0!r} has {1} seeds; index {2} is out of range"
                                  .format(split, len(values), index))
        seeds = dict(self.seeds)
        seeds[split] = tuple(values)
        return SeedLedger(task_family=self.task_family, master_seed=self.master_seed, seeds=seeds,
                          band_width=self.band_width,
                          max_derivatives_per_base=self.max_derivatives_per_base,
                          version=self.version)


def verify_disjoint(ledgers: Iterable[SeedLedger]) -> None:
    """Cross-ledger check: one seed must not be used for two different task families."""
    owner: Dict[int, Tuple[str, str]] = {}
    for ledger in ledgers:
        for split in SPLITS:
            for seed in ledger.seeds_for(split):
                previous = owner.get(seed)
                if previous is not None and previous != (ledger.task_family, split):
                    raise LedgerIntegrityError(
                        "seed {0} is used by {1!r}/{2!r} and by {3!r}/{4!r}"
                        .format(seed, previous[0], previous[1], ledger.task_family, split))
                owner[seed] = (ledger.task_family, split)


__all__ = [
    "SEED_LEDGER_VERSION", "SPLITS", "BAND_WIDTH", "MAX_DERIVATIVES_PER_BASE",
    "SeedLedger", "SeedLedgerError", "UnknownSplitError", "OverlappingSplitRangeError",
    "SeedExhaustionError", "LedgerIntegrityError",
    "split_band", "allocate_split_ranges", "default_band_layout", "base_slot", "derive_seed",
    "derive_derivative_seed", "derivative_bucket", "verify_disjoint",
]
