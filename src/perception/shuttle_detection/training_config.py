"""Validated schema for a shuttlecock baseline training run.

Why this module exists: a baseline is only reusable if two runs that claim to be
"the baseline" really did use the same settings. A YAML file that nothing but the
training script reads cannot guarantee that -- a typo such as `nc: 2` or a renamed
key would silently change the experiment. This module is the single place that
turns the baseline YAML into one frozen, validated configuration, so a run either
uses exactly the declared baseline or refuses to start.

The detection task is fixed by spec section 2 (nc=1, class 0 "shuttlecock"); the
schema therefore rejects any other class count instead of training a different
problem under the same run name.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_training_config.py -v
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping

import yaml

# src/perception/shuttle_detection/training_config.py -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "shuttle_detection" / "baseline.yaml"

# The one class this sub-project detects. Spec section 2 fixes both values.
SHUTTLECOCK_CLASS_INDEX = 0
SHUTTLECOCK_CLASS_NAME = "shuttlecock"


class ConfigError(ValueError):
    """The YAML cannot describe a valid nc=1 shuttlecock baseline run."""


def _as_int(name: str, value: Any) -> int:
    """Coerce a YAML scalar to int, rejecting values that only look numeric.

    bool is checked first on purpose: in Python True is an int, and `imgsz: yes`
    would otherwise quietly become 1 px.
    """
    if isinstance(value, bool):
        raise ConfigError(name + " must be an integer, got a boolean: " + repr(value))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ConfigError(name + " must be an integer, got " + repr(value))


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigError(name + " must be a number, got a boolean: " + repr(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ConfigError(name + " must be a number, got " + repr(value))


def _as_bool(name: str, value: Any) -> bool:
    """Strict boolean: a config says true or false, not "yes" and not 1.

    Strictness matters here because this switch changes only how images are LOADED, so a typo that
    silently left it false would look like "caching did not help" rather than "caching was ignored".
    """
    if isinstance(value, bool):
        return value
    raise ConfigError(name + " must be a YAML boolean true/false, got " + repr(value))


def _as_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(name + " must be a non-empty string, got " + repr(value))
    return value.strip()


@dataclass(frozen=True)
class BaselineConfig:
    """One immutable baseline run configuration.

    Defaults are the declared baseline of spec/plan Task 1, so a short YAML file
    cannot leave a hyper-parameter undefined. load() is the only supported way to
    build one from a file; direct construction is for tests and for picking the
    baseline explicitly.
    """

    model: str = "yolo26s.pt"
    nc: int = 1
    name: str = "shuttlecock"
    imgsz: int = 640
    seed: int = 17
    epochs: int = 100
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    # I/O-only switch, DEFAULT FALSE so every config written before this field existed behaves exactly
    # as it did and the declared baseline is unchanged. When true, Ultralytics reads the images once
    # into RAM instead of decoding every image again on every epoch; the model, the data, the
    # augmentation, the seed and the optimiser are untouched, so a cached run is the same experiment
    # with a different data-loading path.
    #
    # Why it is worth having: on this host the storage is a fuseblk mount under heavy contention, and
    # epochs were measured at 61 s, 559 s, 274 s, 247 s - roughly 7 hours for 100 epochs, against a
    # historical 11.7 s/epoch for the same recipe on the same machine. That made the retrain
    # impractical rather than merely slow.
    cache: bool = False

    # The complete, ordered field list. Anything else in the YAML is a mistake.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "model",
        "nc",
        "name",
        "imgsz",
        "seed",
        "epochs",
        "batch",
        "optimizer",
        "lr0",
        "cache",
    )

    def __post_init__(self) -> None:
        """Validate on construction so an invalid config cannot exist at all."""
        _as_text("model", self.model)
        if self.nc != 1:
            raise ConfigError(
                "nc must be 1 for the shuttlecock detection task, got " + repr(self.nc)
            )
        if self.name != SHUTTLECOCK_CLASS_NAME:
            raise ConfigError(
                "name must be "
                + repr(SHUTTLECOCK_CLASS_NAME)
                + " for the nc=1 task, got "
                + repr(self.name)
            )
        if self.imgsz <= 0:
            raise ConfigError("imgsz must be positive, got " + repr(self.imgsz))
        if self.epochs < 1:
            raise ConfigError("epochs must be at least 1, got " + repr(self.epochs))
        if self.batch == 0:
            raise ConfigError("batch must not be 0 (use -1 for the Ultralytics auto batch)")
        _as_text("optimizer", self.optimizer)
        if self.lr0 <= 0:
            raise ConfigError("lr0 must be positive, got " + repr(self.lr0))
        if not isinstance(self.cache, bool):
            raise ConfigError("cache must be a YAML boolean, got " + repr(self.cache))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BaselineConfig":
        """Build a config from already-parsed YAML content."""
        if not isinstance(mapping, Mapping):
            raise ConfigError("baseline config must be a YAML mapping, got " + repr(mapping))
        unknown = sorted(str(key) for key in mapping if str(key) not in cls.FIELDS)
        if unknown:
            raise ConfigError(
                "unknown baseline config key(s): "
                + ", ".join(unknown)
                + "; known keys are "
                + ", ".join(cls.FIELDS)
            )
        values = {str(key): value for key, value in mapping.items()}
        defaults = cls()
        return cls(
            model=_as_text("model", values.get("model", defaults.model)),
            nc=_as_int("nc", values.get("nc", defaults.nc)),
            name=_as_text("name", values.get("name", defaults.name)),
            imgsz=_as_int("imgsz", values.get("imgsz", defaults.imgsz)),
            seed=_as_int("seed", values.get("seed", defaults.seed)),
            epochs=_as_int("epochs", values.get("epochs", defaults.epochs)),
            batch=_as_int("batch", values.get("batch", defaults.batch)),
            optimizer=_as_text("optimizer", values.get("optimizer", defaults.optimizer)),
            lr0=_as_float("lr0", values.get("lr0", defaults.lr0)),
            cache=_as_bool("cache", values.get("cache", defaults.cache)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "BaselineConfig":
        """Read and validate a baseline YAML file.

        Raises FileNotFoundError when the file is absent, and ConfigError when the
        document is not a legal nc=1 baseline.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError("baseline config not found: " + str(config_path))
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if document is None:
            raise ConfigError("baseline config is empty: " + str(config_path))
        if not isinstance(document, Mapping):
            raise ConfigError(
                "baseline config must be a YAML mapping, got " + type(document).__name__
            )
        return cls.from_mapping(document)

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable export; recorded as the resolved config."""
        return asdict(self)


def declared_fields() -> tuple[str, ...]:
    """Field names taken from the dataclass itself, for cross-checking the schema."""
    return tuple(field.name for field in fields(BaselineConfig))
