import argparse
from collections import defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, get_args, get_origin
from hashlib import md5
import uuid

import numpy as np
import tomli_w
from cycler import Cycler, cycler


def _to_toml_serializable(obj: Any) -> Any:
    """Recursively convert non-TOML-serializable types (e.g. Path) to strings."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_toml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_toml_serializable(v) for v in obj]
    return obj

from hrd_tools.config import (
    AnalyzerCalibration,
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    Diffractometer,
    SimConfig,
    SimScanConfig,
    SourceConfig,
    GeneratorInvocation,
)


def effective_half_phi(tth: float, d: float, detector_width: float) -> float:
    """Half-phi coverage of a detector on the Debye-Scherrer cone.

    Parameters
    ----------
    tth : float
        Two-theta angle in radians.
    d : float
        Sample-to-detector distance (any consistent unit with *detector_width*).
    detector_width : float
        Full transverse width of the detector (same unit as *d*).

    Returns
    -------
    float
        Half-angle in radians of the arc subtended by the detector.
    """
    R = d * np.abs(np.sin(tth))
    return np.arctan2(detector_width / 2, R)


def _pad_source_range(config: CompleteConfig) -> CompleteConfig:
    """Set source min/max_tth to be 2.5% wider than scan start/stop."""
    scan_range = config.scan.stop - config.scan.start
    padding = 0.025 * scan_range
    return replace(
        config,
        source=replace(
            config.source,
            min_tth=config.scan.start - padding,
            max_tth=config.scan.stop + padding,
        ),
    )


def _set_delta_phi(config: CompleteConfig) -> CompleteConfig:
    """Set source delta_phi to just cover the detector at the scan start angle."""
    detector_width = config.detector.pitch * config.detector.transverse_size  # mm
    d = config.analyzer.R  # mm
    tth_rad = np.deg2rad(config.scan.start)
    half_phi = effective_half_phi(tth_rad, d, detector_width)
    return replace(
        config,
        source=replace(config.source, delta_phi=2*np.rad2deg(half_phi)),
    )


def get_defaults(start=20, stop=20.5):
    config = CompleteConfig(
        **{
            "source": SourceConfig(
                E_incident=29_400,
                pattern_path="/home/tcaswell/scratch/2026/02/28/11bmb_7871_Y1.xye",
                dx=1,
                dz=1,
                dy=1,
                # will be overridden by _set_delta_phi
                delta_phi=3,
                E_hwhm=1.4e-4,
                # default to unfocused
                h_div=np.rad2deg(22e-6),
                v_div=np.rad2deg(0.9e-6),
                # will be overriden by _pad_source_range
                max_tth=stop,
                min_tth=start,
                source_offset_x=0,
                source_offset_y=0,
                source_offset_z=0,
            ),
            "sim": SimConfig(nrays=500_000),
            "detector": DetectorConfig(
                # default to mediapix3
                pitch=0.055,
                transverse_size=256,
                height=10,
            ),
            "analyzer": AnalyzerConfig(
                R=950,
                Rd=115,
                cry_offset=np.deg2rad(2.5),
                cry_width=102,
                cry_depth=54,
                N=1,
                # will be set when buliding end station
                incident_angle=0,
                thickness=10,
                roll=0,
            ),
            "scan": SimScanConfig(start=start, stop=stop, delta=1e-4),
            "diffractometer": Diffractometer(),
        }
    )
    config = _pad_source_range(config)
    config = _set_delta_phi(config)
    return config


def convert_cycler(cycle: Cycler) -> list[CompleteConfig]:
    defaults = get_defaults()
    cycle_keys = set(cycle.keys)
    user_set_source_tth = bool({"source.min_tth", "source.max_tth"} & cycle_keys)
    user_set_delta_phi = "source.delta_phi" in cycle_keys
    out = []
    for entry in cycle:
        nested: dict[str, dict[str, Any]] = defaultdict(dict)
        for k, v in entry.items():
            outer, _, inner = k.partition(".")
            nested[outer][inner] = v

        config = replace(
            defaults,
            **{k: replace(getattr(defaults, k), **v) for k, v in nested.items()},
        )
        if not user_set_source_tth:
            config = _pad_source_range(config)
        if not user_set_delta_phi:
            config = _set_delta_phi(config)
        out.append(config)
    return out


config_classes = [
    AnalyzerConfig,
    DetectorConfig,
    Diffractometer,
    SimConfig,
    SourceConfig,
    SimScanConfig,
    AnalyzerCalibration,
]


# Helper: given a field's annotation, return its conversion function.
def get_conversion_func(annot):
    origin = get_origin(annot)
    if origin is tuple:
        args = get_args(annot)
        # Assume homogeneous tuple such as tuple[float, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        else:
            return args[0]
    return annot


# Build a mapping from configuration key to its conversion function.
type_map = {}
allowed_keys = set()
full_name_map = {}
for cfg_name_field in fields(CompleteConfig):
    cls = cfg_name_field.type
    cfg_name = cfg_name_field.name
    for field in fields(cls):
        if field.name in allowed_keys:
            raise ValueError(f"Name colision on {field.name}")
        allowed_keys.add(field.name)
        type_map[field.name] = get_conversion_func(field.type)
        full_name_map[field.name] = f"{cfg_name}.{field.name}"


# Utility: convert a string to a list of values using conv.
#
# Supported formats (for numeric types):
#   start:stop:step   → np.arange(start, stop, step)
#   (start,stop,num)  → np.linspace(start, stop, int(num))
#   val1,val2,val3    → [conv(v) for v in vals]
def list_type(value_str: str, conv):
    s = value_str.strip().strip("\"'")
    is_numeric = conv in (float, int)

    # slice syntax: start:stop:step → np.arange
    if is_numeric and ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Slice syntax requires exactly 3 parts start:stop:step, got {s!r}"
            )
        try:
            start, stop, step = (float(p) for p in parts)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Could not parse slice values: {e}"
            ) from e
        return [conv(v) for v in np.arange(start, stop, step)]

    # linspace syntax: (start,stop,num) → np.linspace
    if is_numeric and s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Linspace syntax requires exactly 3 parts (start,stop,num), got {s!r}"
            )
        try:
            start, stop = float(parts[0]), float(parts[1])
            num = int(parts[2])
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Could not parse linspace values: {e}"
            ) from e
        return [conv(v) for v in np.linspace(start, stop, num)]

    # default: comma-separated list
    items = [item.strip() for item in s.split(",") if item.strip()]
    try:
        return [conv(item) for item in items]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not convert values: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Generate cycler objects from per-key command-line flags with proper type conversion.\n"
        "Flags with multiple values are combined via addition (Cartesian product), while flags "
        "with a single value are incorporated via multiplication (broadcasting the constant)."
    )
    # Add optional subdirectory flag
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional subdirectory name under 'configs/' to nest the generated config files.",
    )
    # Dynamically add a flag for each allowed key.
    for key in sorted(allowed_keys):
        conv = type_map[key]
        # Special case: make short_description required at argparse level
        is_required = (key == "short_description")
        parser.add_argument(
            f"--{key}",
            type=lambda s, conv=conv: list_type(s, conv),
            help=f"Comma-separated list of values for '{key}' (converted to {conv.__name__}).",
            required=is_required,
        )

    args = parser.parse_args()

    # Separate cyclers into variable (multiple values) and constant (single value).
    variable_cyclers = []
    constant_cyclers = []
    for key in sorted(allowed_keys):
        values = getattr(args, key)
        if values is not None:
            c = cycler(full_name_map[key], values)
            if len(values) == 1:
                constant_cyclers.append(c)
            else:
                variable_cyclers.append(c)

    if not variable_cyclers and not constant_cyclers:
        parser.error("At least one configuration flag must be provided.")

    # Combine variable cyclers using addition (Cartesian sum).
    combined: Cycler | None = None
    if variable_cyclers:
        combined = variable_cyclers[0]
        for c in variable_cyclers[1:]:
            combined = combined + c

    # Incorporate constant cyclers using multiplication.
    for c in constant_cyclers:
        if combined is None:
            combined = c
        else:
            combined = combined * c
    assert (
        combined is not None
    )  # Should never be None here due to earlier check, helps type checker.
    print("Generated cycler object:")
    print(combined)

    configs = convert_cycler(combined)
    config_path = Path("configs")
    if args.subdir:
        config_path = config_path / args.subdir
    config_path.mkdir(parents=True, exist_ok=True)
    for f in config_path.glob("config_*.toml"):
        f.unlink()

    gen_md = GeneratorInvocation(
        **{
            "cycler": repr(combined),
            "uuid": str(uuid.uuid4()),
            "md5sum": md5(repr(combined).encode()).hexdigest(),
        }
    )
    for j, config in enumerate(configs):
        print(
            ", ".join(
                f"{k}: {getattr(getattr(config, sub), key)}"
                for k, sub, _, key in [(k, *k.partition(".")) for k in combined.keys]
            )
        )

        with open(config_path / f"config_{j}.toml", "wb") as fout:
            tomli_w.dump(_to_toml_serializable({**asdict(config), "generator": asdict(gen_md)}), fout)

    subdir = args.subdir or ""
    print(
        f"\nTo launch:\n"
        f"  sbatch --export=ALL,CONFIG_SUBDIR={subdir} --array=0-{len(configs) - 1} batch.job"
    )


if __name__ == "__main__":
    main()
