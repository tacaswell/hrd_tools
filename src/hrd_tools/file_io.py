from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import (
    AnalyzerCalibration,
    CompleteConfig,
    Diffractometer,
    SimScanConfig,
)


def load_all_config(path: Path, *, ext="h5", prefix="") -> dict[Path, CompleteConfig]:
    configs = {}
    for _ in sorted(path.glob(f"**/{prefix}*{ext}")):
        config = load_config(_)

        configs[_] = config
    return configs


def load_config(fname, *, tlg="sim"):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        return load_config_from_group(g)


def load_config_from_group(grp):
    configs = {}

    for fld in fields(CompleteConfig):
        try:
            config_grp = grp[f"{fld.name}_config"]
        except KeyError:
            if fld.name not in ("scan", "diffractometer"):
                print(f"missing {fld.name}")
        else:
            config_attrs = dict(**config_grp.attrs)
            if fld.name == "analyzer" and "acceptance_angle" in config_attrs:
                config_attrs["incident_angle"] = config_attrs.pop("acceptance_angle")
            configs[fld.name] = fld.type(**config_attrs)
    if "scan" not in configs:
        tth = grp["tth"][:]
        configs["scan"] = SimScanConfig(
            start=np.rad2deg(tth[0]),
            stop=np.rad2deg(tth[-1]),
            delta=np.rad2deg(np.mean(np.diff(tth))),
        )
    return CompleteConfig(**configs)


def dflt_config(complete_config):
    print(complete_config)
    return AnalyzerCalibration(
        detector_centers=(
            [complete_config.detector.transverse_size / 2] * complete_config.analyzer.N
        ),
        psi=[
            np.rad2deg(complete_config.analyzer.cry_offset) * j
            for j in range(complete_config.analyzer.N)
        ],
        roll=[complete_config.analyzer.roll] * complete_config.analyzer.N,
    )


def load_data(fname, *, tlg="sim", scale=1):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        block = g["block"][:]
        block *= scale
        return np.rad2deg(g["tth"][:]), block.astype("int32")


def find_varied_config(configs):
    out = defaultdict(set)
    for c in configs:
        if isinstance(c, dict):
            cd = c
        else:
            print(type(c))
            cd = asdict(c)
        for k, sc in cd.items():
            for f, v in sc.items():
                out[(k, f)].add(v)
    return [k for k, s in out.items() if len(s) > 1]


def config_to_table(config):
    df = pd.DataFrame(
        {
            k: {
                f"{outer}.{inner}": v
                for outer, cfg in asdict(v).items()
                for inner, v in cfg.items()
            }
            for k, v in config.items()
        }
    ).T.infer_objects()
    df["job"] = [str(_.name.partition("-")[0]) for _ in df.index]
    return df
