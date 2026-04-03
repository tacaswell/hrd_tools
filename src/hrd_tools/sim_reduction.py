import functools
from collections import defaultdict
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import multianalyzer.opencl
import numpy as np
import pandas as pd
import tiled
import tiled.client
from matplotlib.figure import Figure
from multianalyzer import Result
from scipy.interpolate import splrep, sproot

from .config import (
    AnalyzerCalibration,
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimScanConfig,
)
from .file_io import dflt_config, find_varied_config, load_config, load_data


@functools.lru_cache
def load_reference(fname):
    return pd.read_csv(
        fname,
        skiprows=3,
        names=["theta", "I1", "I0"],
        sep=" ",
        skipinitialspace=True,
        index_col=False,
    )


def peak_fwhm(tth, normed, limits: tuple[float, float]):
    slc = slice(*np.searchsorted(tth, limits))
    tth = tth[slc]
    normed = normed[slc]
    if len(tth) == 0:
        return np.array([])
    half_max = np.max(normed) / 2.0
    s = splrep(tth, normed - half_max, k=3, s=np.max(normed) * 0.01)
    return np.diff(sproot(s))


def cat_to_fwhm(
    cat,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    limits: tuple[float, float],
    *,
    config_filter=None,
):
    ret = {}
    for key, run in cat.items():
        if config_filter is not None and not config_filter(run.metadata):
            continue
        (res, _config, _) = results[(cat.uri, key)]
        tth = res.tth
        (normed,) = normalize_result(res)
        fwhm = peak_fwhm(tth, normed, limits=limits)
        if len(fwhm) != 1:
            continue
        ret[key] = float(fwhm)

    reference_data = load_reference(
        cat.metadata["static_values"]["source"]["pattern_path"]
    )

    return ret, peak_fwhm(reference_data["theta"], reference_data["I1"], limits=limits)


def plot_cat_fwhm_1d(cat, results, peak_windows: list[tuple[float, float]]):
    fwhm = []
    for limits in peak_windows:
        fwhm.append(cat_to_fwhm(cat, results, limits))

    # flatten the varied values
    varied = {
        k: {f"{ok}.{ik}": val for ok, config in v.items() for ik, val in config.items()}
        for k, v in cat.metadata["varied_values"].items()
    }
    first = next(iter(varied.values()))
    print(first)
    fig = plt.figure(layout="constrained")
    ax_lst = fig.subplots(len(first), squeeze=False).ravel()
    for ax, key in zip(ax_lst, first, strict=True):
        for j, (marker, (peak_fwhm, ref_peak)) in enumerate(
            zip("ox^s", fwhm, strict=False)
        ):
            x = [varied[scan][key] for scan in peak_fwhm]
            y = list(peak_fwhm.values())
            (ln,) = ax.plot(x, y, marker, label=f"peak {j + 1}")
            ax.axhline(ref_peak, linestyle="--", color=ln.get_color())
        ax.legend()
        ax.set_xlabel(key)
        ax.set_ylabel("fwhm")

    return fig, fwhm


def load_config_from_tiled(grp: tiled.client.container.Container):
    configs = {}
    md = grp.metadata
    for fld in fields(CompleteConfig):
        try:
            config_grp = md[fld.name]
        except KeyError:
            if fld.name not in ("scan", "diffractometer"):
                print(f"missing {fld.name}")
        else:
            if fld.name == "analyzer" and "acceptance_angle" in config_grp:
                config_grp["incident_angle"] = config_grp.pop("acceptance_angle")
            configs[fld.name] = fld.type(**config_grp)
    if "scan" not in configs:
        tth = grp["tth"].read()
        configs["scan"] = SimScanConfig(
            start=np.rad2deg(tth[0]),
            stop=np.rad2deg(tth[-1]),
            delta=np.rad2deg(np.mean(np.diff(tth))),
        )
    return CompleteConfig(**configs)


def reduce_catalog(
    cat: tiled.client.container.Container,
    calib: AnalyzerCalibration | None = None,
    mode: str = "opencl",
    dtth_scale: float = 5.0,
    phi_max: float = 90,
):
    out = {}
    for k, sim in cat.items():
        config = load_config_from_tiled(sim)
        if calib is None:
            calib_sim = dflt_config(config)
        else:
            calib_sim = calib
        dtth = config.scan.delta * dtth_scale

        tth = np.rad2deg(sim["tth"].read())
        channels = sim["block"].read().astype("int32")
        # for j in range(config.analyzer.N):
        #     plot_raw(tth, channels[:, j, :, :].squeeze(), f"crystal {j}")
        ret = reduce_raw(
            block=channels,
            tths=tth,
            tth_min=np.float64(config.scan.start),
            tth_max=config.scan.stop
            + (config.analyzer.N - 1) * np.rad2deg(config.analyzer.cry_offset),
            dtth=dtth,
            analyzer=config.analyzer,
            detector=config.detector,
            calibration=calib_sim,
            phi_max=np.float64(phi_max),
            mode=mode,
            width=0,
        )
        out[(cat.uri, k)] = (ret, config, calib)

    # fig, ax = plt.subplots()
    # for j in range(config.analyzer.N):
    #     break
    #     ax.plot(ret.tth, 0.1 * j + ret.signal[j, :, 0] / ret.norm[j], label=j)
    # ax.legend()
    return out


def reduce_raw(
    block,
    tths,
    *,
    tth_min: float,
    tth_max: float,
    dtth: float,
    analyzer: AnalyzerConfig,
    detector: DetectorConfig,
    calibration: AnalyzerCalibration,
    phi_max: float = 90,
    mode="opencl",
    width=0,
):
    if mode == "cython":
        cls = multianalyzer.MultiAnalyzer
    elif mode == "opencl":
        cls = multianalyzer.opencl.OclMultiAnalyzer
        cls.NUM_CRYSTAL = np.int32(analyzer.N)
    else:
        raise ValueError
    mma = cls(
        # sample to crystal
        L=analyzer.R,
        # crystal to detector
        L2=analyzer.Rd,
        # pixel size
        pixel=detector.pitch,
        # pixel column direct beam is on
        # TODO pull from calibration structure
        center=np.array(calibration.detector_centers),
        # incident angle of crystals
        tha=analyzer.incident_angle,
        # global offset of MCA vs arm position
        # TODO pull from calibration structure
        # passing 0 defaults to 2x tha
        thd=0.0,
        # offsets of each crystal from origin of MCA
        # TODO pull from calibration structure
        psi=calibration.psi,
        # mis-orientation of the analyzer along x (°)
        # TODO pull from calibration structure
        rollx=[calibration.roll] * analyzer.N,
        # TODO pull from calibration structure
        # mis-orientation of the analyzer along y (°)
        rolly=[0.0] * analyzer.N,
    )
    _frames, _, _num_col, rox_max = block.shape
    ret = mma.integrate(
        # data all stacked as one
        roicollection=block,
        # arm positions
        arm=tths,
        # IO, currently constant
        mon=np.ones_like(tths),
        # range and resolution for binning
        tth_min=np.float64(tth_min),
        tth_max=np.float64(tth_max),
        dtth=np.float64(dtth),
        # shape of "ROIs"
        num_row=detector.transverse_size,
        num_col=1,
        roi_max=rox_max,
        # how to understand the shape of roicollection
        columnorder=1,
        phi_max=phi_max,
        width=width,
    )
    return ret


def sum_mca(mca):
    point_data = mca.sum(axis=1)
    return (point_data - point_data.min()) / np.ptp(point_data)


def plot_raw(
    tth,
    mca,
    title,
    *,
    fig: matplotlib.figure.Figure | None = None,
    label_max: bool = False,
):
    row, col = mca.shape
    if (row,) != tth.shape:
        raise RuntimeError

    if fig is None:
        fig = plt.figure(layout="constrained")

    ax_d = fig.subplot_mosaic([["parasite", "mca"]], width_ratios=(1, 5), sharey=True)

    fig.suptitle(title)
    im = ax_d["mca"].imshow(
        mca, extent=(0, col, tth[0], tth[-1]), origin="lower", aspect="auto"
    )
    ax_d["mca"].set_xlabel("channel index")
    point_data = sum_mca(mca)
    if label_max:
        max_indx = np.argmax(point_data)
        max_tth = tth[max_indx]
        ax_d["parasite"].axhline(max_tth, color="k", alpha=0.5, zorder=0)
        ax_d["parasite"].annotate(
            f"{max_tth:.2f}°",
            xy=(0, max_tth),
            xytext=(0, -5),
            textcoords="offset points",
            ha="left",
            va="top",
        )
    ax_d["parasite"].plot(point_data, tth)
    ax_d["parasite"].set_xlabel("column sum (arb)")
    ax_d["parasite"].set_ylabel("arm angle (deg)")
    cb = fig.colorbar(im, ax=ax_d["mca"])
    cb.set_label("counts")
    return ax_d


def reduce_file(
    fname: Path,
    calib: AnalyzerCalibration | None = None,
    mode: str = "opencl",
    dtth_scale: float = 5.0,
    phi_max: float = 90,
):
    config = load_config(fname)
    if calib is None:
        calib = dflt_config(config)
    dtth = config.scan.delta * dtth_scale
    print(f"{dtth=}")
    tth, channels = load_data(fname)
    # for j in range(config.analyzer.N):
    #     plot_raw(tth, channels[:, j, :, :].squeeze(), f"crystal {j}")
    ret = reduce_raw(
        block=channels,
        tths=tth.astype("float64"),
        tth_min=np.float64(config.scan.start),
        tth_max=config.scan.stop
        + config.analyzer.N * np.rad2deg(config.analyzer.cry_offset),
        dtth=dtth,
        analyzer=config.analyzer,
        detector=config.detector,
        calibration=calib,
        phi_max=np.float64(phi_max),
        mode=mode,
        width=0,
    )
    # fig, ax = plt.subplots()
    # for j in range(config.analyzer.N):
    #     break
    #     ax.plot(ret.tth, 0.1 * j + ret.signal[j, :, 0] / ret.norm[j], label=j)
    # ax.legend()
    return ret, config, calib


def reduce_and_plot(
    files: list[Path],
    # configs: dict[Path, tuple[CompleteConfig, AnalyzerCalibration]],
    **kwargs,
) -> tuple[dict[Path, Result], Figure]:
    reduced = {k: reduce_file(k, **kwargs) for k in files}

    label_keys = find_varied_config([v[1] for v in reduced.values()])
    print(label_keys)
    fig, ax = plt.subplots()
    for _, (ret, config, _) in reduced.items():
        for j in range(config.analyzer.N):
            label = " ".join(
                f"{sec}.{parm}={getattr(getattr(config, sec), parm):.02g}"
                for sec, parm in label_keys
            )
            normed = ret.signal[j, :, 0] / ret.norm[j]
            mask = np.isfinite(normed)
            ax.plot(ret.tth[mask], 0.1 * j + normed[mask], label=label)
    ax.legend()
    return reduced, fig


def normalize_result(res: Result, scale_to_max=True):
    signal = res.signal.sum(axis=2)
    out = np.zeros_like(signal, dtype=float)
    # first dimension is number of crystals
    for j in range(signal.shape[0]):
        normed = signal[j] / res.norm[j]
        if scale_to_max:
            normed /= np.nanmax(normed)
        out[j] = normed
    return out


def plot_ref(df, ax, scale_to_max=True, **kwargs):
    x = df["theta"]
    y = df["I1"]
    if scale_to_max:
        y /= y.max()

    ax.plot(x, y, **{"label": "reference", "scalex": False, **kwargs})


def plot_reduced(
    res: Result,
    ax=None,
    *,
    label: str | None = None,
    scale_to_max: bool = False,
    orientation: str = "h",
    **kwargs,
):
    if ax is None:
        _fig, ax = plt.subplots(layout="constrained")
        # this should be length 1, sum just to be safe, is higher if in non-ROI mode
    for j, normed in enumerate(normalize_result(res, scale_to_max)):
        if label is not None:
            label = f"{label} ({j})"
        mask = np.isfinite(normed)
        if orientation == "h":
            ax.plot(res.tth[mask], normed[mask], label=label, **kwargs)
        elif orientation == "v":
            ax.plot(normed[mask], res.tth[mask], label=label, **kwargs)


def raw_grid(cat, results, config_filter=None):
    fig = plt.figure(layout="constrained", figsize=(15, 15))
    grid_size = int(np.ceil(np.sqrt(len(cat))))
    label_keys = cat.metadata["varied_key"]
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    grid_size2 = grid_size
    while (grid_size * (grid_size2 - 1)) >= len(cat):
        grid_size2 -= 1
    sub_figs = fig.subfigures(grid_size, grid_size2)
    for (k, sim), sfig in zip(cat.items(), sub_figs.ravel(), strict=False):
        if config_filter is not None and not config_filter(sim.metadata):
            continue
        tth = np.rad2deg(sim["tth"].read())
        block = sim["block"].read().astype("int32")
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in (_.split(".") for _ in label_keys)
        )
        axd = plot_raw(tth, block[:, 0, 0, :], label, fig=sfig)
        plot_reduced(res, ax=axd["parasite"], orientation="v", scale_to_max=True)

    return fig


def plot_reduced_cat(
    cat,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    title=None,
    reference_pattern=True,
    xlim=None,
    *,
    config_filter=None,
):
    fig, ax = plt.subplots(layout="constrained")
    if title:
        fig.suptitle(title)
    label_keys = tuple(_.split(".") for _ in cat.metadata["varied_key"])
    print(label_keys)
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    for k, run in cat.items():
        if config_filter is not None and not config_filter(run.metadata):
            continue
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in label_keys
        )
        plot_reduced(res, ax=ax, scale_to_max=True, label=label, alpha=0.5)
    if reference_pattern:
        reference_data = load_reference(
            cat.metadata["static_values"]["source"]["pattern_path"]
        )
        plot_ref(reference_data, ax, scalex=False, linestyle=":")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylabel("I (arb)")
    ax.set_xlabel(r"2$\theta$ (deg)")
    ax.legend(loc="upper left")
    return fig
