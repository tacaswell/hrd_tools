from itertools import count
import contextlib
from dataclasses import asdict, fields, is_dataclass
import pyarrow as pa
import pyarrow.parquet as pq
import sparse
import numpy as np
import tomllib
import tqdm
import json

from hrd_tools.config import (
    AnalyzerConfig,
    DetectorConfig,
    Diffractometer,
    SimConfig,
    SimScanConfig,
    SourceConfig,
    GeneratorInvocation,
)
from hrd_tools.xrt.endstation import Endstation


def get_frames(
    self,
    states=(1, 2),
):
    detector_config = self.detector
    screen_beams = self.run_process()

    isScreen = True

    shape = (
        int(detector_config.height // detector_config.pitch),
        detector_config.transverse_size,
    )

    limits = list(
        (detector_config.pitch * np.array([[-0.5, 0.5]]).T * np.array([shape])).T
    )

    out = []

    for k, lb in sorted(screen_beams.items()):
        # print(lb.x, lb.y, lb.z, lb.state)
        good = lb.state == states[0]
        for s in states[1:]:
            good |= lb.state == s

        if isScreen:
            x, y = lb.x[good], lb.z[good]
        else:
            x, y = lb.x[good], lb.y[good]

        flux = lb.Jss[good] + lb.Jpp[good]
        hist2d, *_ = np.histogram2d(y, x, bins=shape, range=limits, weights=flux)

        out.append(
            sparse.COO.from_numpy(np.floor(hist2d).astype("uint16"))[
                np.newaxis, np.newaxis, ...
            ]
        )

    return sparse.concat(out)


def scan_to_file(
    bl: Endstation,
    scan_config: SimScanConfig,
    *,
    writer,
):
    cache_rate = writer.send(None)
    with contextlib.closing(writer):
        start, stop, delta = np.deg2rad(
            [scan_config.start, scan_config.stop, scan_config.delta]
        )
        N = int((stop - start) / delta)
        tths = np.linspace(start, stop, N)
        writer.send((bl, np.rad2deg(tths), np.ones_like(tths), scan_config))
        for batch in tqdm.tqdm(range(len(tths) // cache_rate + 1)):
            batch_tth = tths[batch * cache_rate : (batch + 1) * cache_rate]
            result_cache = []
            try:
                for j, tth in enumerate(tqdm.tqdm(batch_tth)):  # noqa: B007
                    bl.set_arm(tth)
                    result_cache.append(get_frames(bl))
            finally:
                if len(result_cache) > 0:
                    writer.send(sparse.concat(result_cache, axis=1))


def dump_coro(output_dir, cache_rate, invocation, *, tlg="sim"):
    bl, tths, monitor, scan_config = yield cache_rate

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names with consistent naming

    scalars_path = output_dir / "scalars.parquet"
    md_path = output_dir / "metadata.json"

    scalars_table = pa.Table.from_arrays(
        [pa.array(tths), pa.array(monitor)],
        names=("tth", "monitor"),
        metadata={"nominal_bin": str(scan_config.delta)},
    )
    pq.write_table(scalars_table, scalars_path, compression="snappy")
    with open(md_path, "w") as fout:
        json.dump(
            {
                "scan_config": asdict(scan_config),
                **{
                    f"{fld.name}_config": asdict(getattr(bl, fld.name))
                    for fld in fields(bl)
                    if is_dataclass(fld.type)
                },
                "generator": asdict(invocation),
            },
            fout,
            indent=2,
        )

    chunk_number = count()
    for chunk in chunk_number:
        try:
            frames = yield
        except GeneratorExit:
            return

        # Write sparse detector images to parquet
        images_table = pa.Table.from_arrays(
            [*frames.coords, frames.data],
            names=("detector", "frame", "row", "col", "data"),
            metadata={"shape": json.dumps(frames.shape)},
        )
        pq.write_table(
            images_table,
            output_dir / f"images_{chunk:03d}.parquet",
            compression="snappy",
            write_statistics=False,
        )


def to_block(data, *, sum_col=True):
    block = np.stack([v for k, v in sorted(data.items())], axis=1)
    if sum_col:
        block = block.sum(axis=2, keepdims=True)
    return block


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run some sims.")
    parser.add_argument(
        "config", help="toml to read to get the configuration.", type=Path
    )
    parser.add_argument("output_dir", help="Director to write results to.", type=Path)

    parser.add_argument(
        "--cache-rate",
        help="flush to disk every this number of angles",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    class_map = {
        "source": SourceConfig,
        "sim": SimConfig,
        "detector": DetectorConfig,
        "analyzer": AnalyzerConfig,
        "diffractometer": Diffractometer,
    }
    with args.config.open("rb") as fin:
        inp = tomllib.load(fin)

    configs = {k: cls(**inp[k]) if k in inp else cls() for k, cls in class_map.items()}
    invocation = GeneratorInvocation(**inp["generator"])

    fin = Path(configs["source"].pattern_path).resolve()
    if not fin.exists():
        msg = "pattern source does not exist"
        raise ValueError(msg)

    bl = Endstation.from_configs(**configs)
    writer = dump_coro(args.output_dir, args.cache_rate, invocation)
    scan_config = SimScanConfig(**inp["scan"])
    scan_to_file(bl, scan_config, writer=writer)
