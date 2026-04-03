import functools
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
import pandas as pd
import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.screens as rscreens

from ..config import (
    AnalyzerConfig,
    DetectorConfig,
    Diffractometer,
    SimConfig,
    SourceConfig,
)
from .geometry import compute_arm_positions
from .sources import XrdSource
from .stops import RectangularBeamstop


@dataclass
class Endstation:
    bl: raycing.BeamLine
    analyzer: AnalyzerConfig
    source: SourceConfig
    detector: DetectorConfig
    sim: SimConfig
    diffractometer: Diffractometer = field(default_factory=Diffractometer)

    @classmethod
    def from_configs(
        cls,
        analyzer: AnalyzerConfig,
        source: SourceConfig,
        detector: DetectorConfig,
        sim: SimConfig,
        diffractometer: Diffractometer | None = None,
    ) -> Self:
        if diffractometer is None:
            diffractometer = Diffractometer()

        crystalSi01 = rmats.CrystalSi(t=analyzer.thickness)
        theta_b = _bragg(crystalSi01, source.E_incident)
        analyzer = replace(analyzer, incident_angle=np.rad2deg(theta_b))

        arm_tth = np.deg2rad(15)
        beamLine = raycing.BeamLine()

        reference_pattern = pd.read_csv(
            source.pattern_path,
            skiprows=3,
            names=["theta", "I1", "I0"],
            sep=" ",
            skipinitialspace=True,
            index_col=False,
        )
        delta_phi = np.deg2rad(source.delta_phi)
        beamLine.geometricSource01 = XrdSource(
            bl=beamLine,
            center=(
                source.source_offset_x,
                source.source_offset_y,
                source.source_offset_z,
            ),
            dx=source.dx,
            dz=source.dz,
            dy=source.dy,
            distxprime=r"annulus",
            dxprime=[source.min_tth, source.max_tth],
            distzprime=r"flat",
            dzprime=[np.pi / 2 - delta_phi, np.pi / 2 + delta_phi],
            distE="normal",
            energies=[
                source.E_incident,
                source.E_incident * source.E_hwhm,
            ],
            pattern=reference_pattern,
            nrays=sim.nrays,
            horizontal_divergence=source.h_div,
            vertical_divergence=source.v_div,
        )
        # TODO switch to plates
        beamLine.screen_main = rscreens.Screen(
            bl=beamLine, center=[0, 150, r"auto"], name="main"
        )

        placements = compute_arm_positions(
            arm_tth, analyzer, theta_b, diffractometer
        )
        for j, p in enumerate(placements):
            setattr(
                beamLine,
                f"oe{j:02d}",
                roes.OE(
                    name=f"cry{j:02d}",
                    bl=beamLine,
                    center=list(p.crystal_center),
                    pitch=p.crystal_pitch,
                    positionRoll=np.pi,
                    material=crystalSi01,
                    limPhysX=[-analyzer.cry_width / 2, analyzer.cry_width / 2],
                    limPhysY=[-analyzer.cry_depth / 2, analyzer.cry_depth / 2],
                    extraRoll=np.deg2rad(analyzer.roll),
                ),
            )
            setattr(
                beamLine,
                f"baffle{j:02d}",
                RectangularBeamstop(
                    name=f"baffle{j:02d}",
                    bl=beamLine,
                    opening=[
                        -analyzer.cry_width / 2,
                        analyzer.cry_width / 2,
                        -0.7 * analyzer.Rd / 2,
                        0.7 * analyzer.Rd / 2,
                    ],
                    center=list(p.baffle_center),
                    z=tuple(p.baffle_z),
                ),
            )
            setattr(
                beamLine,
                f"screen{j:02d}",
                rscreens.Screen(
                    bl=beamLine,
                    center=list(p.screen_center),
                    x=(1, 0, 0),
                    z=tuple(p.screen_z),
                ),
            )

        return cls(beamLine, analyzer, source, detector, sim, diffractometer)

    @property
    def crystals(self):
        return [oe for oe in self.bl.oes if oe.name.startswith("cry")]

    @property
    def baffles(self):
        return [oe for oe in self.bl.slits if oe.name.startswith("baffle")]

    def set_arm(self, arm_tth: float):
        crystals = self.crystals
        baffles = self.baffles
        screens = self.bl.screens[1:]

        theta_b = _bragg(crystals[0].material, self.source.E_incident)

        placements = compute_arm_positions(
            arm_tth, self.analyzer, theta_b, self.diffractometer
        )
        for p, cry, baffle, screen in zip(
            placements, crystals, baffles, screens, strict=True
        ):
            cry.center = list(p.crystal_center)
            cry.pitch = p.crystal_pitch

            baffle.center = list(p.baffle_center)
            baffle.z = tuple(p.baffle_z)

            screen.center = list(p.screen_center)
            screen.z = tuple(p.screen_z)

    def run_process(self):
        # "raw" beam
        beamLine = self.bl
        geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()
        screen01beamLocal01 = beamLine.screen_main.expose(
            beam=geometricSource01beamGlobal01
        )

        outDict = {
            "source": geometricSource01beamGlobal01,
            "source_screen": screen01beamLocal01,
        }
        N = len([oe for oe in beamLine.oes if oe.name.startswith("cry")])

        for j in range(N):
            oeglobal, oelocal = getattr(beamLine, f"oe{j:02d}").reflect(
                beam=geometricSource01beamGlobal01
            )
            outDict[f"cry{j:02d}_local"] = oelocal
            outDict[f"cry{j:02d}_global"] = oeglobal

            outDict[f"baffle{j:0d}_local"] = getattr(
                beamLine, f"baffle{j:02d}"
            ).propagate(beam=oeglobal)

            outDict[f"screen{j:02d}"] = getattr(beamLine, f"screen{j:02d}").expose(
                beam=oeglobal
            )

        return {k: v for k, v in outDict.items() if k.startswith("screen")}

    def get_frames(self):
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

        out = {}
        _, yedges, xedges = np.histogram2d([], [], bins=shape, range=limits)
        for k, lb in screen_beams.items():
            # print(lb.x, lb.y, lb.z, lb.state)
            inner = {}
            for kt, good in zip(
                ("good",),
                (((lb.state == 1) | (lb.state == 2)),),
                strict=True,
            ):
                if isScreen:
                    x, y = lb.x[good], lb.z[good]
                else:
                    x, y = lb.x[good], lb.y[good]

                flux = lb.Jss[good] + lb.Jpp[good]
                hist2d, yedges, xedges = np.histogram2d(
                    y, x, bins=shape, range=limits, weights=flux
                )
                inner[kt] = hist2d
            out[k] = inner

        return out, yedges, xedges

    def get_free_ray_image(self):
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

        states = np.vstack([v.state for v in screen_beams.values()])
        (free_ray_indx,) = np.where((states == 3).sum(axis=0) == len(screen_beams))
        free_rays = np.zeros(states.shape[1], dtype=bool)
        free_rays[free_ray_indx] = True
        out = {}

        for k, lb in screen_beams.items():
            # print(lb.x, lb.y, lb.z, lb.state)
            inner = {}
            for kt, good in zip(
                ("bad",),
                (free_rays,),
                strict=True,
            ):
                if isScreen:
                    x, y = lb.x[good], lb.z[good]
                else:
                    x, y = lb.x[good], lb.y[good]

                flux = lb.Jss[good] + lb.Jpp[good]
                hist2d, yedges, xedges = np.histogram2d(
                    y, x, bins=shape, range=limits, weights=flux
                )
                inner[kt] = hist2d
            out[k] = inner

        return out, yedges, xedges


@functools.lru_cache(50)
def _bragg(mat, energy):
    return mat.get_Bragg_angle(energy)
