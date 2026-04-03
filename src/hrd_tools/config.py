from dataclasses import dataclass
from pathlib import Path

# TODO look into this game
# from typing import NewType
# rad = NewType("rad", float)
# mm = NewType("mm", float)


@dataclass(frozen=True)
class AnalyzerConfig:
    # sample to (central) crystal
    R: float
    # crystal to detector distance
    Rd: float
    # angular offset between crystals in rad
    cry_offset: float
    # crystal width (transverse to beam) in mm
    cry_width: float
    # crystal depth (direction of beam) in mm
    cry_depth: float
    # number of crystals
    N: int
    # incident angle of crystals in rad
    incident_angle: float
    # thickness of crystals in mm
    thickness: float
    # roll of the analyzer crystals in deg
    roll: float = 0


@dataclass(frozen=True)
class DetectorConfig:
    # pixel pitch in mm
    pitch: float
    # pixel width in transverse direction
    transverse_size: int
    # Size of active area in direction of beam in mm
    height: float


@dataclass(frozen=True)
class SimConfig:
    # number of rays in the simulation
    nrays: int


@dataclass(frozen=True)
class SourceConfig:
    # Energy
    E_incident: float
    # location of source pattern
    # TODO maybe make actually pattern?
    pattern_path: Path
    # width of source in mm
    dx: float
    # height of source in mm
    dz: float
    # depth of source in mm
    dy: float
    # phi range around vertical to simulate in Deg
    delta_phi: float
    # energy bandwidth
    E_hwhm: float
    # vertical divergence in Deg
    v_div: float = 0
    # horizontal divergence in Deg
    h_div: float = 0
    # minimum allowed tth generated in Deg
    min_tth: float = 0
    # maximum allowed tth generated in Deg
    max_tth: float = 180
    # off set of source from the true center of rotation of the crystals
    source_offset_x: float = 0
    source_offset_y: float = 0
    source_offset_z: float = 0


@dataclass(frozen=True)
class SimScanConfig:
    # start angle in deg
    start: float
    # stop angle in deg
    stop: float
    # delta between positions in deg
    delta: float
    # short description about why we did this
    short_description: str = ""


@dataclass(frozen=True)
class Diffractometer:
    # rotation about the y-axis (beam direction) in degrees
    roll: float = 0.0
    # rotation about the z-axis (vertical) in degrees
    yaw: float = 0.0
    # displacement of center of rotation from source center in mm
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


@dataclass(frozen=True)
class CompleteConfig:
    source: SourceConfig
    sim: SimConfig
    detector: DetectorConfig
    analyzer: AnalyzerConfig
    scan: SimScanConfig
    diffractometer: Diffractometer = Diffractometer()


@dataclass(frozen=True)
class AnalyzerCalibration:
    # the location (in pixels) of the center of the beam on the detector
    detector_centers: tuple[float, ...]
    # offsets from "0" on arm of each crystal in deg
    psi: tuple[float, ...]
    # roll of detector about direction of beam in deg
    roll: tuple[float, ...]


@dataclass(frozen=True)
class GeneratorInvocation:
    cycler: str
    uuid: str
    md5sum: str
