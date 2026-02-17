from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CalibrationDefaults:
    """
    Default calibration offset priors by band/group name.

    Keys correspond to values used in the CalGroup column of the
    observation file.
    """

    offsets: Dict[str, dict] = field(default_factory=lambda: {
        "B": dict(
            name="B_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
        "V": dict(
            name="V_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "R": dict(
            name="R_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "I": dict(
            name="I_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "J": dict(
            name="J_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
        "H": dict(
            name="H_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "K": dict(
            name="K_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.06},
        ),
        "g": dict(
            name="g_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "r": dict(
            name="r_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "i": dict(
            name="i_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
        "z": dict(
            name="z_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "uvw1": dict(
            name="uvw1_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
        "uvw2": dict(
            name="uvw2_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
        "uvm2": dict(
            name="uvm2_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "uvot-u": dict(
            name="uvot-u_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "uvot-b": dict(
            name="uvot-b_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.02},
        ),
        "uvot-v": dict(
            name="uvot-v_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.02},
        ),
        "F775W": dict(
            name="F775W_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.03},
        ),
        "F125W": dict(
            name="F125W_offset",
            prior={"type": "gaussian", "mu": 0.0, "sigma": 0.04},
        ),
    })

    def get(self, band: str) -> dict:
        """Return default ParamSpec for a calibration band."""
        try:
            return self.offsets[band]
        except KeyError:
            return self.offsets[band.replace('_offset', '')]


@dataclass
class PowerLawDefaults:
    """

    """

    params: Dict = field(default_factory=lambda: {
        "E52": dict(
            name="E52",
            infer_scale="log",
            model_scale="linear",
            prior=dict(type="uniform", lower=-2.0, upper=4.0)
        ),
        "lf0": dict(
            name="lf0",
            infer_scale="log",
            model_scale="linear",
            prior=dict(type="uniform", lower=1.69, upper=3.0)
        ),
        "n017": dict(
            name="n017",
            infer_scale="log",
            model_scale="linear",
            prior=dict(type="uniform", lower=-6.0, upper=6.0)
        ),
        "eps_e": dict(
            name="eps_e",
            infer_scale="log",
            model_scale="linear",
            prior=dict(type="uniform", lower=-6.0, upper=0.0)
        ),
        "eps_b": dict(
            name="eps_b",
            infer_scale="log",
            model_scale="linear",
            prior=dict(type="uniform", lower=-6.0, upper=0.0)
        ),
        "p": dict(
            name="p",
            prior=dict(type="uniform", lower=2.0, upper=3.0)
        ),
        "k": dict(
            name="k",
            prior=dict(type="uniform", lower=-3.0, upper=3.0)
        ),
        "hmf": dict(
            name="hmf",
            prior=dict(value=0.7)
        )
    })
