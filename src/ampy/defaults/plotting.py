from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Mapping, Any


@dataclass(frozen=True, slots=True)
class BandColorMap:
    """
    Default color mapping for photometric bands.

    This class maps band or filter names to colors used when plotting
    photometric data. It also supports alias names that map to canonical
    band identifiers.
    """

    colors: Mapping[str, str] = field(default_factory=lambda: {
        # Johnson–Cousins Optical/NIR
        "U": "#8601AF",
        "B": "#0247FE",
        "V": "#66B032",
        "R": "#FE2712",
        "I": "#4424D6",
        "J": "#66B032",
        "H": "#FC600A",
        "K": "#FE2712",

        # SDSS Optical
        "u": "tab:purple",
        "g": "tab:blue",
        "r": "tab:orange",
        "i": "tab:red",
        "z": "tab:pink",

        # Swift UVOT / X-ray
        "uvot-u": "cyan",
        "uvot-b": "lightblue",
        "uvot-v": "lightgreen",
        "uvw2": "pink",
        "uvm2": "darkblue",
        "uvw1": "green",
        "xray": "black",

        # HST
        "F775W": "green",
        "F125W": "blue",

        # Radio / mm
        "C": "royalblue",
        "C2": "purple",
        "Ka": "peachpuff",
        "Kb": "peru",
        "Kc": "palevioletred",
        "Kd": "lightcoral",
        "W": "teal",
        "S": "teal",
    })

    aliases: Mapping[str, str] = field(default_factory=lambda: {
        "r2": "r",
        "i2": "i",
        "z2": "z",
        "Rc": "R",
        "Ic": "I",
        "Ks": "K",
        "uprime": "u",
        "gprime": "g",
        "rprime": "r",
        "iprime": "i",
        "zprime": "z",
    })

    default: str = "black"
    """Fallback color used when a band is not recognized."""

    def get(self, band: str) -> str:
        """
        Return the plotting color for a given band name.

        Parameters
        ----------
        band : str
            Band or filter identifier.

        Returns
        -------
        str
            Color associated with the band. If the band is unknown,
            the default color is returned.
        """
        key = self.aliases.get(band, band)
        return self.colors.get(key, self.default)


#<editor-fold desc="Corner Plot Defaults">
@dataclass(frozen=True, slots=True)
class CornerPlotDefaults:
    """
    Default configuration for corner plots.

    This class defines the default keyword arguments passed to
    :func:`corner.corner` when visualizing posterior samples.
    """

    label_size: int = 16
    """Base label size used by the corner plot (if supported by the caller)."""

    show_titles: bool = True
    """Whether to show summary statistics in titles on the diagonal."""

    color: str = "mediumblue"
    """Primary contour/line color used for the corner plot."""

    plot_datapoints: bool = False
    """Whether to draw individual sample points (can be slow/overplotted)."""

    quantiles: tuple[float, float, float] = (0.16, 0.5, 0.84)
    """Quantiles shown in titles (e.g., 16th/50th/84th percentiles)."""

    label_kwargs: dict[str, Any] = field(default_factory=lambda: {"fontsize": 14})
    """Keyword arguments forwarded to label text creation."""

    title_kwargs: dict[str, Any] = field(default_factory=lambda: {"fontsize": 14})
    """Keyword arguments forwarded to title text creation."""

    fill_contours: bool = True
    """Whether to fill contour levels."""

    smooth: float = 0.7
    """Smoothing applied to 2D histograms (KDE-ish; see ``corner`` docs)."""

    smooth1d: float = 0.7
    """Smoothing applied to 1D histograms (diagonal panels)."""

    def kwargs(self) -> dict[str, Any]:
        """
        Return keyword arguments compatible with :func:`corner.corner`.

        Returns
        -------
        dict
            Suitable for unpacking into ``corner.corner(..., **kwargs)``.
        """
        return {
            "show_titles": self.show_titles,
            "color": self.color,
            "plot_datapoints": self.plot_datapoints,
            "quantiles": list(self.quantiles),
            "label_kwargs": dict(self.label_kwargs),
            "title_kwargs": dict(self.title_kwargs),
            "fill_contours": self.fill_contours,
            "smooth": self.smooth,
            "smooth1d": self.smooth1d,
        }
#</editor-fold>


# <editor-fold desc="Spectral Plot Defaults">
@dataclass(frozen=True, slots=True)
class ObservedIndexStyle:
    """
    Default styling for observed spectral index points.

    This class defines the default keyword arguments passed to
    :meth:`matplotlib.axes.Axes.errorbar` when plotting observed
    spectral indices.
    """

    fmt: str = "o"
    """Marker style used for the observed points."""

    linestyle: str = "none"
    """Line style connecting points (typically disabled)."""

    capsize: float = 3.0
    """Size of the error bar caps, in points."""

    color: str = "black"
    """Color used for the observed data points."""

    zorder: int = 999
    """Z-order of the observed points (drawn above most elements)."""

    def kwargs(self) -> dict:
        """
        Return the style as a dictionary of keyword arguments.

        Returns
        -------
        dict
            Dictionary suitable for unpacking into a Matplotlib
            plotting call (e.g., ``ax.errorbar(**kwargs)``).
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpectralLegendStyle:
    """
    Default configuration for spectral plot legends.

    This style is intended for figure-level legends placed above the
    multi-panel spectral plot.
    """

    mode: str = "expand"
    """If mode is set to "expand" the legend will be horizontally expanded to 
    fill the Axes area (or bbox_to_anchor if defines the legend's size)."""

    loc: str = "upper center"
    """Legend location string passed to Matplotlib."""

    bbox_to_anchor: tuple[float, float, float, float] = (.09, .97, 0.9, .01)
    """Bounding box anchor for the legend in figure coordinates."""

    frameon: bool = True
    """Whether to draw a frame around the legend."""

    fancybox: bool = False
    """Whether to use a rounded legend box."""

    edgecolor: str = "black"
    """Color of the legend frame."""

    def kwargs(self) -> dict:
        """Return legend keyword arguments."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpectralLayout:
    """
    Default configuration for spectral subplots.
    """

    figsize: tuple[float, float] = (8.0, 10.0)
    """The (width, height) in inches."""

    height_ratios: tuple[float, float] = (3.0, 1.0)
    """Ratio of the subplot heights."""

    sharex: bool = True
    """Whether to share the x-axis between subplots."""

    def kwargs(self) -> dict:
        """ Matplotlib wants gridspec_kw as a dict. """
        return {
            "figsize": self.figsize,
            "sharex": self.sharex,
            "gridspec_kw": {"height_ratios": list(self.height_ratios)},
        }


@dataclass(frozen=True, slots=True)
class SpectralColors:
    """
    Default configuration for the frequency curve colors.
    """
    nu_a: str = "tab:green"
    """Color of the synchrotron self-absorption curves."""

    nu_m: str = "tab:blue"
    """Color of the synchrotron frequency curves."""

    nu_c: str = "tab:orange"
    """Color of the cooling frequency curves."""

    def mapping(self) -> Mapping[str, str]:
        return {"nu_a": self.nu_a, "nu_m": self.nu_m, "nu_c": self.nu_c}
# </editor-fold>


# <editor-fold desc="Density Profile Defaults">
@dataclass(frozen=True, slots=True)
class DensityDistributionStyle:
    """
    Default styling for sampled or distribution density profiles.

    This style is used when plotting an ensemble of density profiles
    (e.g., posterior samples or uncertainty bands).
    """

    alpha: float = 0.3
    """Transparency applied to individual density profile curves."""

    linewidth: float = 0.5
    """Line width of individual density profile curves."""

    linestyle: str = "-"
    """Line style used for distribution curves."""

    color: str = "tab:purple"
    """Color used for the density profile distribution."""

    def kwargs(self) -> dict:
        """
        Return the style as Matplotlib keyword arguments.

        Returns
        -------
        dict
            Dictionary suitable for unpacking into a Matplotlib
            line plotting call (e.g., ``ax.plot(**kwargs)``).
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DensityBestStyle:
    """
    Default styling for the most-likely (best-fit) density profile.

    This style is intended to visually distinguish the best-fit
    density profile from the broader distribution.
    """

    linewidth: float = 2.0
    """Line width of the best-fit density profile curve."""

    linestyle: str = "-"
    """Line style used for the best-fit density profile."""

    color: str = "tab:orange"
    """Color used for the best-fit density profile."""

    label: str = "Most Likely"
    """Legend label for the best-fit density profile."""

    def kwargs(self) -> dict:
        """
        Return the style as Matplotlib keyword arguments.

        Returns
        -------
        dict
            Dictionary suitable for unpacking into a Matplotlib
            line plotting call (e.g., ``ax.plot(**kwargs)``).
        """
        return asdict(self)
# </editor-fold>


@dataclass(frozen=True, slots=True)
class SpectralPlotDefaults:
    """
    Default plotting configuration for spectral figures.

    This class groups all default styling and layout options used when
    generating spectral plots in AMPy. Users may override any subset of these
    defaults by constructing a new instance.
    """

    observed: ObservedIndexStyle = field(default_factory=ObservedIndexStyle)
    """Defaults for observed spectral index markers."""

    legend: SpectralLegendStyle = field(default_factory=SpectralLegendStyle)
    """Defaults for the figure-level legend."""

    layout: SpectralLayout = field(default_factory=SpectralLayout)
    """Defaults for subplot layout and sizing."""

    colors: SpectralColors = field(default_factory=SpectralColors)
    """Default colors for spectral break curves."""


@dataclass(frozen=True, slots=True)
class DensityProfileDefaults:
    """
    Default plotting configuration for density profile figures.

    This class groups all default styling options used when plotting
    circumburst density profiles in AMPy.
    """

    distribution: DensityDistributionStyle = field(
        default_factory=DensityDistributionStyle
    )
    """Defaults for sampled or distribution density profile curves."""

    best: DensityBestStyle = field(default_factory=DensityBestStyle)
    """Defaults for the most-likely (best-fit) density profile curve."""
