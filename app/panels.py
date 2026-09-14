"""Plotly figure builders for the dashboard.

Kept free of Streamlit calls (no ``st.*``) so each figure can be built and,
if ever needed, tested in isolation — only ``dashboard.py`` talks to
Streamlit. Every function here takes already-computed arrays or a
:class:`~avrmap.render.Raster`; none of them load data, filter points, or
build grids themselves.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from avrmap.render import Raster
from avrmap.semantics import class_name, colorize

# Fixed seed so repeated dashboard reruns (e.g. toggling an unrelated widget)
# subsample the same points instead of visibly jittering the cloud each time.
_SUBSAMPLE_SEED = 0


def subsample_indices(n: int, max_points: int) -> np.ndarray:
    """Indices of a reproducible random subsample, or every index if n is small."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(_SUBSAMPLE_SEED)
    return rng.choice(n, size=max_points, replace=False)


def point_cloud_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: np.ndarray,
    *,
    colorscale: str | None,
    is_rgb: bool,
    colorbar_title: str,
    title: str,
    marker_size: float = 1.5,
) -> go.Figure:
    """A 3D point cloud, coloured either by a continuous scalar or by RGB.

    ``is_rgb`` selects the mode: True expects ``color`` as an (N, 3) uint8
    array (semantic colours), False expects a 1D scalar array with a
    ``colorscale`` (e.g. elevation or intensity).
    """
    marker = dict(size=marker_size, opacity=0.85)
    if is_rgb:
        marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in color]
    else:
        marker.update(
            color=color, colorscale=colorscale,
            colorbar=dict(title=colorbar_title), showscale=True,
        )
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker, hoverinfo="skip")]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x forward (m)", yaxis_title="y left (m)", zaxis_title="z up (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
    )
    return fig


def semantic_point_cloud_figure(x, y, z, sem: np.ndarray, classes: dict, title: str) -> go.Figure:
    """3D point cloud coloured by PandaSet semantic class.

    A legend-like hover label is not attempted per-point (Scatter3d has no
    per-point legend entries without one trace per class, which would be slow
    at these point counts); class names are shown in the class table alongside
    the plot instead.
    """
    rgb = colorize(sem)
    return point_cloud_figure(
        x, y, z, rgb, colorscale=None, is_rgb=True,
        colorbar_title="", title=title,
    )


def raster_scalar_figure(raster: Raster, colorscale: str, title: str, colorbar_title: str, zmin=None, zmax=None) -> go.Figure:
    """A continuous-valued raster (e.g. elevation), NaN cells left blank."""
    x = raster.coord_axis()
    y = raster.coord_axis()[::-1]  # row 0 is +extent_m; see avrmap.render
    fig = px.imshow(
        raster.image, x=x, y=y, origin="upper", color_continuous_scale=colorscale,
        zmin=zmin, zmax=zmax, aspect="equal",
    )
    fig.update_coloraxes(colorbar_title=colorbar_title)
    fig.update_layout(
        title=title, xaxis_title="x forward (m)", yaxis_title="y left (m)",
        margin=dict(l=0, r=0, t=30, b=0), height=480,
    )
    return fig


def raster_rgb_figure(raster: Raster, title: str) -> go.Figure:
    """An RGB raster (semantic classes)."""
    x = raster.coord_axis()
    y = raster.coord_axis()[::-1]
    fig = px.imshow(raster.image, x=x, y=y, origin="upper", aspect="equal")
    fig.update_layout(
        title=title, xaxis_title="x forward (m)", yaxis_title="y left (m)",
        margin=dict(l=0, r=0, t=30, b=0), height=480,
    )
    return fig


def raster_zone_figure(raster: Raster, n_zones: int, title: str) -> go.Figure:
    """Zone-id raster: which distance zone (and cell size) produced each pixel."""
    x = raster.coord_axis()
    y = raster.coord_axis()[::-1]
    # 0 = empty (rendered as the first colour but masked below); zones are 1..n_zones
    fig = px.imshow(
        raster.image.astype(np.float32), x=x, y=y, origin="upper",
        color_continuous_scale="Turbo", zmin=0, zmax=n_zones, aspect="equal",
    )
    fig.update_coloraxes(colorbar_title="zone (0 = empty)")
    fig.update_layout(
        title=title, xaxis_title="x forward (m)", yaxis_title="y left (m)",
        margin=dict(l=0, r=0, t=30, b=0), height=480,
    )
    return fig


def class_breakdown_table(sem_class: np.ndarray, classes: dict) -> list[dict]:
    """Class-id / name / count rows, sorted by count, for an ``st.dataframe``."""
    ids, counts = np.unique(sem_class, return_counts=True)
    order = np.argsort(-counts)
    return [
        {"class": int(ids[i]), "name": class_name(classes, int(ids[i])), "count": int(counts[i])}
        for i in order
    ]


def cost_comparison_figure(methods: list[str], cells: list[int], title: str = "Occupied cells by method") -> go.Figure:
    fig = go.Figure(data=[go.Bar(x=methods, y=cells, text=[f"{c:,}" for c in cells], textposition="outside")])
    fig.update_layout(title=title, yaxis_title="cells", margin=dict(l=0, r=0, t=30, b=0), height=340)
    return fig


def band_quality_figure(band_labels: list[str], series: dict[str, list[float]], y_title: str, title: str) -> go.Figure:
    """Grouped bars, one group per band, one bar per method, for RMSE or agreement."""
    fig = go.Figure()
    for method, values in series.items():
        fig.add_trace(go.Bar(name=method, x=band_labels, y=values))
    fig.update_layout(
        title=title, barmode="group", yaxis_title=y_title, xaxis_title="distance band",
        margin=dict(l=0, r=0, t=30, b=0), height=360, legend=dict(orientation="h", y=-0.2),
    )
    return fig


def add_resolution_zones(
    fig: go.Figure,
    zones,
    *,
    ring_color: str = "rgba(255,255,255,0.45)",
    label_color: str = "#ffffff",
    ring_width: float = 0.8,
    label_angle_rad: float = np.pi / 4,
    label_bgcolor: str = "rgba(0,0,0,0.45)",
) -> go.Figure:
    """Overlay the foveated resolution ladder onto a map figure.

    Draws one dashed white concentric circle per zone boundary (the ``r_max``
    of each zone, centred on the ego at the origin) plus a small, unobtrusive
    label showing just that zone's cell size, and a marker for the ego vehicle
    at the centre. Passing the *live* zone ladder means the overlay follows
    the dashboard's zone editor exactly.
    """
    theta = np.linspace(0, 2 * np.pi, 240)
    cx, cy = np.cos(theta), np.sin(theta)
    for zone in zones:
        fig.add_trace(
            go.Scatter(
                x=zone.r_max * cx, y=zone.r_max * cy, mode="lines",
                line=dict(color=ring_color, width=ring_width, dash="dot"),
                hoverinfo="skip", showlegend=False,
            )
        )
        mid = (zone.r_min + zone.r_max) / 2
        fig.add_annotation(
            x=mid * np.cos(label_angle_rad), y=mid * np.sin(label_angle_rad),
            text=f"{zone.cell_m:g} m",
            showarrow=False, font=dict(size=10, color=label_color),
            bgcolor=label_bgcolor, borderwidth=0, borderpad=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0], y=[0.0], mode="markers", name="ego vehicle",
            marker=dict(size=9, color="white", symbol="circle",
                        line=dict(color="#111111", width=1.5)),
            hoverinfo="skip",
        )
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.18))
    return fig
