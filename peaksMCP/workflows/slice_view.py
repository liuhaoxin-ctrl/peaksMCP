"""View a slice of a multidimensional (mapping) DataArray as a figure.

Contract (read this before calling):
- One verb per call: pick the workflow, give parameters, read the one-line
  print and the returned dict.
- Every figure is rendered inline in the notebook for the user. The model
  receives only the printed line; image pixels stay in the notebook.
- ``debug=True`` is a user-requested escape hatch only: it adds diagnostic
  prints. Enable it only when the user asks you to debug; it is a door for
  user-requested inspection of internals.
- No progress bars, no plots written to disk.

Language style: plain technical English; errors name the function, the
offending value and the fix in one sentence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _present_figure(fig: Any) -> None:
    """Render the figure inline when running inside IPython/Jupyter.

    The figure is always returned for programmatic reuse; display is skipped
    under headless backends so library tests stay warning-free.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            display(fig)
            return
    except Exception:
        pass
    # Headless/scripted use: nothing to present; the caller owns the figure.


def _coord_label(da: xr.DataArray, dim: str) -> str:
    """Human axis label: coordinate name plus its unit, if recorded."""
    unit = da[dim].attrs.get("units") or da[dim].attrs.get("unit")
    return f"{dim} [{unit}]" if unit else dim


def _resolve_slice_index(da: xr.DataArray, dim: str, index: int | str | None) -> int:
    """Return the integer position for ``index`` ('middle' = central slice)."""
    size = da.sizes[dim]
    if index is None or index == "middle":
        return size // 2
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(
            f"show_mapping_slice: index must be an int or 'middle', got {index!r}."
        )
    if not -size <= index < size:
        raise ValueError(
            f"show_mapping_slice: index {index} out of range for dim "
            f"'{dim}' (size {size})."
        )
    return index % size


def show_mapping_slice(
    data: xr.DataArray,
    *,
    dim: str,
    index: int | str | None = "middle",
    title: str | None = None,
    cmap: str = "viridis",
    debug: bool = False,
) -> dict[str, Any]:
    """Render one slice of a mapping DataArray and print a one-line report.

    Slice ``data`` along ``dim`` at ``index`` (default: central slice). The
    remaining dimensions become the axes: two dimensions render a 2-D map,
    one dimension renders a curve. The figure appears inline in the notebook
    for the user; the model sees the printed line only.

    Parameters
    ----------
    data : xarray.DataArray
        Mapping with >= 2 dimensions (e.g. ``(eV, x, y)`` or ``(eV, theta,
        deflector_perp)``).
    dim : str
        Dimension to slice (an energy axis gives a real-space/angle map; a
        spatial axis gives an EDC-style curve).
    index : int or "middle", default "middle"
        Position along ``dim`` (negative counts from the end).
    title : str, optional
        Optional figure title. Defaults to the dim/label slice summary.
    cmap : str, default "viridis"
        Colormap for 2-D slices.
    debug : bool, default False
        User-requested diagnostics only: prints extra structure lines.

    Returns
    -------
    dict
        ``{"status": "ok", "verb": "show_mapping_slice", "slice": {...},
        "rendered": [ax0_size, ax1_size], "figure": <matplotlib Figure>}``.

    Examples
    --------
    >>> show_mapping_slice(map3d, dim="eV")          # central energy map
    >>> show_mapping_slice(map3d, dim="eV", index=0) # first energy map
    >>> show_mapping_slice(map3d, dim="y", index=12) # one column as a curve
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            "show_mapping_slice: data must be an xarray.DataArray, got "
            f"{type(data).__name__}."
        )
    try:
        data = data.pint.dequantify()  # drop units before array casts
    except Exception:
        pass
    if dim not in data.dims:
        raise ValueError(
            f"show_mapping_slice: dim {dim!r} not in data dims "
            f"{list(data.dims)}. Pick one of those."
        )
    position = _resolve_slice_index(data, dim, index)
    slice_ = data.isel({dim: position})
    slice_.load()  # small by design; explicit so a lazy backend never surprises
    remaining = list(slice_.dims)
    if len(remaining) > 2:
        raise ValueError(
            f"show_mapping_slice: slicing '{dim}' leaves {len(remaining)} "
            f"dimensions ({remaining}); slice along one more axis first."
        )
    if debug:
        print(
            f"show_mapping_slice[debug]: dims={list(data.dims)} "
            f"sizes={dict(data.sizes)} slice_shape={list(slice_.shape)}"
        )
    slice_coord = slice_.coords.get(dim)
    label = float(slice_coord) if slice_coord is not None and slice_coord.size == 1 else position
    import matplotlib.pyplot as plt

    fig_title = title or f"{dim} = {label:.4g}"
    if len(remaining) == 2:
        x_name, y_name = remaining[1], remaining[0]
        x = np.asarray(slice_[x_name].data, dtype=float)
        y = np.asarray(slice_[y_name].data, dtype=float)
        z = np.asarray(slice_.data, dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        im = ax.pcolormesh(x, y, z, cmap=cmap, shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xlabel(_coord_label(slice_, x_name))
        ax.set_ylabel(_coord_label(slice_, y_name))
        ax.set_title(fig_title)
        rendered = [z.shape[1], z.shape[0]]
        kind = "2-D map"
    else:
        curve_dim = remaining[0]
        x = np.asarray(slice_[curve_dim].data, dtype=float)
        z = np.asarray(slice_.data, dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(x, z, lw=1.0)
        ax.set_xlabel(_coord_label(slice_, curve_dim))
        ax.set_ylabel("intensity")
        ax.set_title(fig_title)
        rendered = [z.size]
        kind = "curve"
    if debug:
        print(
            f"show_mapping_slice[debug]: figure built {kind}; "
            f"z-range [{float(z.min()):.4g}, {float(z.max()):.4g}]"
        )
    _present_figure(fig)  # inline in the notebook for the user
    print(
        f"show_mapping_slice: {kind} at {dim}={label:.4g} "
        f"(index {position}) rendered for the user; shape {rendered}."
    )
    return {
        "status": "ok",
        "verb": "show_mapping_slice",
        "slice": {"dim": dim, "position": position, "label": label},
        "rendered": rendered,
        "figure": fig,
    }
