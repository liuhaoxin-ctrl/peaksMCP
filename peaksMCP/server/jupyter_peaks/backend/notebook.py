"""Read-only notebook and namespace operations."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import xarray as xr

from .base import SharedState


def _json_value(value: Any, limit: int = 80) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, limit) for item in value[:limit]]
    if isinstance(value, dict):
        return {str(key): _json_value(item, limit) for key, item in list(value.items())[:limit]}
    return repr(value)[:500]


def summarize_xarray(value: xr.DataArray | xr.Dataset | xr.DataTree) -> dict[str, Any]:
    """Describe xarray structure without materializing lazy array values.

    Parameters
    ----------
    value : xarray.DataArray, xarray.Dataset, or xarray.DataTree
        Live notebook object whose metadata and lazy structure should be inspected.

    Returns
    -------
    dict
        JSON-safe dims, sizes, dtypes, coordinates, units, attrs, chunks and Peaks accessors.

    Examples
    --------
    >>> import xarray as xr
    >>> summarize_xarray(xr.DataArray([1, 2], dims="eV"))["sizes"]
    {'eV': 2}
    """
    if isinstance(value, xr.DataTree):
        return {
            "type": "xarray.DataTree",
            "name": value.name,
            "groups": [node.path for node in value.subtree],
            "attrs": _json_value(dict(value.attrs)),
        }
    variables = value.data_vars if isinstance(value, xr.Dataset) else {value.name or "data": value}
    coordinates: dict[str, Any] = {}
    for name, coordinate in value.coords.items():
        coordinates[name] = {
            "dims": list(coordinate.dims),
            "size": int(coordinate.size),
            "dtype": str(coordinate.dtype),
            "units": coordinate.attrs.get("units") or coordinate.attrs.get("unit"),
            "attrs": _json_value(dict(coordinate.attrs)),
        }
    accessors = [name for name in ("S", "T", "F", "G", "M", "k", "spatial", "tr", "sym") if hasattr(value, name)]
    chunks = getattr(value, "chunks", None)
    return {
        "type": f"xarray.{type(value).__name__}",
        "name": getattr(value, "name", None),
        "dims": list(value.dims),
        "sizes": {str(key): int(item) for key, item in value.sizes.items()},
        "dtype": str(value.dtype) if isinstance(value, xr.DataArray) else None,
        "variables": {
            str(name): {"dims": list(array.dims), "dtype": str(array.dtype), "units": array.attrs.get("units") or array.attrs.get("unit")}
            for name, array in variables.items()
        },
        "coords": coordinates,
        "units": value.attrs.get("units") or value.attrs.get("unit"),
        "attrs": _json_value(dict(value.attrs)),
        "chunks": _json_value(chunks),
        "peaks_accessors": accessors,
        "lazy": chunks is not None or hasattr(getattr(value, "data", None), "dask"),
    }


class NotebookBackend:
    """Read notebook state and live variables from a shared IPython namespace."""

    def __init__(self, state: SharedState) -> None:
        self.state = state

    def list_variables(self) -> dict[str, Any]:
        variables = []
        for name, value in sorted(self.state.namespace.items()):
            if name.startswith("_") or name in {"In", "Out", "get_ipython", "exit", "quit", "open"}:
                continue
            item = {"name": name, "type": f"{type(value).__module__}.{type(value).__name__}"}
            if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
                item.update({"dims": list(value.dims) if not isinstance(value, xr.DataTree) else None, "sizes": dict(value.sizes) if not isinstance(value, xr.DataTree) else None})
            elif isinstance(value, np.ndarray):
                item.update({"shape": list(value.shape), "dtype": str(value.dtype)})
            variables.append(item)
        return {"variables": variables, "count": len(variables)}

    def read_variable(self, name: str) -> dict[str, Any]:
        if name not in self.state.namespace:
            raise KeyError(f"variable {name!r} does not exist")
        value = self.state.namespace[name]
        if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
            return {"name": name, **summarize_xarray(value)}
        if isinstance(value, np.ndarray):
            return {"name": name, "type": "numpy.ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "size": int(value.size)}
        return {"name": name, "type": f"{type(value).__module__}.{type(value).__name__}", "repr": repr(value)[:4000]}

    def active_cell(self) -> dict[str, Any]:
        if self.state.bridge and self.state.bridge.connected:
            try:
                result = self.state.bridge.request("read_active_cell", timeout=5)
                if isinstance(result, dict):
                    self.state.active_cell = result
            except Exception:
                pass
        return dict(self.state.active_cell)

    def active_cell_output(self) -> dict[str, Any]:
        return {"outputs": list(self.state.active_cell_output)}

    def notebook_content(self) -> dict[str, Any]:
        if not self.state.bridge:
            raise RuntimeError("JupyterLab Comm bridge is not connected")
        return self.state.bridge.request("read_notebook")

    def move_cursor(self, direction: str = "next", index: int | None = None) -> dict[str, Any]:
        if direction not in {"next", "previous", "index"}:
            raise ValueError("direction must be next, previous, or index")
        if direction == "index" and index is None:
            raise ValueError("index is required when direction='index'")
        return self.state.bridge.request("move_cursor", {"direction": direction, "index": index})

    def server_status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "mode": self.state.mode.value,
            "uptime_s": round(time.time() - self.state.started_at, 3),
            "comm_connected": bool(self.state.bridge and self.state.bridge.connected),
            "api_index_ready": self.state.api_index is not None,
            "api_count": len(self.state.api_index.entries) if self.state.api_index else 0,
            "index_stale": bool(self.state.api_index and self.state.api_index.is_stale()),
        }

    def kernel_status(self) -> dict[str, Any]:
        return {"state": self.state.kernel_state, "busy_since": self.state.busy_since}

    def wait_for_kernel(self, timeout: float = 30.0, poll_interval: float = 0.1) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self.state.kernel_state == "idle":
                return {"ready": True, **self.kernel_status()}
            time.sleep(max(0.02, poll_interval))
        return {"ready": False, "timed_out": True, **self.kernel_status()}
