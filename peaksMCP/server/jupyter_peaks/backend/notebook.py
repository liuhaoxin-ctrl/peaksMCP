"""Read-only notebook and namespace operations."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import xarray as xr

from ..active_cell_bridge import CommBridge
from .base import SharedState

# Single source of truth for the cached-cell-output cap (shared with the bridge).
_MAX_CACHED_CELL_OUTPUTS = CommBridge.MAX_CACHED_CELLS


def _json_value(value: Any, limit: int = 80) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, limit) for item in value[:limit]]
    if isinstance(value, dict):
        return {str(key): _json_value(item, limit) for key, item in list(value.items())[:limit]}
    return repr(value)[:500]


def _storage_is_in_memory(value: xr.DataArray | xr.Dataset) -> bool:
    """Report whether all backing storage is already resident in memory.

    Inspects xarray's internal storage type only (``Variable._in_memory``).
    It must never touch ``.data``/``.values``: on a lazy backend array those
    properties call ``get_duck_array()`` and materialize the whole disk-backed
    variable (extra reads, memory and CPU) just to answer the query.
    """
    variables = (
        value.variables.values()
        if isinstance(value, xr.Dataset)
        else (value.variable,)
    )
    return all(bool(getattr(variable, "_in_memory", True)) for variable in variables)


def _peaks_api_names(value: Any) -> list[str]:
    """Find Peaks-owned xarray descriptors without invoking their getters."""
    names: set[str] = set()
    for cls in type(value).__mro__:
        for name, descriptor in vars(cls).items():
            candidates = (
                descriptor,
                getattr(descriptor, "_accessor", None),
                getattr(descriptor, "func", None),
            )
            modules = {
                str(getattr(candidate, "__module__", ""))
                for candidate in candidates
                if candidate is not None
            }
            module_name = str(getattr(descriptor, "module_name", ""))
            if module_name:
                modules.add(module_name)
            if any(module == "peaks" or module.startswith("peaks.") for module in modules):
                names.add(name)
    return sorted(names)


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
    peaks_apis = _peaks_api_names(value)
    # Dataset.chunks returns an empty dict even when no variable is chunked;
    # normalize it to None so "chunks is not None" means real chunking.
    chunks = getattr(value, "chunks", None) or None
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
        "peaks_apis": peaks_apis,
        "lazy": chunks is not None or not _storage_is_in_memory(value),
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
                    outputs = result.get("outputs")
                    cell_id = result.get("id")
                    if isinstance(outputs, list):
                        self.state.active_cell_output = list(outputs)
                        if isinstance(cell_id, str) and cell_id:
                            self.state.cell_outputs.pop(cell_id, None)
                            self.state.cell_outputs[cell_id] = list(outputs)
                            while (
                                len(self.state.cell_outputs)
                                > _MAX_CACHED_CELL_OUTPUTS
                            ):
                                oldest = next(iter(self.state.cell_outputs))
                                self.state.cell_outputs.pop(oldest, None)
            except Exception:
                pass
        return dict(self.state.active_cell)

    def active_cell_output(self) -> dict[str, Any]:
        cell_id = self.state.active_cell.get("id")
        if isinstance(cell_id, str) and cell_id in self.state.cell_outputs:
            return {
                "cell_id": cell_id,
                "outputs": list(self.state.cell_outputs[cell_id]),
            }
        return {"cell_id": cell_id, "outputs": list(self.state.active_cell_output)}

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
            "uptime_s": round(time.time() - self.state.started_at, 3),
            "kernel_instance_id": self.state.kernel_instance_id,
            "mcp_instance_id": self.state.mcp_instance_id,
            "extension_loaded": True,
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
