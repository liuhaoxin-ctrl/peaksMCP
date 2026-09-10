"""Read-only notebook and namespace operations.

Object summaries follow a generic protocol so no peaksMCP class gets a
special case: xarray structures use :func:`summarize_xarray`; numpy arrays
their shape/dtype; LoadedScans-like indices (anything whose entries carry
``representation``/``stem`` and that exposes ``stems``) are summarized
structurally (representations, conversion state, provenance) — never
classified (classification is inspect_experiment's job); everything else
falls back to a bounded repr.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import xarray as xr

from .base import SharedState

#: Bounds enforced on every object summary (the model never sees unbounded
#: listings or reprs through the notebook tools).
_SUMMARY_LINE_MAX = 160
_STEMS_MAX = 200
_REPR_MAX = 4000
_LIST_VARIABLES_LIMIT = 50
_ACTIVE_CELL_SOURCE_MAX = 4000
#: 单 cell 纯文本输出回读上限（text 历史只读，图片永不回传）。
_TEXT_OUTPUT_MAX = 8000


def extract_text_outputs(outputs: Any, limit: int = _TEXT_OUTPUT_MAX) -> tuple[str, bool]:
    """把一个 cell 的 text 类输出拼成有界字符串（回读用）。

    只取 stream（stdout/stderr）与 display/execute 的 ``text/plain``；
    图片/交互组件等 payload 永不进入文本回读。返回 ``(text, truncated)``。
    """
    chunks: list[str] = []
    if not isinstance(outputs, list):
        return "", False
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if output.get("output_type") == "stream":
            text = output.get("text") or []
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(text, list):
                chunks.extend(str(part) for part in text)
            continue
        data = output.get("data") or {}
        if not isinstance(data, dict):
            continue
        plain = data.get("text/plain")
        if plain:
            chunks.append("".join(plain) if isinstance(plain, list) else str(plain))
    text = "".join(chunks)
    truncated = len(text) > limit
    return text[:limit], truncated


def _json_value(value: Any, limit: int = 80) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, limit) for item in value[:limit]]
    if isinstance(value, dict):
        return {str(key): _json_value(item, limit) for key, item in list(value.items())[:limit]}
    return repr(value)[:500]


def _units_text(attrs: dict[str, Any]) -> str | None:
    """Text form of a units attribute.

    Peaks data carries pint objects (``Unit`` / ``Quantity``) in ``units``;
    they are not JSON-serialisable, and a structured tool reply that cannot be
    serialised degrades to unstructured text, which strict MCP clients reject.
    """
    units = attrs.get("units") or attrs.get("unit")
    return None if units is None else str(units)


def _variable_is_in_memory(variable: Any) -> bool:
    """Whether one xarray Variable's storage is resident in memory.

    xarray reports pint-backed variables as *not* in memory (a pint Quantity is
    a duck array), which mislabelled fully loaded Peaks data as lazy.  The
    wrapped magnitude decides instead: a numpy magnitude is in memory, a dask
    magnitude stays lazy.  Non-pint backends keep xarray's own answer.
    """
    data = getattr(variable, "_data", None)
    magnitude = getattr(data, "magnitude", None)
    if magnitude is not None:
        return not (
            getattr(magnitude, "chunks", None) is not None
            or hasattr(magnitude, "__dask_graph__")
        )
    return bool(getattr(variable, "_in_memory", True))


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
    return all(_variable_is_in_memory(variable) for variable in variables)


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
            "units": _units_text(coordinate.attrs),
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
            str(name): {"dims": list(array.dims), "dtype": str(array.dtype), "units": _units_text(array.attrs)}
            for name, array in variables.items()
        },
        "coords": coordinates,
        "units": _units_text(value.attrs),
        "attrs": _json_value(dict(value.attrs)),
        "chunks": _json_value(chunks),
        "peaks_apis": peaks_apis,
        "lazy": chunks is not None or not _storage_is_in_memory(value),
    }


# --------------------------------------------------------------------------- #
# Generic object-summary protocol (no LoadedScans special case)                #
# --------------------------------------------------------------------------- #

def _as_index(value: Any) -> Any | None:
    """Detect a LoadedScans-like index by protocol shape (duck-typed).

    Anything exposing an ``entries`` sequence whose items carry
    ``stem``/``representation`` plus a ``stems`` view is summarized through
    the generic index protocol — no peaksMCP import, no special case.
    """
    entries = getattr(value, "entries", None)
    if isinstance(entries, (list, tuple)) and entries and hasattr(entries[0], "stem") and hasattr(entries[0], "representation") and hasattr(value, "stems"):
        return value
    return None


def _cap(stems: Any, limit: int = _STEMS_MAX) -> tuple[list[str], bool]:
    if stems is None:
        return [], False
    items = [str(item) for item in stems]
    return items[:limit], len(items) > limit


def summarize_index(value: Any, *, limit: int = _STEMS_MAX) -> dict[str, Any]:
    """Summarize a LoadedScans-like index through the generic protocol.

    Structural only: representation counts, conversion state and metadata
    provenance.  Classification (gold/cut/mapping...) is deliberately absent —
    inspect_experiment is the single owner of that.
    """
    representations: dict[str, int] = {}
    for entry in value.entries:
        representation = str(getattr(entry, "representation", "unknown"))
        representations[representation] = representations.get(representation, 0) + 1
    stems, stems_truncated = _cap(getattr(value, "stems", None), limit)
    needs = getattr(value, "needs_conversion", None)
    needs_list = [str(item) for item in needs] if needs is not None else None
    needs_list, needs_truncated = _cap(needs_list, limit)
    summary = str(getattr(value, "summary_line", lambda: "")() or "")[:_SUMMARY_LINE_MAX]
    detail: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "source": getattr(value, "source", None),
        "metadata_source": getattr(value, "metadata_source", None),
        "n_files": len(value.entries),
        "representations": representations,
        "needs_conversion": needs_list,
        "needs_conversion_truncated": needs_truncated or None,
        "stems": stems,
        "stems_truncated": stems_truncated or None,
        "summary": summary,
    }
    return detail


def _one_line_summary(value: Any) -> str:
    """One bounded line describing a variable for listings."""
    index = _as_index(value)
    if index is not None:
        representations = summarize_index(index)["representations"]
        parts = ", ".join(f"{key}={n}" for key, n in sorted(representations.items()))
        needs = getattr(index, "needs_conversion", None)
        text = f"{type(value).__name__} n={len(index.entries)} [{parts}]"
        if needs:
            text += f"; {len(needs)} need conversion"
        return text[:_SUMMARY_LINE_MAX]
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        name = getattr(value, "name", None)
        dims = ", ".join(f"{key}={int(item)}" for key, item in value.sizes.items())
        text = f"{type(value).__name__}"
        if name:
            text += f" {name!r}"
        if isinstance(value, xr.Dataset):
            # Dataset has no ``dtype``; report its variable count instead
            # (fit_gold returns a Dataset, so this path is routine).
            text += f" dims=[{dims}] n_vars={len(value.data_vars)}"
        else:
            text += f" dims=[{dims}] dtype={value.dtype}"
        text += f" lazy={summarize_xarray(value)['lazy']}"
        return text[:_SUMMARY_LINE_MAX]
    if isinstance(value, xr.DataTree):
        return f"xarray.DataTree groups={len(list(value.subtree))}[{_SUMMARY_LINE_MAX}]"[:_SUMMARY_LINE_MAX]
    if isinstance(value, np.ndarray):
        return f"ndarray shape={list(value.shape)} dtype={value.dtype}"[:_SUMMARY_LINE_MAX]
    return repr(value)[:_SUMMARY_LINE_MAX]


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
            index = _as_index(value)
            if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
                item.update({"dims": list(value.dims) if not isinstance(value, xr.DataTree) else None, "sizes": dict(value.sizes) if not isinstance(value, xr.DataTree) else None})
            elif index is not None:
                representations = summarize_index(index)["representations"]
                needs = getattr(index, "needs_conversion", None)
                item.update({
                    "indexed": {
                        "n": len(index.entries),
                        "representations": representations,
                        "needs_conversion": len(needs) if needs is not None else 0,
                    }
                })
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
        index = _as_index(value)
        if index is not None:
            return {"name": name, **summarize_index(index)}
        return {"name": name, "type": f"{type(value).__module__}.{type(value).__name__}", "repr": repr(value)[:_REPR_MAX]}

    def active_cell(self) -> dict[str, Any]:
        """Return the current frontend cell, outputs included.

        The tool layer normalises ``outputs`` before the model sees them; this
        backend method itself stays transport-neutral and returns the raw cell
        snapshot exactly as the frontend produced it.
        """
        if self.state.bridge and self.state.bridge.connected:
            try:
                result = self.state.bridge.request("read_active_cell", timeout=5)
                if isinstance(result, dict):
                    self.state.active_cell = {
                        key: value for key, value in result.items() if key != "outputs"
                    }
                    return result
            except Exception:
                pass
        return dict(self.state.active_cell)

    def inspect(
        self,
        target: str = "variables",
        *,
        variable_name: str | None = None,
        detail: str = "summary",
        limit: int = 10,
        cell: str | int | None = None,
        offset: int = 0,
        with_text_outputs: bool = False,
    ) -> dict[str, Any]:
        """Generic inspect_notebook protocol (target x detail, bounded).

        Targets: ``variables`` (listing rows), ``variable`` (one named
        variable), ``active_cell`` / ``cells`` / ``cell`` (cell identity,
        source and - when ``with_text_outputs=True`` - the cell's bounded
        TEXT outputs: stream stdout/stderr + text/plain only, capped at
        ~8KB/cell; images and other payloads are never returned, so the
        single output channel rule keeps constraining image payloads only).
        ``detail``: ``summary`` / ``preview``; ``limit``/``offset`` page the
        ``cells`` history.
        """
        if target == "variables":
            return self._inspect_variables(detail=detail, limit=limit)
        if target == "variable":
            if not variable_name:
                raise ValueError("inspect_notebook: target='variable' requires variable_name")
            return self._inspect_variable(variable_name, detail=detail)
        if target == "active_cell":
            return self._inspect_active_cell(detail=detail, with_text_outputs=with_text_outputs)
        if target == "cells":
            return self._inspect_cells(detail=detail, limit=limit, offset=offset,
                                       with_text_outputs=with_text_outputs)
        if target == "cell":
            if cell is None:
                raise ValueError(
                    "inspect_notebook: target='cell' requires cell (id or index)"
                )
            return self._inspect_cell(cell, detail=detail,
                                      with_text_outputs=with_text_outputs)
        raise ValueError(
            f"inspect_notebook: unknown target {target!r}; expected "
            "variables | variable | active_cell | cells | cell"
        )

    def _inspect_variables(self, *, detail: str, limit: int) -> dict[str, Any]:
        limit = max(1, min(int(limit), _LIST_VARIABLES_LIMIT))
        all_rows = self.list_variables()["variables"]
        truncated = len(all_rows) > limit
        rows = all_rows[:limit]
        if detail == "preview":
            for row in rows:
                value = self.state.namespace[row["name"]]
                structural = self._structural_detail(value)
                if structural:
                    row["structural"] = structural
        else:
            for row in rows:
                row["summary"] = _one_line_summary(self.state.namespace[row["name"]])
        return {
            "target": "variables",
            "detail": detail,
            "count": len(all_rows),
            "truncated": truncated or None,
            "variables": rows,
        }

    def _structural_detail(self, value: Any) -> dict[str, Any] | None:
        """Kind-specific bounded structural detail for preview rows."""
        index = _as_index(value)
        if index is not None:
            return {key: item for key, item in summarize_index(index).items()
                    if key not in {"type", "summary"}}
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            summary = summarize_xarray(value)
            return {
                "dims": summary["dims"],
                "sizes": summary["sizes"],
                "dtype": summary["dtype"],
                "units": summary["units"],
                "coords": sorted(summary["coords"]),
                "lazy": summary["lazy"],
            }
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "size": int(value.size)}
        if isinstance(value, xr.DataTree):
            return summarize_xarray(value)
        return None

    def _inspect_variable(self, name: str, *, detail: str) -> dict[str, Any]:
        if name not in self.state.namespace:
            raise KeyError(f"variable {name!r} does not exist")
        value = self.state.namespace[name]
        if detail == "preview":
            index = _as_index(value)
            if index is not None:
                return {"target": "variable", "detail": detail, "name": name,
                        **summarize_index(index)}
            if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree, np.ndarray)):
                return {"target": "variable", "detail": detail, "name": name,
                        **self.read_variable(name)}
            return {"target": "variable", "detail": detail, "name": name,
                    "type": f"{type(value).__module__}.{type(value).__name__}",
                    "repr": repr(value)[:_REPR_MAX]}
        return {
            "target": "variable",
            "detail": detail,
            "name": name,
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "summary": _one_line_summary(value),
        }

    def _inspect_active_cell(self, *, detail: str, with_text_outputs: bool) -> dict[str, Any]:
        """Active-cell inspection NEVER returns raw outputs or image payloads.

        Executed outputs travel exactly once, settled inside the run_cell
        reply.  When ``with_text_outputs`` is set the cell's bounded TEXT
        outputs are returned (streams + text/plain only, ~8KB cap) so the
        model can re-read what a cell printed; images are still only ever
        reported as rendered markers.
        """
        cell = self.active_cell()
        source = str(cell.get("source") or "")
        outputs = cell.get("outputs")
        base: dict[str, Any] = {
            "target": "active_cell",
            "detail": detail,
            "id": cell.get("id"),
            "index": cell.get("index"),
            "cell_type": cell.get("cell_type"),
            "execution_count": cell.get("execution_count"),
            "n_outputs": len(outputs) if isinstance(outputs, list) else None,
            "outputs_omitted": True,
        }
        if with_text_outputs:
            text, text_truncated = extract_text_outputs(outputs)
            base["text_outputs"] = text
            base["text_truncated"] = text_truncated or None
        if detail == "preview":
            base["source"] = source[:_ACTIVE_CELL_SOURCE_MAX]
        else:
            base["source_preview"] = source[:_SUMMARY_LINE_MAX]
        return base

    def _read_cells_from_frontend(self, limit: int, offset: int,
                                   with_text_outputs: bool) -> dict[str, Any] | None:
        """Read bounded cell history rows from the frontend document model.

        Returns None when no frontend bridge is available (headless kernel).
        Rows never carry image payloads; with ``with_text_outputs`` the
        frontend attaches each cell's bounded text outputs.
        """
        if not (self.state.bridge and self.state.bridge.connected):
            return None
        try:
            result = self.state.bridge.request(
                "read_cells",
                {"limit": limit, "offset": offset, "with_text_outputs": with_text_outputs},
                timeout=5,
            )
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _inspect_cells(self, *, detail: str, limit: int, offset: int,
                       with_text_outputs: bool) -> dict[str, Any]:
        """Bounded notebook-history read: trailing cell summaries (paged).

        ``limit`` caps how many trailing cells are returned (1..50),
        ``offset`` skips the newest N cells for paging older history.  With
        ``with_text_outputs`` each row carries bounded text outputs (never
        image payloads).
        """
        limit = max(1, min(int(limit), 50))
        offset = max(0, int(offset))
        rows = self._read_cells_from_frontend(limit, offset, with_text_outputs)
        if rows is None:
            return {
                "target": "cells",
                "detail": detail,
                "available": False,
                "note": "no frontend notebook attached; cell history is unavailable",
                "cells": [],
            }
        cells = [cell for cell in rows.get("cells") or [] if isinstance(cell, dict)]
        for cell in cells:
            cell.pop("outputs", None)
            text, text_truncated = self._bounded_text(cell)
            source = str(cell.get("source") or "")
            if detail == "summary":
                cell["source_preview"] = source[:_SUMMARY_LINE_MAX]
                cell.pop("source", None)
            else:
                cell["source"] = source[:_ACTIVE_CELL_SOURCE_MAX]
            cell["outputs_omitted"] = True
            if with_text_outputs:
                cell["text_outputs"] = text
                cell["text_truncated"] = text_truncated or None
            else:
                cell.pop("text_outputs", None)
                cell.pop("text_truncated", None)
        return {
            "target": "cells",
            "detail": detail,
            "available": True,
            "count": len(cells),
            "offset": offset,
            "truncated": bool(rows.get("truncated")),
            "cells": cells,
        }

    @staticmethod
    def _bounded_text(cell: dict[str, Any]) -> tuple[str, bool]:
        """取一行里的 text 输出并截断（兼容前端已截断或后端直出两种来源）。"""
        raw = cell.get("text_outputs")
        if isinstance(raw, str):
            truncated = cell.get("text_truncated") or len(raw) > _TEXT_OUTPUT_MAX
            return raw[:_TEXT_OUTPUT_MAX], bool(truncated)
        outputs = cell.get("outputs")
        if isinstance(outputs, list):
            return extract_text_outputs(outputs)
        return "", False

    def _inspect_cell(self, cell: str | int, *, detail: str,
                      with_text_outputs: bool) -> dict[str, Any]:
        """One cell by id or positional index: source + light metadata, with
        bounded text outputs when requested.  Image payloads never return.
        """
        if not (self.state.bridge and self.state.bridge.connected):
            raise KeyError(f"cell {cell!r}: no frontend notebook attached")
        result = self.state.bridge.request(
            "read_cell", {"cell": cell, "with_text_outputs": with_text_outputs}, timeout=5
        )
        if not isinstance(result, dict) or result.get("cell") is None:
            raise KeyError(f"cell {cell!r} does not exist in the notebook")
        row: dict[str, Any] = dict(result["cell"])
        row.pop("outputs", None)
        text, text_truncated = self._bounded_text(row)
        source = str(row.get("source") or "")
        row["outputs_omitted"] = True
        if detail == "summary":
            row["source_preview"] = source[:_SUMMARY_LINE_MAX]
            row.pop("source", None)
        else:
            row["source"] = source[:_ACTIVE_CELL_SOURCE_MAX]
        if with_text_outputs:
            row["text_outputs"] = text
            row["text_truncated"] = text_truncated or None
        else:
            row.pop("text_outputs", None)
            row.pop("text_truncated", None)
        return {
            "target": "cell",
            "detail": detail,
            "id": row.get("id"),
            "index": row.get("index"),
            "cell_type": row.get("cell_type"),
            "execution_count": row.get("execution_count"),
            **row,
        }

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
