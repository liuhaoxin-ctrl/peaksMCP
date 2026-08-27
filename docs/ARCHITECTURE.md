# Architecture

## Purpose

Use Claude Desktop to process ARPES data in natural language and produce publication-quality
ARPES spectra.

## Flow

```text
Claude Desktop -> STDIO proxy -> kernel HTTP MCP server -> notebook backend
                                                     <-> JupyterLab Comm frontend
Supervisor      -> JupyterLab process, health, logs, dashboard and restart recovery
```

## Components

- `discovery`: live Peaks/xarray API index and source-level signatures.
- `server/jupyter_peaks`: MCP registrars, notebook state, Comm bridge and security.
- `pxt_utils`: datasheet translation, PXT loading and atomic NetCDF conversion.
- `plotting` and `batch`: publication layout and bounded parallel execution.
- `app`: profile-driven supervisor, status API and static dashboard.
- `claude_plugin`: Claude Desktop MCP declaration and analysis skill.

## Development entry points

- `peaksMCP launch`
- `%load_ext peaksMCP.server.jupyter_peaks.jupyter_mcp_extension`
- `pytest tests`

