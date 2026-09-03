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

The dashboard is part of the supervisor lifecycle. It publishes the runfile only after
Uvicorn has bound successfully, authenticates operator API access with a separate
operator-console token, and rejects non-loopback binding unless a profile explicitly sets
`dashboard.allow_remote: true`.

MCP restarts are verified with a per-MCP instance ID, and kernel restarts with a
per-kernel instance ID. With a live frontend
(``require_comm``) recovery reports READY only after that ID changes and the extension,
Comm, MCP initialize, the exact 15-tool inventory and status tool have all recovered; when the
frontend is offline the restart degrades to a plain REST restart and reports READY
without the Comm stage (kernel + MCP still fully recovered).

## Development entry points

- `peaksMCP launch`
- `%load_ext peaksMCP.server.jupyter_peaks.jupyter_mcp_extension`
- `pytest tests`
