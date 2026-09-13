# Architecture

## Purpose

Use Claude Desktop to process ARPES data in natural language and produce publication-quality
ARPES spectra, and measure whether an autonomous agent can do the same unprompted (see
`benchmark/PROTOCOL.md`).

## Flow

```text
Claude Desktop -> STDIO proxy -> kernel HTTP MCP server -> notebook backend
                                                     <-> JupyterLab Comm frontend
Supervisor      -> JupyterLab process, health, logs, dashboard and restart recovery
```

## Components

- `discovery`: the model-visible Peaks/xarray API index and source-level
  signatures, curated by the unified `config/api_catalog.yaml`.
- `server/jupyter_peaks`: MCP registrars, notebook state, Comm bridge and security.
- `pxt_utils`: datasheet translation, PXT loading and atomic NetCDF conversion.
- `plotting` and `batch`: publication layout and bounded parallel execution.
- `app`: profile-driven supervisor, status API and static dashboard.
- `claude_plugin`: Claude Desktop MCP declaration and analysis skill.
- `benchmark`: the autonomous-agent evaluation platform - protocol, cases, rubric, grader and
  campaign runner. It observes the product from outside; the product never imports it.

## Public Analysis Surface

The workflow adds only two public Peaks entry points:

- `peaks.pxt2nc(...)` creates or reuses the atomic, owned PXT-to-NetCDF cache.
  Raw bytes and the complete normalized metadata document participate in the
  fingerprint; files without its owner/schema marker are never overwritten.
- `peaks.load_experiment(...)` discovers metadata, indexes the experiment,
  classifies gold/cut/mapping records and lazily exposes individual scans through
  one `ExperimentIndex`.

All other fitting, metadata correction, momentum conversion and plotting calls
are existing Peaks APIs. `config/api_catalog.yaml` is the single model-facing
catalog for both entry points and those native APIs; callables absent from it are
hidden. The initial public facade count is zero. `peaksMCP.overrides` remains
internal compatibility code and is not a second API surface.

Scientific results live in named kernel variables and are rendered as inline
Notebook output. Figures are not written to disk. The cache and metadata sidecar
managed by `peaks.pxt2nc` are the only automatic persistence in the analysis
workflow; conversion never mutates its raw input.

The dashboard is part of the supervisor lifecycle. It publishes the runfile only after
Uvicorn has bound successfully, authenticates operator API access with a separate
operator-console token, and rejects non-loopback binding unless a profile explicitly sets
`dashboard.allow_remote: true`.

The supervisor accepts an explicit Jupyter root directory and records it in the runfile. Managed
benchmark trials use the trial `workspace/` as that root and address the notebook as
`work.ipynb`; host reuse compares both fields so two trials with the same notebook filename never
share a kernel workspace accidentally.

MCP restarts are verified with a per-MCP instance ID, and kernel restarts with a
per-kernel instance ID. With a live frontend
(``require_comm``) recovery reports READY only after that ID changes and the
extension, Comm, MCP initialize, the exact 5-tool inventory (search / get /
inspect_notebook / run_cell / save_with_consent) and the private loopback
``/healthz`` payload have all recovered; when the frontend is offline the
restart degrades to a plain REST restart and reports READY without the Comm
stage (kernel + MCP still fully recovered).

`run_cell` is the single model-facing append operation. With `cell_type="code"`
it API-checks and executes a new code cell; with `cell_type="markdown"` it
appends a non-executed record or final summary. Both forms append at the end,
and neither can edit, delete, or reorder notebook history.

## Development entry points

- `peaksMCP dash`
- `%load_ext peaksMCP.server.jupyter_peaks.jupyter_mcp_extension`
- `python tools/test.py --list` (deterministic test layers; no pytest layer drives a model)
- `python tools/trial.py run --provider deepseek --model deepseek-v4-flash` (live model trial: the
  Pi TUI against a fresh managed kernel, sessions under `.pi_sessions/`)
- `python benchmark/run_campaign.py ... --runner pi-tui` (the same Pi TUI inside the paired
  campaign design, for comparing configurations)
