# Agent instructions

This project enables Claude Desktop to process ARPES data in natural language and produce
publication-quality ARPES figures.

## Boundaries

- Work only inside this repository unless the user explicitly requests otherwise.
- Treat the installed `peaks` package as an external dependency; do not edit site-packages.
- Keep the MCP server inside the Jupyter kernel and the supervisor outside it.
- Preserve raw PXT inputs. Conversion writes new NetCDF files atomically.
- Do not add Qt/GUI plotting paths; use Jupyter inline Matplotlib output.
- Public Python APIs use NumPy-style docstrings.

## Development

Use the `peaks` conda environment:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pip install -e '.[dev]'
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest
```

Run fast by default (e2e excluded) and only bring up live kernels/browsers for
the explicit e2e acceptance:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest            # unit + integration
/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python -m pytest -m e2e     # live-kernel + browser
```

### CPU budget for tests and agent runs

- **Never let test/agent processes hold CPU above 60% sustained.** Keep compute
  light, prefer single-threaded workloads, and cap native thread pools (BLAS/OpenMP)
  when tests exercise numeric code.
- Check CPU before/after a heavy run (`ps aux -r | head`); if anything lingers
  above 60%, stop and investigate rather than piling on more work.

When adding or changing an MCP tool, update its implementation, metadata in
`peaksMCP/config/metadata_baseline.yaml`, the exact tool-list tests, and the Claude test helper.
When changing API discovery, run the full name-coverage and natural-language ranking tests.

