# peaksMCP

peaksMCP connects Claude Desktop to a live Jupyter kernel so ARPES data can be explored,
processed, converted and plotted using natural language while every executable analysis remains
visible in the notebook.

```text
Claude Desktop <-> STDIO proxy <-> HTTP MCP <-> Jupyter kernel <-> JupyterLab Comm
```

## Quick start

### 1. Create a Python environment

```bash
conda create -n peaks python=3.12 -y
conda activate peaks
```

If `conda activate` reports `Run 'conda init' before 'conda activate'`, your terminal
has not loaded the conda shell hook yet — run `source ~/.zshrc` (or open a new terminal
window) once, then verify with `type conda` (it should print `conda is a shell function`).

### 2. Get the source and install the scientific dependency

```bash
git clone https://github.com/phrgab/peaksMCP && cd peaksMCP
# ARPES analysis library (peaksMCP data tools call it at runtime)
pip install git+https://github.com/phrgab/peaks
```

### 3. Install peaksMCP

```bash
pip install -e '.[dev]'
```

### 4. Build and install the JupyterLab extension

```bash
cd peaksMCP/extensions/jupyterlab && jlpm install && jlpm build:prod && cd ../..
peaksMCP install-extension
```

### 5. Install the managed kernel

```bash
peaksMCP install-kernel
```

### 6. Configure Claude Desktop (STDIO)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` and restart
Claude Desktop:

```json
{
  "mcpServers": {
    "peaksMCP": {
      "type": "stdio",
      "command": "/abs/path/to/peaks/env/bin/peaksMCP",
      "args": ["stdio-proxy"]
    }
  }
}
```

Use the **STDIO** transport only. Do not add peaksMCP as a *remote* MCP server: Claude
requires remote server URLs to start with `https://`, while the in-kernel MCP listener is
plain HTTP on `127.0.0.1`.

### 7. Start and verify

```bash
peaksMCP dash            # single entry: start the dashboard host if needed and open the operator console (alias: open)
peaksMCP status          # expect host RUNNING + kernel_id once Jupyter is up
peaksMCP mcp-ping        # expect ok: true, 9 tools (7 read-only/guidance + 2 append-only mutation)
```

The dashboard host must be running before Claude Desktop uses the tools (the STDIO
proxy forwards to the kernel-hosted MCP endpoint).

### Dashboard host (co-hosted operator console)

`peaksMCP dash` starts the host (a detached background process) if it is not
already running and opens the console. The dashboard is served within a second
and shows JupyterLab starting up; it lets you **monitor** JupyterLab / kernel /
in-kernel MCP / Comm and **control** the stack:

```bash
peaksMCP dash            # ensure host + open http://127.0.0.1:8765 (alias: open)
```

An explicitly requested notebook or profile is authoritative: if the active
singleton host runs a different workspace, `dash` replaces it before returning
(no silent reuse); an already matching host is reused. The managed notebook is
then opened from the console's **Open Notebook** button, which only enables once
JupyterLab is actually reachable.

`peaksMCP open` supplies a tokenised login URL and stores the operator-console
credential in an HttpOnly, SameSite cookie. Opening port 8765 directly is intentionally
rejected. Control APIs require the same credential and reject cross-origin requests.

In the console you can start / stop **Jupyter** (service + kernel) and **MCP**
(in-kernel) as groups, open the managed Notebook, and convert data. Stopping the
dashboard host (`peaksMCP stop`) also gracefully tears down the Jupyter/MCP tree
it manages. PXT conversion and datasheet translation are pure file operations and
always available.

### CLI reference

```bash
# Lifecycle (the host manages JupyterLab + kernel + in-kernel MCP internally)
peaksMCP dash [notebook] [--profile NAME] [--timeout SECONDS]  # start the host if needed + open the console
peaksMCP status                                         # host / jupyter state
peaksMCP stop                                           # stop the dashboard host (and the tree it manages)
peaksMCP restart                                   # stop the host and start a fresh one (like dash, no browser)
peaksMCP restart {kernel|mcp|'kernel&mcp'}          # kernel-side only (kernel&mcp = kernel + MCP)
peaksMCP logs [-n LINES] [-f]                           # show / follow host logs
# Verification & diagnostics
peaksMCP mcp-ping [--profile NAME]                      # verify the MCP endpoint + tool count
peaksMCP version                                        # package version

# Open in browser
peaksMCP dash [notebook] [--profile NAME]    # operator-console dashboard (http://127.0.0.1:8765);
                                             # open the managed notebook from the console's
                                             # "Open Notebook" button (enabled once Jupyter is up)

# Install
peaksMCP install-extension [--develop]                  # install the JupyterLab extension
peaksMCP install-kernel [--profile NAME]                # install the managed kernelspec
peaksMCP uninstall-kernel [--profile NAME]
peaksMCP stdio-proxy [--profile NAME]                   # Claude Desktop STDIO proxy

# Profiles
peaksMCP profiles list                                  # list profiles
peaksMCP profiles show [NAME]
peaksMCP profiles path [NAME]                           # print the profile file path
```

`peaksMCP dash` runs the dashboard host as a detached background process, so a
terminal `Ctrl+C` does **not** stop it — use `peaksMCP stop`.

> **Data operations are deliberately not CLI commands.** PXT conversion,
> datasheet translation and loading a scan run as **notebook cells** through the
> MCP tools, composing the curated facades of `peaksMCP.overrides`
> (`load_data` / `convert_experiment` / `translate_datasheet` /
> `fit_gold_reference` / `preprocess_cut` / `preprocess_mapping` /
> `preprocess_batch` / `save_result`), so the code scanner, the API check and the
> consent gate always apply. The CLI and the operator console only control
> processes (Jupyter, kernel, MCP) and snapshots.

See `docs/ARCHITECTURE.md` for the compact architecture map.
