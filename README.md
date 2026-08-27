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
peaksMCP launch          # single entry: JupyterLab + kernel + in-kernel MCP (127.0.0.1:8123/mcp) + operator dashboard (127.0.0.1:8765)
peaksMCP status          # expect RUNNING + kernel_id
peaksMCP mcp-ping        # expect ok: true, 17 tools (12 read-only + 5 consent-gated)
peaksMCP dash            # open the operator-console dashboard in the browser (alias: open)
```

The supervisor must be running before Claude Desktop uses the tools (the STDIO proxy
forwards to the kernel-hosted MCP endpoint).

### Operator dashboard (co-hosted)

`peaksMCP launch` also starts the operator-console dashboard in the same process, so
there is exactly one startup command. The dashboard lets you **monitor** JupyterLab /
kernel / in-kernel MCP / Comm and **control** the stack:

```bash
peaksMCP dash            # open http://127.0.0.1:8765 in the browser (alias: open)
```

`peaksMCP open` supplies a tokenised login URL and stores the operator-console
credential in an HttpOnly, SameSite cookie. Opening port 8765 directly is intentionally
rejected. Control APIs require the same credential and reject cross-origin requests.

In the console: **Start MCP** (when the in-kernel MCP is down), **Restart MCP** (kernel
variables preserved), **Restart Kernel**, **Restart All** and **Open managed Notebook**.
Stopping the whole stack is done from the CLI with . PXT conversion and
datasheet translation are pure file operations and always available.

### CLI reference

```bash
# Lifecycle
peaksMCP launch [--profile NAME] [--timeout SECONDS]   # single entry: supervisor + operator dashboard
peaksMCP status                                         # supervisor status
peaksMCP stop                                           # stop the supervisor (and the dashboard)
peaksMCP restart                                   # restart the whole stack (like launch)
peaksMCP restart {kernel|mcp|'kernel&mcp'}          # kernel-side only (kernel&mcp = kernel + MCP)
peaksMCP logs [-n LINES] [-f]                           # show / follow supervisor logs
```

`peaksMCP launch` runs the supervisor as a detached background process, so a terminal
`Ctrl+C` does **not** stop it — use `peaksMCP stop`.

# Verification & diagnostics
peaksMCP doctor [--profile NAME]                        # deps / kernel / extension / ports
peaksMCP mcp-ping [--profile NAME]                      # verify the MCP endpoint + tool count
peaksMCP version                                        # package version

# Open in browser
peaksMCP dash                                           # operator-console dashboard  (http://127.0.0.1:8765)
peaksMCP dash --jupyter                                 # JupyterLab (tokenised URL)
# http://127.0.0.1:8765/?token=t99boB3au1qRb1xZxxt4IWdwhZKAeJfB remains as a legacy alias.

# Install
peaksMCP install-extension [--develop]                  # install the JupyterLab extension
peaksMCP install-kernel [--profile NAME]                # install the managed kernelspec
peaksMCP uninstall-kernel [--profile NAME]
peaksMCP stdio-proxy [--profile NAME]                   # Claude Desktop STDIO proxy

# Profiles
peaksMCP profiles list                                  # list profiles
peaksMCP profiles show [NAME]
peaksMCP profiles path [NAME]                           # print the profile file path

# Data conversion
peaksMCP metadata translate path/to/datasheet.csv [--output metadata.json]
peaksMCP convert <pxt-file-or-folder> [--metadata metadata.json] [--out OUT]
          [--filter SUBSTRING] [--cpu-limit PERCENT] [--force]
```

Examples:

```bash
peaksMCP convert raw/ --filter BP_ --cpu-limit 60        # auto-translates raw/datasheet.csv
peaksMCP convert raw/ --metadata experiment_metadata.json --filter BP_ --cpu-limit 60
peaksMCP convert BP_0003.pxt --out converted/          # single file into a directory
```

When no `--metadata` is given, peaksMCP looks for a `datasheet.csv` in the same
folder as the data (or its parent) and auto-translates it into
`<output>/experiment_metadata.json` before converting — so a datasheet kept next
to the raw data is used automatically. The dashboard panel is the same flow
("CSV & PXT Conversion").

See `docs/ARCHITECTURE.md` for the compact architecture map.
