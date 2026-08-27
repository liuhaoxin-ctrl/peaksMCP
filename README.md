# peaksMCP

peaksMCP connects Claude Desktop to a live Jupyter kernel so ARPES data can be explored,
processed, converted and plotted using natural language while every executable analysis remains
visible in the notebook.

```text
Claude Desktop <-> STDIO proxy <-> HTTP MCP <-> Jupyter kernel <-> JupyterLab Comm
```

## Quick start

```bash
python -m pip install -e '.[dev]'
peaksMCP install-extension
peaksMCP install-kernel
peaksMCP launch
```

Install `claude_plugin/` in Claude Desktop. Its MCP configuration starts
`peaksMCP stdio-proxy` and connects to the server hosted by the active notebook kernel.

Useful commands:

```bash
peaksMCP status
peaksMCP doctor
peaksMCP mcp-ping
peaksMCP metadata translate path/to/datasheet.csv
peaksMCP convert path/to/pxt-folder --metadata experiment_metadata.json
```

See `docs/ARCHITECTURE.md` for the compact architecture map.

