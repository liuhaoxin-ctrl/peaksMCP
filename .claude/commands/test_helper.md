---
description: Interactively verify every peaksMCP transport and tool category
argument-hint: [connection|api|notebook|images|pxt|restart|all]
---

# peaksMCP test helper

Test the requested category against a running peaksMCP notebook. Start with server and kernel
status, list the exposed tools, then call representative tools with small read-only inputs.
Before unsafe calls, explain the expected consent dialog. Record the exact tool name, arguments,
result, duration and any error. For `all`, cover connection, API discovery, xarray variable
summary, inline image output, PXT metadata/convert dry inputs, MCP restart, kernel restart and
STDIO reconnection. Finish with passed/failed/skipped totals and reproducible failure steps.

