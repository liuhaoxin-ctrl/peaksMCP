"""peaksMCP public package."""

__version__ = "0.1.0"

from .plotting.layout import plot_batch

# Note: importing peaksMCP deliberately does NOT import pxt_utils.loader or
# peaks. PXT reading is reachable only through the conversion entry points
# (peaksMCP convert / dashboard), and the L112 NetCDF loader is registered
# explicitly by the in-kernel extension when the MCP server starts.

__all__ = ["__version__", "plot_batch"]

