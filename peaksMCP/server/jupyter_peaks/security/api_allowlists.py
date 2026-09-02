"""Pure-data allowlists for the API-check classification.

These are the static surfaces an ARPES analysis notebook may touch without a
Peaks API: xarray/pandas/numpy/matplotlib methods, Python builtins, and generic
library modules.  They are data only — the classification logic lives in
:mod:`peaksMCP.server.jupyter_peaks.security.api_provenance` and
:mod:`peaksMCP.server.jupyter_peaks.backend.notebook_unsafe`.
"""

from __future__ import annotations

# Python builtins that appear as bare calls (``print``, ``len``, ``range``,
# ``str``, ...).  They are never Peaks APIs and never block execution.
# ``exec``/``eval``/``compile``/``getattr`` are deliberately absent: the
# security scanner hard-blocks them, and keeping them unverifiable adds a
# second rejection layer.
BUILTIN_NAMES = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "complex", "delattr", "dict",
    "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hasattr", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max", "memoryview",
    "min", "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
    "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
})

# Ordinary methods on data receivers (xarray / pandas / numpy / matplotlib /
# stdlib).  These are NOT Peaks APIs and never block execution; anything else
# used as ``obj.<name>`` that is neither a Peaks API nor in this set is an
# unverifiable API reference and hard-blocks execution.
GENERIC_METHODS = frozenset({
    "sel", "isel", "mean", "median", "max", "min", "sum", "std", "var",
    "count", "cumsum", "diff", "argmax", "argmin", "clip", "abs", "round",
    "plot", "plot_fit", "plot_residuals", "assign_coords", "drop_vars",
    "rename", "stack", "unstack", "transpose", "squeeze", "expand_dims",
    "swap_dims", "values", "data", "item", "to_dataset", "to_netcdf",
    "compute", "load", "where", "fillna", "interp", "groupby", "resample",
    "shift", "rolling", "coarsen", "quantile", "copy", "dims", "sizes",
    "coords", "attrs", "metadata", "history", "pint", "real", "imag",
    "astype", "reshape", "flatten", "tolist", "split",
    "join", "replace", "strip", "lower", "upper", "format", "append",
    "extend", "pop", "keys", "items", "get", "setdefault", "update",
    "add", "remove", "discard", "union", "difference", "intersection",
    "dumps", "loads", "dump", "reader", "writer", "DictReader",
    "DictWriter", "read_csv", "read_excel", "read_json", "to_csv",
    "to_excel", "to_json", "to_numpy", "DataFrame", "Series",
    "value_counts", "dropna", "describe", "head", "tail", "iloc", "loc",
    "set_index", "reset_index", "sort_values", "sort_index", "merge",
    "concat", "pivot", "melt", "iterrows", "apply", "map", "unique",
    "isin", "sample", "drop", "insert", "sort", "reverse", "index",
    "read", "write", "close", "open", "readline", "readlines",
    "writelines", "seek", "tell", "flush", "truncate", "read_text",
    "write_text", "read_bytes", "write_bytes", "exists", "mkdir",
    "unlink", "glob", "rglob", "iterdir",
    "is_file", "is_dir", "resolve", "joinpath", "with_suffix", "stem",
    "suffix", "parent", "name", "listdir", "makedirs", "walk", "getcwd",
    "chdir", "getenv", "path", "splitext", "dirname", "basename",
    "startswith", "endswith", "find", "zfill", "encode",
    "decode", "isdigit", "isalpha", "isalnum", "isnumeric", "isspace",
    "splitlines", "rstrip", "lstrip", "capitalize", "title", "partition",
    "rpartition", "center", "expandtabs", "swapcase", "casefold",
    "popitem", "fromkeys", "clear", "symmetric_difference", "issubset",
    "issuperset", "any", "all", "next", "iter", "sorted",
    "reversed", "enumerate", "zip", "filter", "hash", "id", "repr",
    "array", "arange", "linspace", "zeros", "ones", "full", "eye",
    "zeros_like", "ones_like", "empty", "empty_like", "full_like",
    "meshgrid", "concatenate", "vstack", "hstack", "ravel", "dot",
    "matmul", "prod", "sqrt", "exp", "log", "log10", "log2", "sin",
    "cos", "tan", "asin", "acos", "atan", "atan2", "gradient",
    "poly1d", "polyfit", "polyval", "percentile", "nanpercentile",
    "nanmean", "nanmax", "nanmin", "histogram", "bincount", "loadtxt",
    "genfromtxt", "savetxt", "save", "savez", "fromfile", "fromstring",
    "asarray", "asanyarray", "seed", "random", "normal", "uniform",
    "randint", "rand", "randn", "choice", "flip",
    "roll", "argsort", "take", "repeat", "tile",
    "pad", "corrcoef", "cov", "apply_along_axis", "vectorize",
    "isnan", "isinf", "isfinite", "nonzero", "mod", "floor",
    "ceil", "around", "sign", "power", "square", "maximum", "minimum",
    "amax", "amin", "ptp", "trapz", "convolve", "correlate", "fft",
    "ifft", "rfft", "irfft", "conjugate", "angle", "hypot", "diag",
    "tril", "triu", "cumprod", "figure", "subplots", "scatter", "imshow",
    "pcolormesh", "bar", "hist", "errorbar", "fill_between", "colorbar",
    "xlabel", "ylabel", "legend", "tight_layout", "show", "savefig",
    "subplots_adjust", "set_title", "set_xlabel", "set_ylabel",
    "axhline", "axvline", "set_ylim", "set_xlim", "xticks", "yticks",
    "grid", "axis", "contour", "contourf", "text", "annotate", "rcParams",
    "set_visible", "get_figure", "convert_to", "twinx",
    "twiny", "loglog", "semilogx", "semilogy", "suptitle", "clf", "cla",
    "gcf", "gca", "xlim", "ylim", "set_xscale", "set_yscale", "cm",
    "search", "match", "findall", "finditer", "sub", "subn", "compile",
    "fullmatch", "escape", "pi", "e", "tau", "inf", "nan", "isclose",
    "fabs", "fmod", "gcd", "factorial", "degrees", "radians", "strftime",
    "strptime", "isoformat", "now", "today", "utcnow", "timestamp",
    "fromtimestamp", "combine", "timedelta", "date", "time", "datetime",
})

# Modules whose members are treated as generic (never block).
GENERIC_MODULES = frozenset({
    "np", "numpy", "scipy", "plt", "matplotlib", "xr", "xarray",
    "pd", "pandas", "os", "sys", "json", "math", "re", "time",
    "glob", "shutil", "pathlib", "Path", "warnings", "pickle",
    "copy", "itertools", "functools", "collections", "datetime",
    "csv", "io", "tempfile", "uuid", "hashlib", "string", "traceback",
    "threading", "dataclasses", "enum", "abc", "typing", "contextlib",
    "calendar", "random", "statistics", "numbers", "decimal", "fractions",
})