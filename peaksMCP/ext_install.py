"""Install the bundled prebuilt JupyterLab extension idempotently."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from jupyter_core.paths import jupyter_path


def extension_source() -> Path:
    """Return the wheel-bundled federated extension directory."""
    return Path(__file__).with_name("extensions") / "jupyterlab" / "labextension"


def install_extension(*, develop: bool = False, destination: str | os.PathLike[str] | None = None) -> Path:
    """Install or link the prebuilt extension into the current environment."""
    source = extension_source()
    manifest = source / "package.json"
    if not manifest.is_file() or not (source / "static").is_dir():
        raise FileNotFoundError("prebuilt JupyterLab extension is missing; run `jlpm install && jlpm build:prod` in peaksMCP/extensions/jupyterlab")
    package = json.loads(manifest.read_text(encoding="utf-8"))
    if package.get("name") != "peaksmcp-jupyterlab":
        raise ValueError("unexpected JupyterLab extension package name")
    target_root = Path(destination) if destination else Path(sys.prefix) / "share" / "jupyter" / "labextensions"
    target = target_root / "peaksmcp-jupyterlab"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return target
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if develop:
        try:
            target.symlink_to(source, target_is_directory=True)
            return target
        except OSError:
            pass
    shutil.copytree(source, target)
    return target


def locate_installed_extension() -> list[Path]:
    """Return all visible installations of the extension."""
    return [Path(root) / "labextensions" / "peaksmcp-jupyterlab" for root in jupyter_path() if (Path(root) / "labextensions" / "peaksmcp-jupyterlab").exists()]

