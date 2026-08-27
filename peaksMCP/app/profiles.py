"""Validated YAML runtime profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class JupyterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(8888, ge=1, le=65535)
    kernel_name: str = "peaksmcp"
    disabled_extensions: list[str] = Field(default_factory=lambda: ["jupyterlab-peaks-agent"])


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(8123, ge=1, le=65535)
    mode: Literal["safe", "unsafe", "dangerous"] = "safe"
    autostart: bool = True
    # The in-kernel MCP listener has no authentication.  Binding to anything
    # other than loopback exposes notebook contents and (in dangerous mode)
    # code execution to the network; this requires an explicit opt-in and the
    # operator is responsible for adding auth/TLS in front of the listener.
    allow_remote: bool = False


class DashboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    allow_remote: bool = False


class Profile(BaseModel):
    """One Jupyter server, one managed kernel and one MCP listener."""

    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    jupyter: JupyterConfig = JupyterConfig()
    mcp: MCPConfig = MCPConfig()
    dashboard: DashboardConfig = DashboardConfig()


def profile_directory() -> Path:
    """Return the user profile directory, respecting ``PEAKSMCP_HOME``."""
    root = Path(os.environ.get("PEAKSMCP_HOME", Path.home() / ".peaksMCP"))
    return root / "profiles"


def default_profile_path() -> Path:
    """Return the packaged default profile path."""
    return Path(__file__).with_name("defaults") / "default.yaml"


def profile_path(name: str = "default") -> Path:
    """Resolve a user profile, falling back to the packaged default."""
    candidate = profile_directory() / f"{name}.yaml"
    if candidate.is_file():
        return candidate
    if name == "default":
        return default_profile_path()
    raise FileNotFoundError(f"profile {name!r} does not exist")


def load_profile(name: str = "default") -> Profile:
    """Load and validate one runtime profile.

    Parameters
    ----------
    name : str, default "default"
        Profile filename stem in the peaksMCP profile directory.

    Returns
    -------
    Profile
        Strictly validated Jupyter, MCP and Dashboard settings.

    Examples
    --------
    >>> profile = load_profile("default")
    >>> profile.mcp.port
    8123
    """
    return Profile.model_validate(yaml.safe_load(profile_path(name).read_text(encoding="utf-8")))


def list_profiles() -> list[str]:
    """List available profile names, including the packaged default."""
    names = {"default"}
    directory = profile_directory()
    if directory.is_dir():
        names.update(path.stem for path in directory.glob("*.yaml"))
    return sorted(names)
