"""Frontend-mediated user consent for unsafe notebook mutations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ConsentManager:
    """Request explicit approval through Comm, or a test callback."""

    def __init__(self, bridge: Any | None = None, callback: Callable[[str, dict[str, Any]], bool] | None = None) -> None:
        self.bridge = bridge
        self.callback = callback

    def request(self, operation: str, details: dict[str, Any], timeout: float = 60) -> bool:
        """Return true only after an affirmative frontend response."""
        if self.callback is not None:
            return bool(self.callback(operation, details))
        if self.bridge is None or not self.bridge.connected:
            return False
        response = self.bridge.request("request_consent", {"requested_operation": operation, "details": details}, timeout=timeout)
        return bool(response.get("approved"))
