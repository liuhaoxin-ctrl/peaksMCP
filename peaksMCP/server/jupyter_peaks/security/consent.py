"""Frontend-mediated user consent for unsafe notebook mutations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ConsentManager:
    """Request explicit approval through Comm, or a test callback."""

    def __init__(
        self,
        bridge: Any | None = None,
        callback: Callable[[str, dict[str, Any]], bool | None] | None = None,
    ) -> None:
        self.bridge = bridge
        self.callback = callback

    def request(self, operation: str, details: dict[str, Any], timeout: float = 60) -> bool | None:
        """Ask the human; ``None`` means nobody could be asked.

        The distinction matters for receipts: ``False`` is a human refusal
        (``denied``), while ``None`` is "no approver was reachable" (the
        frontend is gone), which must never be reported as a rejection.
        """
        if self.callback is not None:
            answer = self.callback(operation, details)
            return None if answer is None else bool(answer)
        if self.bridge is None or not self.bridge.connected:
            return None
        response = self.bridge.request(
            "request_consent",
            {"requested_operation": operation, "details": details},
            timeout=timeout,
        )
        return bool(response.get("approved"))
