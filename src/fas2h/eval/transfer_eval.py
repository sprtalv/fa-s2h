"""Transfer evaluation interfaces."""

from __future__ import annotations

from typing import Any


def run_transfer_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run transfer evaluation placeholder.

    TODO(fas2h): black-box VLM caption evaluation later.
    NOTE(fas2h): MVP does not call any external model API.
    """
    return {"status": "placeholder", "message": "transfer eval is not implemented in the MVP"}
