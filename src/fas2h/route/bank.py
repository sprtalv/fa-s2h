"""Target evidence bank helpers."""

from __future__ import annotations

from fas2h.features import BankBundle


def build_target_evidence_bank(bank_bundle: BankBundle) -> BankBundle:
    """Return the fixed target evidence bank for the MVP attack.

    NOTE(fas2h): target evidence banks are pair-specific, model-specific, and
    layer-specific. The MVP keeps them fixed during PGD to isolate the effect of
    shallow shortcut injection via `L_inj`.
    """
    if bank_bundle.target_bank is None:
        raise ValueError("BankBundle.target_bank must be populated before use.")
    return bank_bundle
