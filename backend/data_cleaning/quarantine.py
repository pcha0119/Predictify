"""
QuarantineLog — accumulates rejected rows during the cleaning pipeline.

Rather than silently dropping bad rows, every rejected row is tagged with a
reason and its source sheet, then written to a single quarantine CSV for
inspection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class QuarantineLog:
    """Accumulate and persist rows rejected during cleaning."""

    def __init__(self) -> None:
        self._records: list[pd.DataFrame] = []

    # ── public methods ────────────────────────────────────────────────────────

    def add(
        self,
        rows: pd.DataFrame,
        reason: str,
        source_sheet: str,
    ) -> None:
        """
        Tag *rows* with quarantine metadata and store them.

        Parameters
        ----------
        rows         : DataFrame slice of the rejected rows.
        reason       : Human-readable rejection reason, e.g. 'null_date'.
        source_sheet : Logical sheet name, e.g. 'sale_lines'.
        """
        if rows.empty:
            return
        tagged = rows.copy()
        tagged["_quarantine_reason"] = reason
        tagged["_quarantine_source"] = source_sheet
        self._records.append(tagged)
        logger.warning(
            "Quarantined %d rows from '%s': %s", len(rows), source_sheet, reason
        )

    def flush(self, output_path: Union[str, Path, None] = None) -> pd.DataFrame:
        """
        Concatenate all quarantined records, write to CSV, and return the
        combined DataFrame.  Safe to call even if no rows were quarantined.
        """
        if not self._records:
            logger.info("No rows were quarantined.")
            return pd.DataFrame()

        combined = pd.concat(self._records, ignore_index=True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)
            logger.info("Quarantine log written to: %s  (%d rows)", output_path, len(combined))

        return combined

    def summary(self) -> dict[str, int]:
        """Return count of quarantined rows grouped by reason."""
        if not self._records:
            return {}
        combined = pd.concat(self._records, ignore_index=True)
        return combined["_quarantine_reason"].value_counts().to_dict()

    @property
    def total(self) -> int:
        """Total number of quarantined rows accumulated so far."""
        return sum(len(r) for r in self._records)
