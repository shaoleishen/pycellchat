"""Load CellChatDB from converted parquet files."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def load_cellchatdb(species: str = "human") -> dict:
    """Load CellChatDB for the given species.

    Parameters
    ----------
    species
        One of ``"human"``, ``"mouse"``, ``"zebrafish"``.

    Returns
    -------
    Dict with keys ``interaction``, ``complex``, ``cofactor``, ``geneInfo``,
    each containing a pandas DataFrame.
    """
    species = species.lower()
    db_dir = _DATA_DIR / species
    if not db_dir.exists():
        raise FileNotFoundError(
            f"No CellChatDB found for species '{species}' at {db_dir}. "
            f"Run scripts/convert_databases.py first."
        )

    result = {}
    for name in ["interaction", "complex", "cofactor", "geneInfo"]:
        path = db_dir / f"{name}.parquet"
        if path.exists():
            result[name] = pd.read_parquet(path)
        else:
            raise FileNotFoundError(f"Missing {name}.parquet in {db_dir}")

    return result
