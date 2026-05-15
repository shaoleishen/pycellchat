"""Convert CellChatDB .rda files to parquet format.

Requires: pip install rdata pandas pyarrow

Usage:
    python scripts/convert_databases.py
"""

import sys
from pathlib import Path

import rdata
import pandas as pd

CELLCHAT_DIR = Path("/home/bioshen/Code/CellChat-main/data")
OUTPUT_DIR = Path(__file__).parent.parent / "python" / "pycellchat" / "database" / "data"


def convert_species(species: str) -> None:
    """Convert CellChatDB for a single species."""
    rda_path = CELLCHAT_DIR / f"CellChatDB.{species}.rda"
    if not rda_path.exists():
        print(f"  Skipping {species}: {rda_path} not found")
        return

    print(f"  Reading {rda_path}...")
    parsed = rdata.parser.parse_file(str(rda_path))
    converted = rdata.conversion.convert(parsed)

    # The .rda contains a single top-level key like "CellChatDB.human"
    db_key = list(converted.keys())[0]
    db = converted[db_key]

    out_dir = OUTPUT_DIR / species
    out_dir.mkdir(parents=True, exist_ok=True)

    for component_name in ["interaction", "complex", "cofactor", "geneInfo"]:
        if component_name not in db:
            print(f"  Warning: '{component_name}' not found in {species} DB")
            continue

        df = db[component_name]
        if isinstance(df, pd.DataFrame):
            # Clean column names: strip whitespace
            df.columns = df.columns.str.strip()
            # Preserve R rownames as 'name' column for complex/cofactor tables
            if component_name in ("complex", "cofactor") and not pd.api.types.is_integer_dtype(df.index):
                df.index.name = "name"
                df = df.reset_index()
            out_path = out_dir / f"{component_name}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Saved {component_name}: {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
        else:
            print(f"  Warning: {component_name} is {type(df).__name__}, not DataFrame")


def convert_ppi() -> None:
    """Convert PPI .rda files."""
    for species in ["human", "mouse"]:
        rda_path = CELLCHAT_DIR / f"PPI.{species}.rda"
        if not rda_path.exists():
            continue

        print(f"  Reading {rda_path}...")
        parsed = rdata.parser.parse_file(str(rda_path))
        converted = rdata.conversion.convert(parsed)

        ppi_key = list(converted.keys())[0]
        ppi = converted[ppi_key]

        out_dir = OUTPUT_DIR / species
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(ppi, pd.DataFrame):
            out_path = out_dir / "ppi.parquet"
            ppi.to_parquet(out_path)
            print(f"  Saved PPI: {ppi.shape[0]} x {ppi.shape[1]} -> {out_path}")
        elif isinstance(ppi, dict):
            # May be a matrix stored as dict
            for k, v in ppi.items():
                if isinstance(v, pd.DataFrame):
                    out_path = out_dir / f"ppi_{k}.parquet"
                    v.to_parquet(out_path)
                    print(f"  Saved PPI.{k}: {v.shape} -> {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Converting CellChatDB...")
    for species in ["human", "mouse", "zebrafish"]:
        print(f"\n[{species}]")
        convert_species(species)

    print("\n\nConverting PPI...")
    convert_ppi()

    print("\nDone! Database files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
