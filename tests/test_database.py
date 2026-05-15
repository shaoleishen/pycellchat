"""Tests for database loading."""

import pandas as pd
import pytest
from pycellchat.database import load_cellchatdb


class TestLoadCellChatDB:
    def test_human(self):
        db = load_cellchatdb("human")
        assert "interaction" in db
        assert "complex" in db
        assert "cofactor" in db
        assert "geneInfo" in db
        assert len(db["interaction"]) == 3233

    def test_mouse(self):
        db = load_cellchatdb("mouse")
        assert len(db["interaction"]) == 3379

    def test_zebrafish(self):
        db = load_cellchatdb("zebrafish")
        assert len(db["interaction"]) == 2774

    def test_interaction_columns(self):
        db = load_cellchatdb("human")
        df = db["interaction"]
        required_cols = ["interaction_name", "pathway_name", "ligand", "receptor"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_complex_has_names(self):
        db = load_cellchatdb("human")
        df = db["complex"]
        assert "name" in df.columns
        assert len(df) > 0
        # First entry should have at least one subunit
        first = df.iloc[0]
        assert pd.notna(first["subunit_1"]) and first["subunit_1"] != ""

    def test_cofactor_has_names(self):
        db = load_cellchatdb("human")
        df = db["cofactor"]
        assert "name" in df.columns

    def test_species_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_cellchatdb("nonexistent_species")
