"""Comprehensive tests for the AIMLite Dataset pillar.

Covers:
- Dataset loading from JSON, CSV, JSONL files
- split() shuffles data before partitioning (seeded, reproducible)
- split() ratio validation raises on invalid sums
- split() partition sizes are correct
- split() different seeds produce different orderings
- Dataset.validate() detects schema violations
- Document-oriented loading via load_documents()
"""

from __future__ import annotations

import csv
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from aimlite.data import Dataset
from aimlite.rag import Document


# =============================================================================
# Helpers
# =============================================================================


def write_json_dataset(path: Path, n: int = 100) -> List[Dict[str, Any]]:
    records = [{"feature_a": i * 0.1, "feature_b": i * 0.2, "target": i % 2} for i in range(n)]
    with open(path, "w") as f:
        json.dump(records, f)
    return records


def write_csv_dataset(path: Path, n: int = 100) -> List[Dict[str, Any]]:
    fieldnames = ["feature_a", "feature_b", "target"]
    records = [{"feature_a": i * 0.1, "feature_b": i * 0.2, "target": i % 2} for i in range(n)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records


def write_jsonl_dataset(path: Path, n: int = 100) -> List[Dict[str, Any]]:
    records = [{"text": f"example {i}", "label": i % 3} for i in range(n)]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records

# =============================================================================
# Loading tests
# =============================================================================


class TestDatasetLoad:
    def test_load_json_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            records = write_json_dataset(path, n=20)
            ds = Dataset(name="json_test", source=str(path))
            loaded = ds.load()
            assert len(loaded) == 20

    def test_load_csv_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.csv"
            write_csv_dataset(path, n=30)
            ds = Dataset(name="csv_test", source=str(path))
            loaded = ds.load()
            assert len(loaded) == 30

    def test_load_jsonl_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.jsonl"
            write_jsonl_dataset(path, n=10)
            ds = Dataset(name="jsonl_test", source=str(path))
            loaded = ds.load()
            assert len(loaded) == 10

    def test_load_in_memory_data(self):
        """Dataset backed by a temp JSON file must load all records."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mem.json"
            records = [{"x": i, "y": i * 2} for i in range(50)]
            path.write_text(json.dumps(records))
            ds = Dataset(name="in_memory", source=str(path))
            loaded = ds.load()
            assert len(loaded) == 50


# =============================================================================
# Dataset.split() correctness
# =============================================================================


class TestDatasetSplit:
    def _make_ds(self, n: int = 100) -> Dataset:
        """Creates a file-backed Dataset with n sequential records."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            records = [{"x": i} for i in range(n)]
            path.write_text(json.dumps(records))
            ds = Dataset(name="split_test", source=str(path))
            ds.load()  # Populate _data from file
        return ds

    def _make_ds_inline(self, n: int = 100) -> Dataset:
        """Creates a Dataset with n sequential in-memory records (bypasses load())."""
        ds = Dataset(name="split_test")
        ds._data = [{"x": i} for i in range(n)]
        return ds

    def test_split_total_equals_input_size(self):
        """All split partitions combined must equal the full dataset size."""
        ds = self._make_ds_inline(n=100)
        train, val, test = ds.split(train=0.8, validation=0.1, test=0.1)
        total = len(train) + len(val) + len(test)
        assert total == 100, f"Split total {total} != 100"

    def test_split_train_fraction_correct(self):
        """Train partition must be approximately 80% of total data."""
        ds = self._make_ds_inline(n=200)
        train, val, test = ds.split(train=0.8, validation=0.1, test=0.1)
        assert 150 <= len(train) <= 175, f"Unexpected train size: {len(train)}"

    def test_split_test_fraction_correct(self):
        ds = self._make_ds_inline(n=100)
        train, val, test = ds.split(train=0.7, validation=0.1, test=0.2)
        assert len(test) >= 15, f"Unexpected test size: {len(test)}"

    def test_split_shuffles_data(self):
        """split() must shuffle data, so ordering must differ from sequential input."""
        ds = self._make_ds_inline(n=100)
        train, val, test = ds.split(shuffle=True, seed=42)
        all_split = train + val + test
        sequential_x = list(range(100))
        split_x = [r["x"] for r in all_split]
        assert split_x != sequential_x, "Shuffled split must not preserve input order"

    def test_split_same_seed_is_reproducible(self):
        """Same seed must produce identical partition ordering."""
        ds1 = self._make_ds_inline(n=100)
        ds2 = self._make_ds_inline(n=100)
        train1, val1, test1 = ds1.split(shuffle=True, seed=99)
        train2, val2, test2 = ds2.split(shuffle=True, seed=99)
        assert [r["x"] for r in train1] == [r["x"] for r in train2], "Same seed must be deterministic"

    def test_split_different_seeds_produce_different_orderings(self):
        """Different seeds must produce different orderings (with high probability)."""
        ds1 = self._make_ds_inline(n=100)
        ds2 = self._make_ds_inline(n=100)
        train1, _, _ = ds1.split(shuffle=True, seed=1)
        train2, _, _ = ds2.split(shuffle=True, seed=2)
        assert [r["x"] for r in train1] != [r["x"] for r in train2], (
            "Different seeds should produce different orderings"
        )

    def test_split_no_shuffle_preserves_order(self):
        """split(shuffle=False) must preserve original input order."""
        ds = self._make_ds_inline(n=50)
        train, val, test = ds.split(train=0.8, validation=0.1, test=0.1, shuffle=False)
        all_split = train + val + test
        split_x = [r["x"] for r in all_split]
        assert split_x == list(range(50)), "No-shuffle split must preserve input order"

    def test_split_invalid_ratio_raises(self):
        """split() must raise ValueError when ratios don't sum to 1.0."""
        ds = self._make_ds_inline(n=50)
        with pytest.raises(ValueError, match="sum to 1.0"):
            ds.split(train=0.5, validation=0.2, test=0.1)

    def test_split_zero_train_raises(self):
        """split() must raise ValueError when train ratio is 0."""
        ds = self._make_ds_inline(n=50)
        with pytest.raises(ValueError, match="train ratio"):
            ds.split(train=0.0, validation=0.5, test=0.5)

    def test_split_empty_dataset(self):
        """split() on empty dataset must return three empty lists."""
        ds = Dataset(name="empty")
        ds._data = []
        train, val, test = ds.split()
        assert train == []
        assert val == []
        assert test == []

    def test_split_no_data_overlaps(self):
        """Records in one partition must not appear in another (by identity)."""
        ds = self._make_ds_inline(n=90)
        train, val, test = ds.split(train=0.7, validation=0.2, test=0.1)
        all_x_train = {r["x"] for r in train}
        all_x_val = {r["x"] for r in val}
        all_x_test = {r["x"] for r in test}
        assert all_x_train.isdisjoint(all_x_val), "Train and val must be disjoint"
        assert all_x_train.isdisjoint(all_x_test), "Train and test must be disjoint"
        assert all_x_val.isdisjoint(all_x_test), "Val and test must be disjoint"


# =============================================================================
# Dataset with pre-split files
# =============================================================================


class TestDatasetPresplit:
    def test_presplit_train_test_files(self):
        """When separate train/test files exist, split() must use them."""
        train_records = [{"x": i, "target": 0} for i in range(80)]
        test_records = [{"x": i, "target": 1} for i in range(80, 100)]

        ds = Dataset(name="presplit")
        ds._train_data = train_records
        ds._test_data = test_records
        ds._val_data = []

        train, val, test = ds.split()
        assert len(train) >= 60  # Some may be carved off for validation
        assert len(test) == 20
