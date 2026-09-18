"""Verifies dataset integrity, schema readiness, and partition splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from cli.discovery import resolve_project_context
from cli.ui import C, arrow, check, cross, vite_header
from aimlite.data import Dataset

SUPPORTED_DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".tab",
    ".json",
    ".jsonl",
    ".parquet",
    ".pq",
    ".txt",
}


def run_data_validate(
    target: Optional[str] = None,
    project_root: Optional[Path] = None,
    **kwargs: Any,
) -> int:
    """Executes Dataset.load() and Dataset.validate(), outputting schema reports for registered datasets.

    Args:
        target: Optional Dataset class name or file name to selectively validate.
        project_root: Optional project root path.

    Returns:
        Exit code: 0 on pass, 1 on failure.
    """
    print(vite_header("data validate"))

    try:
        ctx = resolve_project_context(start_dir=project_root)
    except Exception as e:
        print(f"{cross(str(e))}\n")
        return 1

    # 1. Discover all data files in data/
    data_files: List[Path] = []
    if ctx.data_dir.is_dir():
        data_files = sorted([
            f for f in ctx.data_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
        ])

    if len(data_files) > 1:
        print(f"  {C.CYAN}Discovered {len(data_files)} data files in {ctx.data_dir.name}/:{C.RESET}")
        for df_path in data_files:
            size_kb = df_path.stat().st_size / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            print(f"    {C.DIM}•{C.RESET} {df_path.name} {C.DIM}({size_str}){C.RESET}")
        print()

    # 2. Resolve dataset classes to validate
    all_datasets = getattr(ctx, "dataset_classes", {}) or {}
    if not all_datasets and ctx.dataset_cls:
        all_datasets = {ctx.dataset_cls.__name__: ctx.dataset_cls}

    if target:
        # Filter by target class name or target filename
        target_lower = target.lower()
        matched = {}
        for name, cls in all_datasets.items():
            fn = getattr(cls, "filename", "") or ""
            if (
                name.lower() == target_lower
                or target_lower in name.lower()
                or fn.lower() == target_lower
                or Path(fn).name.lower() == target_lower
            ):
                matched[name] = cls
        if matched:
            datasets_to_validate = list(matched.values())
        else:
            print(f"{cross(f'No registered dataset matching \"{target}\".')}")
            if all_datasets:
                print(f"  {C.DIM}Available datasets: {', '.join(all_datasets.keys())}{C.RESET}\n")
            return 1
    else:
        # Prioritize custom user datasets over starter AppDataset
        custom_ds = {k: v for k, v in all_datasets.items() if k != "AppDataset"}
        if custom_ds:
            datasets_to_validate = list(custom_ds.values())
        elif all_datasets:
            datasets_to_validate = list(all_datasets.values())
        else:
            datasets_to_validate = [ctx.dataset_cls or Dataset]

    all_passed = True
    validated_any = False
    validated_files = set()

    for ds_cls in datasets_to_validate:
        cls_name = ds_cls.__name__
        try:
            dataset = ds_cls(name=f"{cls_name.lower()}_data", config=ctx.config)
        except Exception as e:
            print(f"{cross(f'Error instantiating {cls_name}: {e}')}\n")
            all_passed = False
            continue

        try:
            records = dataset.load()
            if records and not getattr(dataset, "_data", None):
                dataset._data = records
            is_valid = dataset.validate()
        except Exception as e:
            print(f"{cross(f'Error loading {cls_name}: {e}')}\n")
            all_passed = False
            continue

        resolved_file = getattr(dataset, "resolved_path", None)
        if not resolved_file and hasattr(dataset, "get_file_path"):
            resolved_file = dataset.get_file_path()

        if resolved_file and isinstance(resolved_file, Path):
            validated_files.add(resolved_file.name)
            try:
                rel_source = str(resolved_file.relative_to(ctx.root_dir))
            except (ValueError, AttributeError):
                rel_source = f"data/{resolved_file.name}"
        elif getattr(dataset, "filename", None):
            rel_source = f"data/{dataset.filename}"
        elif getattr(dataset, "source", None):
            rel_source = str(dataset.source)
        else:
            rel_source = "data/"

        if not is_valid or not records:
            has_files = bool(data_files)
            if not getattr(dataset, "filename", None) and not getattr(dataset, "source", None):
                print(f"{cross(f'{cls_name}: No filename declared.')}")
                print(f"  {C.YELLOW}Please set filename = \"<filename>\" on {cls_name} (e.g. filename = \"dataset.csv\").{C.RESET}")
            else:
                print(f"{cross(f'{cls_name}: No valid records found in {rel_source}.')}")
                if not has_files:
                    print(f"  {C.DIM}No supported data files found in {ctx.data_dir}.{C.RESET}")
                else:
                    print(f"  {C.YELLOW}Ensure your data file contains valid rows and headers (CSV, JSON, Parquet, etc.).{C.RESET}")
            all_passed = False
            continue

        validated_any = True
        total_records = len(records) if hasattr(records, "__len__") else len(dataset)

        # Generate partition breakdown
        try:
            train, val, test = dataset.split()
            train_count = len(train)
            val_count = len(val)
            test_count = len(test)
        except Exception:
            train_count = total_records
            val_count = 0
            test_count = 0

        col_names = getattr(dataset, "columns", [])
        col_str = f"({', '.join(col_names[:6])}{'...' if len(col_names) > 6 else ''})" if col_names else "(none)"
        partitions_str = f"train: {train_count:,} | val: {val_count:,} | test: {test_count:,}"

        print(arrow("Dataset", f"{cls_name} {C.DIM}({rel_source}){C.RESET}"))
        print(arrow("Records", f"{total_records:,} rows"))
        print(arrow("Columns", f"{len(col_names)} {C.DIM}{col_str}{C.RESET}"))
        print(arrow("Partitions", partitions_str))
        print()

    # If multiple files exist in data/ and some were not validated, show helpful hint
    unmapped_files = [f for f in data_files if f.name not in validated_files]
    if unmapped_files and not target:
        unmapped_names = ", ".join(f.name for f in unmapped_files)
        print(f"  {C.DIM}Note: Other data files in data/: {unmapped_names}")
        print(f"  To bind a file, set filename = \"<filename>\" on a Dataset class.{C.RESET}\n")

    if all_passed and validated_any:
        print(f"{check('Dataset validation passed.')}\n")
        return 0

    return 1
