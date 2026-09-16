"""Verifies dataset integrity, schema readiness, and partition splits."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cli.discovery import resolve_project_context
from cli.ui import C, arrow, check, cross, vite_header
from modelkit.data import Dataset


def run_data_validate(project_root: Optional[Path] = None, **kwargs: Any) -> int:
    """Executes Dataset.load() and Dataset.validate(), outputting schema reports for all registered datasets.

    Args:
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

    # Check if multiple dataset classes are registered in data.py
    datasets_to_validate = (
        list(ctx.dataset_classes.values())
        if getattr(ctx, "dataset_classes", None)
        else [ctx.dataset_cls or Dataset]
    )

    all_passed = True
    validated_any = False

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
            is_valid = dataset.validate()
        except Exception as e:
            print(f"{cross(f'Error loading {cls_name}: {e}')}\n")
            all_passed = False
            continue

        if not is_valid or not records:
            has_files = ctx.data_dir.is_dir() and any(
                f.is_file() and not f.name.startswith(".") for f in ctx.data_dir.iterdir()
            )
            target_fn = getattr(dataset, "filename", None) or getattr(dataset, "source", None) or "data/"
            print(f"{cross(f'{cls_name}: No valid records found in {target_fn}.')}")
            if not has_files:
                print(f"  {C.DIM}No data files found in {ctx.data_dir}.{C.RESET}")
            else:
                print(f"  {C.YELLOW}Ensure the file contains valid rows and headers (CSV, JSON, Parquet, etc.).{C.RESET}")
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

        resolved_source = getattr(dataset, "filename", None) or getattr(dataset, "source", None) or "data/"
        print(arrow("Dataset", f"{cls_name} {C.DIM}({resolved_source}){C.RESET}"))
        print(arrow("Records", f"{total_records:,} rows"))
        print(arrow("Columns", f"{len(col_names)} {C.DIM}{col_str}{C.RESET}"))
        print(arrow("Partitions", partitions_str))
        print()

    if all_passed and validated_any:
        print(f"{check('Dataset validation passed.')}\n")
        return 0

    return 1
