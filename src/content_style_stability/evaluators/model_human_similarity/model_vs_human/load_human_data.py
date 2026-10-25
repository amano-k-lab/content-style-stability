"""Load human response data for model-vs-human evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from .constants import DATASETS, HUMAN_DATA_DIR


def load_human_data(
    root: Optional[Path | str] = None,
    datasets: Optional[Sequence[str]] = None,
    filename_transform: Optional[Callable[[str], str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load human response data per dataset as DataFrames.

    Expected directory layout::

        <root>/
            <dataset_name>/
                *.csv   # one or more CSV files

    Each CSV is read with pandas; all CSVs in a dataset directory are
    concatenated into one DataFrame (columns as in the CSV).

    Parameters
    ----------
    root : Path | str, optional
        Root directory of human data.
    datasets : Sequence[str], optional
        List of dataset names to load.
    filename_transform : callable (filename: str) -> str, optional
        Called on each value in the ``filename`` column (CSV may contain path
        or filename strings). Return value is used as the normalized filename
        for merging with model predictions. Must accept a single string and
        return a single string.

    Returns
    -------
    dict[str, pd.DataFrame]
        dataset_name -> DataFrame (CSV columns preserved)
    """
    root = Path(HUMAN_DATA_DIR if root is None else root)
    if not root.exists():
        raise FileNotFoundError(f"Human data root not found: {root}")
    dataset_names = list(datasets) if datasets is not None else list(DATASETS)
    out: dict[str, pd.DataFrame] = {}

    for name in dataset_names:
        dir_path = root / name
        if not dir_path.is_dir():
            out[name] = pd.DataFrame()
            continue
        dfs = [
            pd.read_csv(
                csv_path,
                encoding="utf-8",
                dtype=str,
            )
            for csv_path in sorted(dir_path.glob("*.csv"))
        ]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        df.rename(
            columns={
                "subj": "subject",
                "imagename": "filename",
                "category": "label",
                "object_response": "prediction",
            },
            inplace=True,
        )
        df = df[
            [
                c
                for c in ["subject", "condition", "filename", "prediction", "label"]
                if c in df.columns
            ]
        ]
        if filename_transform is not None:
            if "filename" not in df.columns:
                raise KeyError(
                    "filename column is required when filename_transform is provided"
                )
            df["filename"] = df["filename"].apply(filename_transform)
        out[name] = df

    return out
