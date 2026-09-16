"""ModelKit App: data.py

Define your application dataset ingestion here.
Just like Django models.Model, defining a Dataset subclass declares your data schema.
You can define multiple Dataset classes and use any format (CSV, JSON, JSONL, Parquet, TSV, etc.).
"""

from modelkit import Dataset


class AppDataset(Dataset):
    """Application dataset definition.

    ModelKit automatically loads files from data/ matching conventions or explicit declarations:
        filename = "my_data.csv"      # or .json, .parquet, .tsv, .jsonl
        # or
        source = "data/custom_file.parquet"

    You can also define multiple Dataset classes (e.g. UserDataset, SalesDataset)
    and bind them to models via `dataset = UserDataset` in model.py.

    If you want to use your own tools (pandas, polars, etc.) to select or drop attributes:
        def load(self, source=None, **kwargs):
            import pandas as pd
            df = pd.read_csv(self._resolve_file_path(source))
            # Keep or drop columns as you wish:
            df = df.drop(columns=["id"], errors="ignore")
            self.columns = list(df.columns)
            self._data = df.to_dict(orient="records")
            return self._data
    """

    pass
