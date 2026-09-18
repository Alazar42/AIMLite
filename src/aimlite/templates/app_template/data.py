"""AIMLite App: data.py

Define your application dataset ingestion here.
Just like Django models.Model, defining a Dataset subclass declares your data schema.
You can define multiple Dataset classes and use any format (CSV, JSON, JSONL, Parquet, TSV, etc.).
"""

from aimlite import Dataset


class AppDataset(Dataset):
    """Application dataset definition.

    Declare your dataset by specifying the filename inside data/:
        filename = "dataset.csv"

    You can define multiple Dataset classes, each with its own filename:
        class UserDataset(Dataset):
            filename = "users.csv"

        class TransactionDataset(Dataset):
            filename = "transactions.parquet"
    """

    filename = "dataset.csv"

