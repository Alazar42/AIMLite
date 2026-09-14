"""ModelKit App: data.py

Define your application dataset ingestion here.
Just like Django models.Model, defining an empty Dataset does nothing until load() or train() is called.
"""

from modelkit import Dataset


class AppDataset(Dataset):
    """Application dataset definition.

    Declare your source or filename attribute:
        filename = "data.csv"
        # or
        source = "data/custom_file.csv"
    """

    pass
