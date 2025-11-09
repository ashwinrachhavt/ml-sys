from pathlib import Path

import pandas as pd

from mlsys.data import DataLoaderRegistry, DatasetLoader


def test_csv_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame({"ID": [1], "is_customer": [1]}).to_csv(csv_path, index=False)

    loader = DataLoaderRegistry.create("csv")
    df = loader.load(csv_path)
    assert not df.empty


def test_dataset_loader_multiple(tmp_path: Path) -> None:
    customers = tmp_path / "customers.csv"
    usage = tmp_path / "usage.csv"
    pd.DataFrame({"ID": [1], "is_customer": [1]}).to_csv(customers, index=False)
    pd.DataFrame({"ID": [1], "ACTIONS": [5]}).to_csv(usage, index=False)

    dataset_loader, datasets = DatasetLoader.from_config(
        "csv",
        {"customers": customers, "usage_actions": usage},
    )
    assert set(datasets) == {"customers", "usage_actions"}
    assert dataset_loader.loader is not None
