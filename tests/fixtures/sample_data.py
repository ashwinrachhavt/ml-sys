"""Sample fixtures for unit tests."""

from __future__ import annotations

import pandas as pd


def make_customer_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "is_customer": [1, 0, 1],
            "INDUSTRY": ["Tech", "Retail", "Finance"],
            "CLOSEDATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )


def make_usage_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 1, 2],
            "ACTIONS_CRM_CONTACTS": [5, 2, 3],
            "USERS_EMAIL": [10, 7, 4],
        }
    )
