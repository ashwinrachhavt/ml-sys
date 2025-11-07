"""Synthetic fallback datasets for tests and CI runs."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


CUSTOMER_ROWS = [
    {
        "id": 1,
        "EMPLOYEE_RANGE": "50-100",
        "INDUSTRY": "Software",
        "COUNTRY": "USA",
        "ALEXA_RANK": 150_000,
        "CLOSEDATE": "2023-01-15",
        "MRR": 5000,
    },
    {
        "id": 2,
        "EMPLOYEE_RANGE": "10-50",
        "INDUSTRY": "Healthcare",
        "COUNTRY": "USA",
        "ALEXA_RANK": 325_000,
        "CLOSEDATE": "2022-11-03",
        "MRR": 2500,
    },
    {
        "id": 3,
        "EMPLOYEE_RANGE": "100-500",
        "INDUSTRY": "Finance",
        "COUNTRY": "Canada",
        "ALEXA_RANK": 475_000,
        "CLOSEDATE": "2023-03-20",
        "MRR": 8000,
    },
]

NONCUSTOMER_ROWS = [
    {
        "id": 4,
        "EMPLOYEE_RANGE": "10-50",
        "INDUSTRY": "Software",
        "COUNTRY": "USA",
        "ALEXA_RANK": 600_000,
        "CLOSEDATE": None,
        "MRR": 0,
    },
    {
        "id": 5,
        "EMPLOYEE_RANGE": "1-10",
        "INDUSTRY": "Education",
        "COUNTRY": "USA",
        "ALEXA_RANK": 1_200_000,
        "CLOSEDATE": None,
        "MRR": 0,
    },
    {
        "id": 6,
        "EMPLOYEE_RANGE": "50-100",
        "INDUSTRY": "Manufacturing",
        "COUNTRY": "Germany",
        "ALEXA_RANK": 900_000,
        "CLOSEDATE": None,
        "MRR": 0,
    },
]

USAGE_ROWS = [
    {
        "id": 1,
        "ACTIONS_login": 15,
        "ACTIONS_invite": 3,
        "USERS_active": 10,
        "USERS_admin": 2,
    },
    {
        "id": 2,
        "ACTIONS_login": 5,
        "ACTIONS_invite": 1,
        "USERS_active": 4,
        "USERS_admin": 1,
    },
    {
        "id": 3,
        "ACTIONS_login": 20,
        "ACTIONS_invite": 7,
        "USERS_active": 15,
        "USERS_admin": 4,
    },
    {
        "id": 4,
        "ACTIONS_login": 2,
        "ACTIONS_invite": 0,
        "USERS_active": 1,
        "USERS_admin": 0,
    },
    {
        "id": 6,
        "ACTIONS_login": 1,
        "ACTIONS_invite": 0,
        "USERS_active": 1,
        "USERS_admin": 0,
    },
]


def load_stub_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return synthetic customer, noncustomer and usage tables."""

    customers = pd.DataFrame(CUSTOMER_ROWS)
    noncustomers = pd.DataFrame(NONCUSTOMER_ROWS)
    usage = pd.DataFrame(USAGE_ROWS)
    return customers, noncustomers, usage


__all__ = ["load_stub_tables"]
