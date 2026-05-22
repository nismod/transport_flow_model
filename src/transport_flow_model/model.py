"""Core model classes"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class OD:
    """Origin-destination flows.

    - sparse matrix representation
    - works with methods to estimate OD from spatial interactions
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized tabular data."""
        return self._data.copy()

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the normalized tabular data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    @classmethod
    def from_csv(cls, path: str | Path, column_map: dict[str, str]) -> OD:
        """Load a CSV and rename source columns to this class' schema."""
        target_columns = set(column_map.values())
        to_rename = set(column_map.keys())
        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS if col not in target_columns
        ]
        if missing_columns:
            raise ValueError(
                "column_map must include mappings for required columns "
                f"{missing_columns}"
            )

        try:
            data = pd.read_csv(path, usecols=to_rename)
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
        data = data.rename(columns=column_map)

        return cls(data)


class Network:
    """Base network model class.

    - graph representation
    - works with methods to allocate OD flows to the network
    """

    REQUIRED_COLUMNS = ("edge_from", "edge_to", "edge_id")
    OPTIONAL_COLUMNS = ("capacity", "cost", "flow")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the normalized tabular data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    @classmethod
    def from_csv(cls, path: str | Path, column_map: dict[str, str]) -> Network:
        """Load a CSV and rename source columns to this class' schema."""
        target_columns = set(column_map.values())
        to_rename = set(column_map.keys())
        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS if col not in target_columns
        ]
        if missing_columns:
            raise ValueError(
                "column_map must include mappings for required columns "
                f"{missing_columns}"
            )

        data = pd.read_csv(path, usecols=to_rename)
        data = data.rename(columns=column_map)

        return cls(data)


class ODFlows:
    """Origin-destination flow paths

    - full path representation of allocated flows for each OD
    - may include multiple paths for a single OD pair with different capacity
      allocation
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return OD flow paths data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    def to_csv(self, path: str | Path, index=False):
        """Write OD flow paths to CSV."""
        self._data.to_csv(path, index=index)


class NetworkFlows:
    """Aggregate flows on network

    - could consider this as Network with calculated attributes
      as the result of an allocation
    - multiple attributes may include total allocation, speed, cost, ...
    """
