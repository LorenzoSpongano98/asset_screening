import polars as pl
from datetime import date
from typing import Protocol


class DataProvider(Protocol):
    """Interface to abstract data fetching and standardization across different providers"""

    async def fetch(
        self, start: date, end: date, index_list: list[str]
    ) -> pl.DataFrame:
        """Fetch raw data for the given indices from start to end

        Parameters
        ----------
        start: date
            start of the time interval
        end: date
            end of the time interval
        index_list: list[str]
            list of indices

        Returns
        -------
        pl.DataFrame:
            polars dataframe for data handling
        """
        ...

    def standardize_schema(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Standardize fetched data for consistent
        storage across different providers

        Parameters
        ----------
        raw: pl.DataFrame
            raw data from provider

        Returns
        -------
        pl.DataFrame:
            standardized schema.
            Example of the standardized schema:

            -----------------------------------------------------------------------------------------
            | ProviderID |  Index  |  Timestamp  |  Open  |  Close   |  Low   |  High  |   Volume   |
            -----------------------------------------------------------------------------------------
            | 0000000001 |  AAPL  |  2006-09-10  | 6.1913 |  6.2044  | 6.1315 | 6.1638 | 396513600  |
            -----------------------------------------------------------------------------------------

        """
        ...
