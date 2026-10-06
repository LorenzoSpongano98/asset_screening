import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

import polars as pl

from .capabilities import ProviderCapabilities


class SymbolErrorType(enum.Enum):
    TEMPORARY = enum.auto()
    PERMANENT = enum.auto()


@dataclass(frozen=True, slots=True)
class SymbolError:
    symbol: str
    error_type: SymbolErrorType
    message: str


@dataclass(frozen=True, slots=True)
class Payload:
    content: bytes
    content_type: str
    request_parameters: Mapping[str, str]
    received_at: datetime

    def __post_init__(self):
        if self.received_at.utcoffset() != timedelta(0):
            raise ValueError(
                f"received_at must be an UTC datetime. Got {self.received_at!r}"
            )


@dataclass(frozen=True, slots=True)
class FetchResult:
    payloads: tuple[Payload, ...]
    symbol_errors: tuple[SymbolError, ...]


class DataProvider(Protocol):
    """Interface to abstract data fetching and standardization across different providers"""

    provider_code: str

    def fetch(self, start: date, end: date, symbols: Sequence[str]) -> FetchResult:
        # TODO: Update the docstring
        """Fetch raw data from the given data provider for the given symbols from start to end

        Parameters
        ----------
        start: date
            start of the time interval
        end: date
            end of the time interval
        symbols: Sequence[str]
            sequence (list, tuple) of symbols

        Returns
        -------

        """
        ...

    def parse(self, payload: Payload) -> pl.DataFrame:
        # TODO: Update the docstring
        ...

    def standardize(self, parsed: pl.DataFrame) -> pl.DataFrame:
        # TODO: Update the docstring
        """Standardize raw data for consistent
        storage across different providers

        Parameters
        ----------
        raw: pl.DataFrame
            raw data from provider

        Returns
        -------
        pl.DataFrame:
            standardized schema.

        """
        ...

    def capabilities(self) -> ProviderCapabilities:
        # TODO: Update the docstring
        ...
