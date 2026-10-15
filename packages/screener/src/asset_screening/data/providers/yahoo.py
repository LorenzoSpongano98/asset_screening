import io
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import polars as pl
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

from asset_screening.data.capabilities import AdjustmentMethod, ProviderCapabilities
from asset_screening.data.contracts import (
    DataProvider,
    FetchResult,
    Payload,
    SymbolError,
    SymbolErrorType,
    UnavailableProviderError,
)


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance provider using yfinance."""

    provider_code = "yahoo"

    _CAPABILITIES = ProviderCapabilities(
        adjustment_method=AdjustmentMethod.SPLIT,
        currencies=frozenset({"USD", "EUR", "GBP"}),
        markets=frozenset({"XNYS", "XNAS", "XMIL", "XPAR", "XETR", "XLON"}),
        max_symbols_per_request=50,
        max_days_per_request=None,
        history_start=None,
        min_request_interval_s=1.0,
    )

    def __init__(self):
        yf.config.debug.hide_exceptions = False

    def capabilities(self) -> ProviderCapabilities:
        return self._CAPABILITIES

    @staticmethod
    def _to_parquet(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        df.reset_index().to_parquet(buf)
        return buf.getvalue()

    def fetch(self, start: date, end: date, symbols: Sequence[str]) -> FetchResult:
        if len(symbols) > self.capabilities().max_symbols_per_request:
            raise ValueError(
                f"Can process up to {self.capabilities().max_symbols_per_request} symbols per requests. Got {len(symbols)}"
            )

        payloads: list[Payload] = []
        errors: list[SymbolError] = []

        for i, symbol in enumerate(symbols):
            if i > 0:
                time.sleep(self.capabilities().min_request_interval_s)
            try:
                res = yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )
            except YFRateLimitError as exc:
                errors.append(
                    SymbolError(
                        symbol=symbol,
                        error_type=SymbolErrorType.TEMPORARY,
                        message=str(exc),
                    )
                )
                errors.extend(
                    SymbolError(
                        symbol=skipped,
                        error_type=SymbolErrorType.TEMPORARY,
                        message="Reached rate limit. Not attempted",
                    )
                    for skipped in symbols[i + 1 :]
                )
                break
            except (YFTzMissingError, YFPricesMissingError) as exc:
                errors.append(
                    SymbolError(
                        symbol=symbol,
                        error_type=SymbolErrorType.PERMANENT,
                        message=str(exc),
                    )
                )
                continue

            received_at = datetime.now(UTC)

            if res.empty:
                errors.append(
                    SymbolError(
                        symbol=symbol,
                        error_type=SymbolErrorType.PERMANENT,
                        message=f"empty response from {symbol}",
                    )
                )
                continue

            payloads.append(
                Payload(
                    content=self._to_parquet(df=res),
                    content_type="application/vnd.apache.parquet",
                    request_parameters={
                        "symbol": symbol,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                    received_at=received_at,
                )
            )

        if (
            not payloads
            and errors
            and all(e.error_type is SymbolErrorType.TEMPORARY for e in errors)
        ):
            raise UnavailableProviderError(
                f"{self.provider_code}: all {len(errors)} requests failed temporarily"
            )
        return FetchResult(payloads=tuple(payloads), symbol_errors=tuple(errors))

    def parse(self, payload: Payload) -> pl.DataFrame:
        raise NotImplementedError("WIP, to be implemented soon.")

    def standardize(self, parsed: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("WIP, to be implemented soon.")
